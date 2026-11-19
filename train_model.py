"""
Machine Learning Training Pipeline for Fake News Detection System.

Performs data sanitization, leakage prevention, and TF-IDF n-gram vectorization.
Trains and compares:
  1. Logistic Regression (Primary Classifier)
  2. Passive-Aggressive-style Linear Classifier (SGD with hinge loss)
  3. Support Vector Machine (LinearSVC)

Computes comprehensive metrics including separate class-level evaluation
(Precision, Recall, F1-Score for FAKE and REAL classes separately) and confusion matrix.
Saves model artifacts to:
  - model/fake_news_model.pkl
  - model/tfidf_vectorizer.pkl
  - model/model_metrics.json
"""

import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Import shared text preprocessor
from preprocessor import clean_text


def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    model_dir = os.path.join(base_dir, "model")
    fake_csv = os.path.join(dataset_dir, "Fake.csv")
    true_csv = os.path.join(dataset_dir, "True.csv")
    return base_dir, dataset_dir, model_dir, fake_csv, true_csv


def check_dataset_files(fake_csv, true_csv):
    """
    Verifies that Fake.csv and True.csv exist in the dataset directory.
    Does NOT automatically generate synthetic data.
    """
    if not os.path.exists(fake_csv) or not os.path.exists(true_csv):
        print("=" * 80)
        print("[ERROR] Required dataset files were not found in the 'dataset/' folder:")
        if not os.path.exists(fake_csv):
            print(f"  Missing: {fake_csv}")
        if not os.path.exists(true_csv):
            print(f"  Missing: {true_csv}")
        print("\nPlease download the official Kaggle Fake and Real News Dataset from:")
        print("  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("Extract and place 'Fake.csv' and 'True.csv' into the 'dataset/' folder.")
        print("\n(Note: For quick offline testing, you can generate an initial demonstration")
        print(" corpus by running: python dataset/generate_sample_data.py)")
        print("=" * 80)
        sys.exit(1)


def load_and_preprocess_data(fake_csv, true_csv):
    """
    Loads Fake and Real news datasets, checks for missing/empty values,
    removes duplicate and near-duplicate articles to prevent data leakage,
    applies clean_text normalization, and assigns binary labels (0 = FAKE, 1 = REAL).
    """
    print("\n[STEP 1/5] Loading and Inspecting Datasets...")
    fake_df = pd.read_csv(fake_csv)
    true_df = pd.read_csv(true_csv)

    print(f"  - Loaded Fake News records : {len(fake_df):,}")
    print(f"  - Loaded Genuine News records: {len(true_df):,}")

    # Check for missing values in raw files
    print(f"  - Fake.csv missing values: title={fake_df['title'].isnull().sum()}, text={fake_df['text'].isnull().sum()}")
    print(f"  - True.csv missing values: title={true_df['title'].isnull().sum()}, text={true_df['text'].isnull().sum()}")

    # Assign binary labels: 0 = FAKE, 1 = REAL
    fake_df["label"] = 0
    true_df["label"] = 1

    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    initial_count = len(df)
    print(f"  - Total combined raw records: {initial_count:,}")

    # Fill missing values
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title and article body for rich contextual patterns
    df["full_content"] = df["title"] + " " + df["text"]

    print("\n[STEP 2/5] Cleaning and Leakage Prevention...")
    # Drop records where full_content is blank
    df = df[df["full_content"].str.strip() != ""].reset_index(drop=True)

    # 1. Deduplicate by exact raw content
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["full_content"]).reset_index(drop=True)
    exact_dups = before_dedup - len(df)

    # 2. Deduplicate identical titles (often syndicated reprints with minor whitespace edits)
    before_title_dedup = len(df)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    title_dups = before_title_dedup - len(df)
    print(f"  - Removed {exact_dups:,} exact content duplicates and {title_dups:,} duplicate-title stories.")

    # 3. Clean text using the shared preprocessor
    print("  - Normalizing text & stripping news-wire headers (e.g. 'Reuters', 'AP')...")
    start_clean = time.time()
    df["cleaned_text"] = df["full_content"].apply(clean_text)

    # 4. Deduplicate on cleaned text (catches near-duplicates differing only in punctuation/html)
    before_clean_dedup = len(df)
    df = df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    clean_dups = before_clean_dedup - len(df)
    if clean_dups > 0:
        print(f"  - Removed {clean_dups:,} near-duplicate records identified after normalization.")

    # 5. Filter out entries with insufficient cleaned content
    df = df[df["cleaned_text"].str.strip().str.len() >= 15].reset_index(drop=True)
    print(f"  - Text normalization completed in {time.time() - start_clean:.2f} seconds.")

    usable_total = len(df)
    real_count = int((df["label"] == 1).sum())
    fake_count = int((df["label"] == 0).sum())

    print(f"  - Final usable dataset: {usable_total:,} articles")
    print(f"    * Genuine News (1): {real_count:,} articles ({real_count / usable_total * 100:.1f}%)")
    print(f"    * Fake News    (0): {fake_count:,} articles ({fake_count / usable_total * 100:.1f}%)")

    # Dataset size check: Alert if working on small demonstration data
    is_sample_dataset = usable_total < 500
    if is_sample_dataset:
        print("\n" + "!" * 80)
        print("[WARNING] Small demonstration dataset detected (fewer than 500 articles).")
        print(f"  Current size: {usable_total} articles.")
        print("  Notice: Machine learning models trained on small corpora will show artificially")
        print("  high evaluation scores (e.g. 100% accuracy) due to limited vocabulary diversity.")
        print("  To evaluate realistic generalizability, place the full Kaggle dataset")
        print("  (44,898 articles) into the 'dataset/' directory.")
        print("!" * 80)

    return df, is_sample_dataset


def split_and_vectorize(df):
    """
    Performs a stratified train-test split (80-20) and extracts TF-IDF n-gram features.
    Vectorization parameters:
      - max_features = 5000
      - ngram_range = (1, 2) [unigrams and bigrams]
      - stop_words = 'english'
      - sublinear_tf = True
    """
    print("\n[STEP 3/5] Performing Stratified Train-Test Split and TF-IDF Vectorization...")

    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"  - Training samples : {len(X_train):,} (80%)")
    print(f"  - Testing samples  : {len(X_test):,} (20%)")

    vectorizer = TfidfVectorizer(
        max_features=25000,
        min_df=2,
        max_df=0.8,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    print("  - Fitting TF-IDF Vectorizer ONLY on training data (no leakage)...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"  - Vocabulary size : {len(vectorizer.vocabulary_):,} features")
    print(f"  - X_train shape   : {X_train_tfidf.shape}")
    print(f"  - X_test shape    : {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains and benchmarks:
      1. Logistic Regression (Primary algorithm)
      2. SGDClassifier (Passive-Aggressive linear style with hinge loss)
      3. Linear Support Vector Machine (LinearSVC)

    Calculates overall and separate class-level metrics for FAKE and REAL classes.
    """
    print("\n[STEP 4/5] Training Machine Learning Models and Computing Evaluation Metrics...")

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=2.0, class_weight="balanced", random_state=42, solver="lbfgs"
        ),
        "Passive-Aggressive Linear (SGD)": SGDClassifier(
            loss="modified_huber", penalty="l2", alpha=1e-4, class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Linear Support Vector Machine": LinearSVC(
            random_state=42, max_iter=2000, class_weight="balanced", dual="auto"
        )
    }

    results = {}
    best_model_name = "Logistic Regression"  # Primary standard algorithm
    best_model_obj = None

    print("-" * 88)
    print(f"{'Model Name':<34} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
    print("-" * 88)

    for name, clf in classifiers.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Detailed class-wise classification report
        report_dict = classification_report(
            y_test, y_pred, target_names=["FAKE", "REAL"], output_dict=True, zero_division=0
        )

        fake_prec = float(report_dict["FAKE"]["precision"])
        fake_rec = float(report_dict["FAKE"]["recall"])
        fake_f1 = float(report_dict["FAKE"]["f1-score"])
        fake_supp = int(report_dict["FAKE"]["support"])

        real_prec = float(report_dict["REAL"]["precision"])
        real_rec = float(report_dict["REAL"]["recall"])
        real_f1 = float(report_dict["REAL"]["f1-score"])
        real_supp = int(report_dict["REAL"]["support"])

        macro_f1 = float(report_dict["macro avg"]["f1-score"])
        weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
            "fake_class": {
                "precision": round(fake_prec * 100, 2),
                "recall": round(fake_rec * 100, 2),
                "f1_score": round(fake_f1 * 100, 2),
                "support": fake_supp
            },
            "real_class": {
                "precision": round(real_prec * 100, 2),
                "recall": round(real_rec * 100, 2),
                "f1_score": round(real_f1 * 100, 2),
                "support": real_supp
            },
            # Overall summary metrics for dashboard display
            "precision": round(real_prec * 100, 2),
            "recall": round(real_rec * 100, 2),
            "f1_score": round(weighted_f1 * 100, 2),
            "confusion_matrix": cm,
            "train_time_sec": round(train_time, 3)
        }

        print(f"{name:<34} {acc*100:>8.2f}% {real_prec*100:>10.2f}% {real_rec*100:>10.2f}% {weighted_f1*100:>8.2f}%")

        if name == "Logistic Regression":
            best_model_obj = clf

    print("-" * 88)
    print(f"[INFO] Primary Deployed Classifier: '{best_model_name}' (TF-IDF + Logistic Regression)")

    # Print Detailed Classification Report
    lr_pred = best_model_obj.predict(X_test)
    print("\nDetailed Per-Class Classification Report for Logistic Regression:")
    print(classification_report(y_test, lr_pred, target_names=["FAKE (Class 0)", "REAL (Class 1)"], zero_division=0))

    return best_model_name, best_model_obj, results


def save_artifacts(best_model_name, best_model_obj, vectorizer, results, df, is_sample_dataset, model_dir, X_train_shape, X_test_shape):
    """
    Saves trained model, TF-IDF vectorizer, and evaluation metrics JSON.
    """
    print("\n[STEP 5/5] Saving Trained Model, Vectorizer, and Evaluation Metrics...")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "fake_news_model.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    metrics_path = os.path.join(model_dir, "model_metrics.json")

    # Serialize trained model and vectorizer using joblib
    joblib.dump(best_model_obj, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    best_metrics = results[best_model_name]
    cm = best_metrics["confusion_matrix"]
    # cm is [[TN, FP], [FN, TP]]
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

    metrics_payload = {
        "status": "ready",
        "selected_model": best_model_name,
        "algorithm_family": "TF-IDF Vectorization with Logistic Regression",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_sample_dataset": is_sample_dataset,
        "sample_dataset_warning": (
            "Notice: Evaluated on a starter demonstration dataset. "
            "For full production evaluation, train on the complete Kaggle Fake and Real News dataset."
            if is_sample_dataset else None
        ),
        "dataset": {
            "total_records": len(df),
            "real_records": int((df["label"] == 1).sum()),
            "fake_records": int((df["label"] == 0).sum()),
            "training_samples": X_train_shape[0],
            "testing_samples": X_test_shape[0],
            "features_extracted": X_train_shape[1],
            "leakage_protection": "Stripped Reuters and news-wire header tags + deduplication"
        },
        "performance": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1_score"],
            "macro_f1": best_metrics["macro_f1"],
            "weighted_f1": best_metrics["weighted_f1"]
        },
        # Separate class metrics for FAKE and REAL
        "class_metrics": {
            "fake_class": best_metrics["fake_class"],
            "real_class": best_metrics["real_class"]
        },
        "confusion_matrix": {
            "matrix": cm,
            "true_negatives_fake_correct": tn,
            "false_positives_fake_as_real": fp,
            "false_negatives_real_as_fake": fn,
            "true_positives_real_correct": tp
        },
        "all_models_comparison": results
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"  [SUCCESS] Model saved to      : {model_path}")
    print(f"  [SUCCESS] Vectorizer saved to : {vectorizer_path}")
    print(f"  [SUCCESS] Metrics saved to    : {metrics_path}")
    print("=" * 80)
    print("Training pipeline successfully finished!")
    print("You can launch the web application by running: python app.py")
    print("=" * 80)


def main():
    base_dir, dataset_dir, model_dir, fake_csv, true_csv = get_paths()
    check_dataset_files(fake_csv, true_csv)
    df, is_sample = load_and_preprocess_data(fake_csv, true_csv)
    X_train, X_test, y_train, y_test, vectorizer = split_and_vectorize(df)
    best_name, best_model, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_artifacts(best_name, best_model, vectorizer, results, df, is_sample, model_dir, X_train.shape, X_test.shape)


if __name__ == "__main__":
    main()
