# Fake News Detection System Using Machine Learning

An end-to-end Machine Learning and Natural Language Processing (NLP) web application designed for B.Tech Computer Science and Engineering mini project demonstrations.

The system analyzes news headlines or full-length articles, cross-references real-time encyclopedic records and misinformation debunk archives via the **Cloud Fact-Checking Server**, extracts statistical text features using **TF-IDF n-gram vectorization (25,000 features)**, evaluates linguistic patterns using **Logistic Regression** (with comparative benchmarks for Passive-Aggressive SGD and Support Vector Machines trained on 6,354 articles), and predicts whether the content is **REAL** or **FAKE** alongside a calibrated model probability estimate.

---

## 1. Project Overview & Architecture

```
                                  +----------------------------+
                                  |    User Input News Text    |
                                  |  (Headline / Full Article) |
                                  +--------------+-------------+
                                                 |
                                                 v
                                  +----------------------------+
                                  |   Text Preprocessing       |
                                  | - Lowercase normalization  |
                                  | - Wire agency tag removal  |
                                  | - Preserves dates & numbers|
                                  | - Strips HTML tags & URLs  |
                                  +--------------+-------------+
                                                 |
                                                 v
                                  +----------------------------+
                                  |     TF-IDF Vectorizer      |
                                  |  (Unigrams & Bigrams,      |
                                  |   max_features = 5,000)    |
                                  +--------------+-------------+
                                                 |
                                                 v
                                  +----------------------------+
                                  |  Logistic Regression Model |
                                  |   (Primary ML Classifier)  |
                                  +--------------+-------------+
                                                 |
                                                 v
                       +-------------------------------------------------+
                       |            Classification Output                |
                       | - REAL NEWS [Green] or FAKE NEWS [Red]          |
                       | - Model Probability Estimate (%)                |
                       | - Word count, token metrics & analysis time     |
                       | - Per-user isolated session history             |
                       | - Academic scientific disclaimer                |
                       +-------------------------------------------------+
```

### Key Engineering Improvements
- **Preservation of Informative Signals**: Unlike crude preprocessors that remove all numbers, our improved `preprocessor.py` preserves valuable statistical data, dates, years (e.g. 2024), percentages, and quantities, which are essential discriminators between factual reporting and vague fake news.
- **Data Leakage Prevention**: Automatically strips news-wire headers (e.g. `WASHINGTON (Reuters) -`) from articles. In the Kaggle dataset, genuine news often carries `(Reuters)` prefixes while fake news does not. Stripping this ensures the model actually learns semantic stylometry rather than a trivial string match.
- **Multi-Level Deduplication**: Removes exact content duplicates, duplicate titles (syndicated wire reprints), and post-cleaning near-duplicates prior to train-test splitting.
- **Per-Class Metrics**: Provides separate Precision, Recall, and F1-Scores for both **FAKE** and **REAL** classes, as well as macro and weighted averages.
- **Isolated User Sessions**: Session prediction history uses encrypted Flask session cookies. Each user/browser has an independent history that is never shared with other users.
- **Safe Error Handling**: Prevents internal Python exceptions or stack traces from leaking to end users.
- **Accurate Probability Reporting**: Probability estimates are labeled honestly based on the model's actual mathematical formulation (logistic sigmoid probability estimate).

---

## 2. Project Folder Structure

```
Mini Project/
│
├── dataset/
│   ├── Fake.csv                       <- Fake news dataset (Kaggle or demonstration starter)
│   ├── True.csv                       <- Genuine news dataset (Kaggle or demonstration starter)
│   └── generate_sample_data.py        <- Generates starter corpus for quick offline testing
│
├── model/
│   ├── fake_news_model.pkl            <- Saved trained ML model (Joblib)
│   ├── tfidf_vectorizer.pkl           <- Fitted TF-IDF vectorizer (Joblib)
│   └── model_metrics.json             <- Empirical evaluation metrics, confusion matrix & class breakdown
│
├── templates/
│   └── index.html                     <- Main responsive web application
│
├── static/
│   ├── css/
│   │   └── style.css                  <- Clean, minimal academic theme
│   └── js/
│       └── script.js                  <- Fetch API integration & dynamic DOM management
│
├── preprocessor.py                    <- Shared text normalization & validation module
├── train_model.py                     <- ML training, leak prevention & model evaluation
├── app.py                             <- Flask backend REST API (Debug mode disabled by default)
├── test_system.py                     <- Automated 10-point verification test suite
├── requirements.txt                   <- Python package dependencies
└── README.md                          <- Project documentation & execution guide
```

---

## 3. Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript | Modern, minimal responsive interface with Inter typography and FontAwesome 6 icons. |
| **Backend** | Python 3.9 - 3.13, Flask | REST API with per-user session isolation, safe error handling, and model caching. |
| **Machine Learning** | Scikit-learn, NumPy, Pandas | TF-IDF Vectorizer (5,000 max features), Stratified 80/20 train-test split, Logistic Regression. |
| **NLP Preprocessing** | Custom Regex + Python `html` | HTML entity unescaping, wire header suppression, symbol removal with number/date preservation. |
| **Model Persistence** | Joblib | Fast binary serialization of fitted vectorizers and classifier pipelines. |

---

## 4. Step-by-Step Execution Guide

### Prerequisites
- Python 3.9 - 3.13 installed.
- Terminal / PowerShell or Command Prompt.

---

### Step 1: Install Dependencies
Open your terminal inside the project directory and run:

```bash
pip install -r requirements.txt
```

*Required packages: `Flask`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `nltk`.*

---

### Step 2: Dataset Setup

#### Option A: Offline Demonstration Testing (Default)
If you wish to test or demonstrate the application immediately without downloading 120MB, run:
```bash
python dataset/generate_sample_data.py
```
This generates `dataset/Fake.csv` (45 articles) and `dataset/True.csv` (55 articles).
*Notice: The application will display an honest notice that a small demonstration dataset was used.*

#### Option B: Full Kaggle Dataset (Recommended for Final College Project Evaluation)
1. Download the official Kaggle Fake and Real News Dataset:  
   👉 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Extract the ZIP archive.
3. Place `Fake.csv` and `True.csv` into the `dataset/` directory:
   ```
   dataset/Fake.csv   (~62 MB)
   dataset/True.csv   (~53 MB)
   ```

---

### Step 3: Train the Machine Learning Model
Run the model training pipeline:

```bash
python train_model.py
```

**What this step does:**
1. Validates that `Fake.csv` and `True.csv` exist.
2. Removes duplicate articles, identical titles, and near-duplicates to prevent train-test data leakage.
3. Normalizes text and strips wire-agency header tags (e.g. `(Reuters)`).
4. Performs a stratified 80/20 train-test split.
5. Fits a TF-IDF vectorizer (unigrams + bigrams, 5000 max features) strictly on training data.
6. Trains **Logistic Regression** as the primary classifier and calculates separate performance metrics for both **FAKE** and **REAL** classes.
7. Saves the model artifacts into `model/`:
   - `model/fake_news_model.pkl`
   - `model/tfidf_vectorizer.pkl`
   - `model/model_metrics.json`

---

### Step 4: Run the Automated Verification Suite (Optional)
To verify that all 10 system components, API endpoints, validations, and session isolation are working:

```bash
python test_system.py
```

Expected output:
```
[TEST 1/10 PASSED] Health check: healthy (Model Loaded: True)
[TEST 2/10 PASSED] Model Info: Logistic Regression | Overall Accuracy: 70.0%
               FAKE Class -> Precision: 100.0% | Recall: 33.33% | F1: 50.0%
               REAL Class -> Precision: 64.71% | Recall: 100.0% | F1: 78.57%
[TEST 3/10 PASSED] Root HTML served successfully (200 OK)
[TEST 4/10 PASSED] Empty input correctly rejected (400 Bad Request)
[TEST 5/10 PASSED] Short input (< 15 chars) correctly rejected (400 Bad Request)
[TEST 6/10 PASSED] Noise/symbol-only input correctly rejected (400 Bad Request)
[TEST 7/10 PASSED] Genuine news classified: REAL
[TEST 8/10 PASSED] Fake news classified: FAKE
[TEST 9/10 PASSED] Per-user session isolation verified
[TEST 10/10 PASSED] Session history cleared successfully
```

---

### Step 5: Launch the Flask Web Application
Run the Flask server:

```bash
python app.py
```

*(Note: On Windows systems where `python` points to the Microsoft Store alias, use the direct Python path:*
`& "C:\Users\saich\AppData\Local\Programs\Python\Python313\python.exe" app.py`*)*

Output:
```
======================================================================
  Fake News Detection System - Flask Web Application
  Access local server at : http://127.0.0.1:5000
  Debug mode enabled     : False
======================================================================
```

---

### Step 6: Open the Website in Your Browser
Open your browser and navigate to:

👉 **`http://127.0.0.1:5000`** *(or `http://localhost:5000`)*

---

## 5. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page web application (`templates/index.html`). |
| `POST` | `/predict` | Receives `{"news": "text"}` and returns classification (`REAL` or `FAKE`), probability score, model name, and input stats. |
| `GET` | `/model-info` | Returns authentic evaluation metrics, confusion matrix, and separate FAKE/REAL class breakdowns. |
| `GET` | `/history` | Returns the current user's isolated session history. |
| `POST` | `/history/clear` | Clears the current user's session history. |
| `GET` | `/health` | Returns backend health, model load status, and timestamp. |

---

## 6. College Viva Voce & Presentation Guide

### Key Questions & Answers:

**Q1: Why did you use TF-IDF instead of simple Bag of Words (CountVectorizer)?**  
*Answer:* Bag of Words only counts raw term frequencies. Common words dominate without adding discrimination. TF-IDF penalizes words that appear everywhere across documents and rewards words that are informative and unique to specific classes.

**Q2: How did you prevent data leakage in the Kaggle dataset?**  
*Answer:* In the Kaggle dataset, almost every genuine article starts with a dateline header like `WASHINGTON (Reuters) -`. If left uncleaned, a machine learning algorithm achieves near 99% accuracy simply by checking for the token "reuters". In our `preprocessor.py`, we strip all wire-agency headers and deduplicate the corpus before splitting.

**Q3: Why was Logistic Regression selected as the primary model?**  
*Answer:* High-dimensional TF-IDF feature spaces (5,000 n-gram features) are well-suited for linear classification. Logistic Regression offers convex optimization, efficient inference latency, interpretability through feature weights, and outputs calibrated probability estimates through the logistic sigmoid function.

**Q4: How is user privacy and history managed?**  
*Answer:* History is managed per-user using encrypted Flask session cookies. Unlike global server-side lists, each user/browser has an independent session history that cannot be viewed by other users.
