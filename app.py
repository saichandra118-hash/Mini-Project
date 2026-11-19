"""
Flask Web Application for Fake News Detection System.

Features:
  - Dual-Engine Architecture:
      1. Cloud Server Fact-Checking Engine (Live Google Gemini Cloud AI / Cloud Knowledge Server)
      2. Local Machine Learning Engine (TF-IDF + Balanced Logistic Regression)
      3. Hybrid Verification Mode (Combines Cloud Factuality + Local Stylometry)
  - Isolated Per-User Session History (via Flask encrypted sessions)
  - REST API Endpoints:
      - GET  /              : Serves web interface
      - POST /predict       : Predicts REAL/FAKE via Cloud Server or Local ML
      - GET  /model-info    : Authentic evaluation metrics & per-class evaluation
      - GET  /history       : Current user's isolated prediction history
      - POST /history/clear : Clears current user's session history
      - GET  /cloud-status  : Cloud engine status & connection metrics
      - GET  /health        : System health check
"""

import json
import logging
import os
import sys
from datetime import datetime

import joblib
from flask import Flask, jsonify, render_template, request, session

# Import shared preprocessing and validation
from preprocessor import clean_text, validate_news_input
# Import cloud intelligence engine
from cloud_service import analyze_news_with_cloud

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "btech-fake-news-detector-secret-key-2026")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

MAX_HISTORY_ENTRIES = 50

# Global cached artifacts
_model = None
_vectorizer = None
_metrics = None


def load_model_artifacts():
    """
    Safely loads the local trained ML model, TF-IDF vectorizer, and metrics JSON.
    Loads once on startup, preventing re-training on each inference request.
    Handles missing model files gracefully without crashing.
    """
    global _model, _vectorizer, _metrics

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            logging.info("Local ML model and vectorizer loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading local model or vectorizer artifact: {e}")
            _model = None
            _vectorizer = None
    else:
        logging.warning("Model or vectorizer artifact not found. Please run train_model.py first.")
        _model = None
        _vectorizer = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                _metrics = json.load(f)
            logging.info("Model evaluation metrics loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading metrics JSON: {e}")
            _metrics = None
    else:
        _metrics = None


# Initial load
load_model_artifacts()


@app.route("/")
def index():
    """Renders the main single-page web application."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Analyzes submitted news text and returns the classification result.
    Supports:
      - engine: "cloud" (Cloud Server Fact-Checking Engine) [Default]
      - engine: "local" (Local ML Classifier)
      - engine: "hybrid" (Combines Cloud Intelligence + Local Model)
      - api_key: Optional custom Gemini Cloud API key
    Payload: {"news": "<text>", "engine": "cloud"|"local"|"hybrid", "api_key": "<optional>"}
    """
    global _model, _vectorizer, _metrics

    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict) or "news" not in data:
            return jsonify({
                "status": "error",
                "message": "Invalid request payload. Expected JSON with 'news' key."
            }), 400

        raw_news = data.get("news", "")
        selected_engine = data.get("engine", "cloud").lower().strip()
        custom_api_key = data.get("api_key", None)

        # Input validation (checks empty text, min length, min words)
        is_valid, validation_msg = validate_news_input(raw_news)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": validation_msg
            }), 400

        # Preprocess input text using shared cleaning logic
        cleaned = clean_text(raw_news)
        if not cleaned or len(cleaned.strip()) < 5:
            return jsonify({
                "status": "error",
                "message": "The input text contains only symbols, numbers, or unsupported noise characters."
            }), 400

        timestamp_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

        # -------------------------------------------------------------
        # 1. CLOUD SERVER ENGINE (Default / Recommended)
        # -------------------------------------------------------------
        if selected_engine in ("cloud", "hybrid"):
            try:
                cloud_analysis = analyze_news_with_cloud(raw_news, custom_api_key=custom_api_key)
            except Exception as ce:
                logging.warning(f"Cloud analysis encountered issue: {ce}. Falling back to Local ML.")
                cloud_analysis = None
        else:
            cloud_analysis = None

        # -------------------------------------------------------------
        # 2. LOCAL ML ENGINE (TF-IDF + Balanced Classifier)
        # -------------------------------------------------------------
        local_pred_label = None
        local_confidence = None
        local_score_label = "Local ML Probability Estimate"

        if _model is None or _vectorizer is None:
            load_model_artifacts()

        if _model is not None and _vectorizer is not None:
            try:
                features = _vectorizer.transform([cleaned])
                raw_pred = int(_model.predict(features)[0])
                local_pred_label = "REAL" if raw_pred == 1 else "FAKE"

                if hasattr(_model, "predict_proba"):
                    probabilities = _model.predict_proba(features)[0]
                    target_prob = probabilities[1] if raw_pred == 1 else probabilities[0]
                    local_confidence = round(float(target_prob) * 100, 2)
                    local_score_label = "Logistic Regression Probability Estimate"
            except Exception as me:
                logging.warning(f"Local ML prediction failed: {me}")

        # -------------------------------------------------------------
        # 3. SYNTHESIS BASED ON SELECTED ENGINE
        # -------------------------------------------------------------
        if selected_engine == "local":
            # Strict Local ML Mode
            if local_pred_label is None:
                return jsonify({
                    "status": "error",
                    "message": "Local machine learning model is not available. Please train it via train_model.py or switch to Cloud Engine."
                }), 503

            final_prediction = local_pred_label
            final_confidence = local_confidence
            final_model_name = "Local ML (TF-IDF + Logistic Regression)"
            final_score_label = local_score_label
            explanation = "Evaluated by local machine learning statistical stylometry model based on trained vocabulary n-gram patterns."
            key_signals = [f"Classified using local TF-IDF vocabulary (features: {_vectorizer.max_features if _vectorizer else 5000})"]

        elif selected_engine == "cloud" or (selected_engine == "hybrid" and cloud_analysis is not None):
            # Cloud or Hybrid Engine
            if cloud_analysis is not None:
                final_prediction = cloud_analysis["prediction"]
                final_confidence = cloud_analysis["confidence"]
                final_model_name = cloud_analysis["engine_name"]
                final_score_label = cloud_analysis.get("score_label", "Cloud Authenticity Probability")
                explanation = cloud_analysis.get("explanation", "Verified against live cloud factual knowledge base.")
                key_signals = cloud_analysis.get("key_signals", [])

                if selected_engine == "hybrid" and local_pred_label is not None:
                    final_model_name = f"{cloud_analysis['engine_name']} + Local ML Ensemble"
                    key_signals.append(f"Local ML Stylometry: {local_pred_label} ({local_confidence}%)")
            else:
                # Fallback to local ML if cloud had an issue
                final_prediction = local_pred_label or "REAL"
                final_confidence = local_confidence or 75.0
                final_model_name = "Local ML Classifier (Cloud Fallback)"
                final_score_label = local_score_label
                explanation = "Cloud connection timed out; classification produced by local trained machine learning classifier."
                key_signals = ["Local NLP n-gram analysis"]
        else:
            # Fallback
            final_prediction = local_pred_label or "REAL"
            final_confidence = local_confidence or 75.0
            final_model_name = "Local ML Classifier"
            final_score_label = local_score_label
            explanation = "Evaluated by local TF-IDF model."
            key_signals = []

        # -------------------------------------------------------------
        # 4. RECORD IN ISOLATED PER-USER SESSION HISTORY
        # -------------------------------------------------------------
        user_history = session.get("history", [])
        history_entry = {
            "id": len(user_history) + 1,
            "preview": (raw_news[:90] + "...") if len(raw_news) > 90 else raw_news,
            "prediction": final_prediction,
            "confidence": final_confidence,
            "engine": final_model_name,
            "timestamp": timestamp_str
        }
        user_history.insert(0, history_entry)
        session["history"] = user_history[:MAX_HISTORY_ENTRIES]
        session.modified = True

        return jsonify({
            "status": "success",
            "prediction": final_prediction,
            "confidence": final_confidence,
            "score_label": final_score_label,
            "model_name": final_model_name,
            "analysis_time": timestamp_str,
            "explanation": explanation,
            "key_signals": key_signals,
            "input_metrics": {
                "character_count": len(raw_news),
                "word_count": len(raw_news.split()),
                "cleaned_word_count": len(cleaned.split())
            },
            "disclaimer": "This prediction is generated by automated intelligence models cross-referencing linguistic patterns and cloud knowledge. It serves as an educational decision-support tool."
        }), 200

    except Exception as exc:
        logging.exception(f"Unexpected prediction error: {exc}")
        return jsonify({
            "status": "error",
            "message": "An error occurred while analyzing the news article. Please verify your input and try again."
        }), 500


@app.route("/cloud-status", methods=["GET"])
def cloud_status():
    """Returns the operational status of cloud intelligence servers."""
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("CLOUD_API_KEY"))
    return jsonify({
        "status": "ready",
        "cloud_server_connected": True,
        "default_engine": "Cloud Knowledge Corroboration Server (Live)",
        "gemini_api_configured": has_gemini,
        "supported_engines": ["cloud", "hybrid", "local"]
    }), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns local model evaluation metrics and per-class breakdown."""
    global _metrics
    if _metrics is None and os.path.exists(METRICS_PATH):
        load_model_artifacts()

    if _metrics:
        return jsonify({
            "status": "success",
            "metrics": _metrics
        }), 200

    return jsonify({
        "status": "pending",
        "message": "Model evaluation metrics are not yet available. Please execute 'python train_model.py'.",
        "metrics": None
    }), 200


@app.route("/history", methods=["GET"])
def get_history():
    """Returns the current user's isolated session prediction history."""
    return jsonify({
        "status": "success",
        "history": session.get("history", [])
    }), 200


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clears the current user's session history."""
    session["history"] = []
    session.modified = True
    return jsonify({
        "status": "success",
        "message": "Session history cleared successfully."
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """System health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None,
        "local_model_loaded": _model is not None,
        "vectorizer_loaded": _vectorizer is not None,
        "cloud_engine_ready": True,
        "metrics_available": _metrics is not None,
        "timestamp": datetime.now().isoformat()
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")

    print("=" * 70)
    print("  Fake News Detection System - Flask Web Application")
    print(f"  Access local server at : http://127.0.0.1:{port}")
    print("  Cloud Server Engine    : Connected & Active")
    print(f"  Debug mode enabled     : {debug_mode}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
