"""
Flask API for Sentiment Analysis Predictions
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Global model and vectorizer
vectorizer = None
model = None
model_name = None


def load_models():
    """Load trained models."""
    global vectorizer, model, model_name
    
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model = joblib.load('models/logistic_regression.pkl')
        model_name = 'Logistic Regression'
        print("✅ Models loaded successfully!")
        return True
    except FileNotFoundError:
        print("⚠️ Models not found. Please train models first using 'python train.py'")
        return False


def predict_sentiment(text, return_probability=True):
    """
    Predict sentiment of a single text.
    
    Args:
        text: Input text string
        return_probability: Whether to return probability scores
    
    Returns:
        Prediction and confidence
    """
    global vectorizer, model
    
    # Vectorize
    text_vec = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    result = {
        "text": text,
        "sentiment": sentiment,
        "prediction": int(prediction)
    }
    
    if return_probability and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vec)[0]
        result["confidence"] = {
            "positive": float(proba[1]),
            "negative": float(proba[0])
        }
        result["confidence_score"] = float(max(proba))
    
    return result


@app.route('/')
def home():
    """API home page."""
    return jsonify({
        "message": "Sentiment Analysis API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict sentiment",
            "/health": "GET - Check API health",
            "/batch": "POST - Batch prediction"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment of input text.
    
    Request body:
    {
        "text": "Your text here"
    }
    
    OR
    
    {
        "texts": ["Text 1", "Text 2", ...]
    }
    """
    global vectorizer, model
    
    if model is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Single text prediction
    if 'text' in data:
        result = predict_sentiment(data['text'])
        return jsonify(result)
    
    # Batch prediction
    elif 'texts' in data:
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list"}), 400
        
        results = []
        for text in texts:
            result = predict_sentiment(text, return_probability=False)
            results.append(result)
        
        return jsonify({
            "predictions": results,
            "count": len(results)
        })
    
    else:
        return jsonify({"error": "Provide 'text' or 'texts' in request body"}), 400


@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint.
    
    Request body:
    {
        "texts": ["Text 1", "Text 2", ...]
    }
    """
    return predict()


def main():
    """Run the Flask API."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS API")
    print("=" * 60)
    
    # Load models
    if not load_models():
        print("⚠️ Starting API without models. Predictions will fail.")
    
    print("\n📡 Starting Flask API...")
    print("Endpoints:")
    print("  - POST /predict  - Predict sentiment")
    print("  - GET  /health   - Health check")
    print("  - POST /batch    - Batch prediction")
    print("\nAPI running at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
