"""
Command-line Prediction Script for Sentiment Analysis
"""

import argparse
import joblib
from preprocessing import TextPreprocessor


def load_models():
    """Load trained models."""
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model = joblib.load('models/logistic_regression.pkl')
        print("✅ Models loaded successfully!")
        return vectorizer, model
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("Please train models first using 'python train.py'")
        return None, None


def predict(text, vectorizer, model):
    """
    Predict sentiment of a text.
    
    Args:
        text: Input text
        vectorizer: TF-IDF vectorizer
        model: Trained model
    
    Returns:
        Prediction result
    """
    # Preprocess
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess(text)
    
    # Vectorize
    text_vec = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    
    result = {
        "original_text": text,
        "processed_text": processed_text,
        "sentiment": sentiment,
        "prediction": int(prediction)
    }
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vec)[0]
        result["confidence"] = {
            "positive": round(proba[1] * 100, 2),
            "negative": round(proba[0] * 100, 2)
        }
        result["confidence_score"] = round(max(proba) * 100, 2)
    
    return result


def interactive_mode(vectorizer, model):
    """Run interactive prediction mode."""
    print("\n" + "=" * 50)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            text = input("📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                continue
            
            result = predict(text, vectorizer, model)
            
            print("\n" + "-" * 40)
            print(f"📊 Sentiment: {result['sentiment']}")
            
            if 'confidence' in result:
                print(f"   Confidence: {result['confidence_score']}%")
                print(f"   • Positive: {result['confidence']['positive']}%")
                print(f"   • Negative: {result['confidence']['negative']}%")
            
            print("-" * 40 + "\n")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Prediction')
    parser.add_argument('--text', '-t', type=str, help='Text to analyze')
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help='Run in interactive mode')
    parser.add_argument('--model', '-m', type=str, default='logistic_regression',
                        help='Model to use (logistic_regression, svm, naive_bayes)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS - PREDICTION")
    print("=" * 60)
    
    # Load models
    vectorizer, model = load_models()
    
    if vectorizer is None or model is None:
        exit(1)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(vectorizer, model)
    
    elif args.text:
        # Single prediction
        result = predict(args.text, vectorizer, model)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"\n📝 Original Text:\n   {result['original_text']}")
        print(f"\n✨ Processed Text:\n   {result['processed_text']}")
        print(f"\n🎯 Sentiment: {result['sentiment']}")
        
        if 'confidence' in result:
            print(f"\n📊 Confidence Scores:")
            print(f"   • Positive: {result['confidence']['positive']}%")
            print(f"   • Negative: {result['confidence']['negative']}%")
        
        print()
    
    else:
        # Demo predictions
        sample_texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible experience. Complete waste of time and money.",
            "It was okay, nothing special but not bad either.",
            "Best film I've seen this year! Highly recommend!",
            "I hate this. Never watching again.",
            "The product quality is amazing and delivery was super fast!",
            "Disappointed with the customer service response time."
        ]
        
        print("\n📊 Demo Predictions:")
        print("-" * 60)
        
        for text in sample_texts:
            result = predict(text, vectorizer, model)
            print(f"\nText: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
            print(f"Result: {result['sentiment']}")
        
        print("\n" + "-" * 60)
        print("\n💡 For interactive mode: python predict.py --interactive")
        print("💡 For single text: python predict.py --text 'Your text here'")


if __name__ == '__main__':
    main()
