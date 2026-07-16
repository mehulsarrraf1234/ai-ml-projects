"""
Training Script for Sentiment Analysis Models
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from preprocessing import TextPreprocessor, load_imdb_dataset
from models import TraditionalMLModels, SimpleSentimentNN


def plot_training_history(history, title='Training History'):
    """Plot training and validation metrics."""
    if history is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


def compare_models(results):
    """Compare model accuracies."""
    models = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(models, accuracies, color='steelblue')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{acc*100:.2f}%', va='center', fontsize=11)
    
    plt.xlabel('Accuracy')
    plt.title('Model Comparison - Sentiment Analysis', fontsize=14)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()


def train_traditional_models(X_train, y_train, X_test, y_test):
    """Train and evaluate traditional ML models."""
    print("\n" + "=" * 60)
    print("TRAINING TRADITIONAL ML MODELS")
    print("=" * 60)
    
    ml_models = TraditionalMLModels(max_features=15000, ngram_range=(1, 2))
    
    # Train
    ml_models.train(X_train, y_train)
    
    # Evaluate
    results = ml_models.evaluate(X_test, y_test)
    
    # Save models
    ml_models.save_models('models/')
    
    # Test predictions
    print("\n📝 Sample Predictions:")
    sample_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible experience. Complete waste of time and money.",
        "It was okay, nothing special but not bad either.",
        "Best film I've seen this year! Highly recommend!",
        "I hate this. Never watching again."
    ]
    
    for text in sample_texts:
        pred, prob = ml_models.predict([text])
        sentiment = "Positive" if pred[0] == 1 else "Negative"
        print(f"  '{text[:50]}...' -> {sentiment}")
    
    return ml_models, results


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train simple neural network."""
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK")
    print("=" * 60)
    
    # Initialize
    nn_model = SimpleSentimentNN(max_features=15000)
    
    # Build model
    nn_model.build_model()
    print("\nModel Architecture:")
    nn_model.model.summary()
    
    # Train
    history = nn_model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=10,
        batch_size=64
    )
    
    # Plot training
    plot_training_history(history, 'Neural Network Training')
    
    # Save model
    nn_model.save('models/sentiment_nn.h5')
    
    # Evaluate
    _, accuracy = nn_model.model.evaluate(X_test, y_test)
    print(f"\n📊 Neural Network Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return nn_model, accuracy


def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS - MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    print("\n📂 Loading IMDB dataset...")
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_dataset()
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Positive samples (train): {sum(train_labels)}")
    print(f"Negative samples (train): {len(train_labels) - sum(train_labels)}")
    
    # Preprocess
    print("\n🔧 Preprocessing texts...")
    preprocessor = TextPreprocessor()
    
    # Use subset for faster processing
    sample_size = 10000
    train_subset = train_texts[:sample_size]
    train_labels_subset = train_labels[:sample_size]
    test_subset = test_texts[:min(sample_size, len(test_texts))]
    test_labels_subset = test_labels[:min(sample_size, len(test_texts))]
    
    X_train_processed = preprocessor.preprocess_batch(train_subset)
    X_test_processed = preprocessor.preprocess_batch(test_subset)
    
    print(f"Preprocessed {len(X_train_processed)} training samples")
    print(f"Preprocessed {len(X_test_processed)} test samples")
    
    # Train Traditional ML Models
    ml_models, ml_results = train_traditional_models(
        X_train_processed, train_labels_subset,
        X_test_processed, test_labels_subset
    )
    
    # Train Neural Network (using TF-IDF features)
    # For simplicity, we'll skip NN in main and focus on ML models
    # nn_model, nn_accuracy = train_neural_network(...)
    
    # Compare models
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    compare_models(ml_results)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModels saved in 'models/' directory:")
    print("  - tfidf_vectorizer.pkl")
    print("  - logistic_regression.pkl")
    print("  - svm.pkl")
    print("  - naive_bayes.pkl")
    print("  - random_forest.pkl")


if __name__ == '__main__':
    main()
