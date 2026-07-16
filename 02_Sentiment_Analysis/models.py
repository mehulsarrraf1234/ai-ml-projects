"""
Sentiment Analysis Models
Includes Traditional ML, Deep Learning, and Transformer-based models
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


class TraditionalMLModels:
    """
    Traditional Machine Learning models for sentiment analysis.
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize models with TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, C=1.0),
            'svm': LinearSVC(max_iter=1000, C=1.0),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        self.trained_models = {}
    
    def vectorize_text(self, texts):
        """Transform texts to TF-IDF features."""
        return self.vectorizer.fit_transform(texts)
    
    def train(self, X_train, y_train):
        """
        Train all traditional ML models.
        
        Args:
            X_train: Training texts
            y_train: Training labels
        
        Returns:
            Dictionary of trained models
        """
        X_train_vec = self.vectorize_text(X_train)
        
        print("\n📊 Training Traditional ML Models...")
        print("-" * 40)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_vec, y_train)
            self.trained_models[name] = model
        
        return self.trained_models
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test texts
            y_test: Test labels
        
        Returns:
            Dictionary of evaluation results
        """
        X_test_vec = self.vectorizer.transform(X_test)
        
        results = {}
        print("\n📈 Model Evaluation Results:")
        print("-" * 40)
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name:20} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return results
    
    def predict(self, texts, model_name='logistic_regression'):
        """
        Make predictions using specified model.
        
        Args:
            texts: List of text strings
            model_name: Name of the model to use
        
        Returns:
            Predictions and probabilities
        """
        X_vec = self.vectorizer.transform(texts)
        
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found. Train models first.")
        
        predictions = model.predict(X_vec)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_vec)
        else:
            probabilities = None
        
        return predictions, probabilities
    
    def save_models(self, path='models/'):
        """Save trained models to disk."""
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.vectorizer, f'{path}tfidf_vectorizer.pkl')
        
        for name, model in self.trained_models.items():
            joblib.dump(model, f'{path}{name}.pkl')
        
        print(f"✅ Models saved to '{path}'")
    
    def load_models(self, path='models/'):
        """Load trained models from disk."""
        self.vectorizer = joblib.load(f'{path}tfidf_vectorizer.pkl')
        
        for name in self.models.keys():
            try:
                self.trained_models[name] = joblib.load(f'{path}{name}.pkl')
            except FileNotFoundError:
                pass
        
        print(f"✅ Models loaded from '{path}'")


# Simple Neural Network Model
class SimpleSentimentNN:
    """
    Simple Neural Network for sentiment analysis using TF-IDF features.
    """
    
    def __init__(self, max_features=10000, max_length=200):
        """
        Initialize the neural network.
        
        Args:
            max_features: Maximum TF-IDF features
            max_length: Maximum sequence length for text
        """
        self.max_features = max_features
        self.max_length = max_length
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build the neural network architecture."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Input
        
        self.model = Sequential([
            Input(shape=(self.max_features,)),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks if validation_data else [],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        probabilities = self.model.predict(X)
        return (probabilities > 0.5).astype(int).flatten()
    
    def save(self, path='models/sentiment_nn.h5'):
        """Save the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"✅ Model saved to '{path}'")
    
    def load(self, path='models/sentiment_nn.h5'):
        """Load the model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        print(f"✅ Model loaded from '{path}'")


# LSTM Model for Sequences
class LSTMSentimentModel:
    """
    LSTM-based model for sequence-based sentiment analysis.
    """
    
    def __init__(self, max_words=10000, max_length=200, embedding_dim=128):
        """
        Initialize LSTM model.
        
        Args:
            max_words: Maximum vocabulary size
            max_length: Maximum sequence length
            embedding_dim: Embedding dimension
        """
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None
    
    def build_model(self):
        """Build LSTM architecture."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, LSTM, Dense, Dropout, 
            Bidirectional, GlobalMaxPooling1D
        )
        
        self.model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def prepare_data(self, texts):
        """
        Tokenize and pad sequences.
        
        Args:
            texts: List of text strings
        
        Returns:
            Padded sequences
        """
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return padded
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        """Train the LSTM model."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
            ModelCheckpoint('models/lstm_best.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks if validation_data else callbacks[:2],
            verbose=1
        )
        
        return self.history
    
    def predict(self, texts):
        """Make predictions."""
        X = self.prepare_data(texts)
        probabilities = self.model.predict(X)
        return (probabilities > 0.5).astype(int).flatten()


if __name__ == '__main__':
    print("=" * 60)
    print("SENTIMENT ANALYSIS MODELS")
    print("=" * 60)
    print("\nAvailable model classes:")
    print("1. TraditionalMLModels - TF-IDF + ML classifiers")
    print("2. SimpleSentimentNN - Neural Network with TF-IDF")
    print("3. LSTMSentimentModel - LSTM for sequences")
