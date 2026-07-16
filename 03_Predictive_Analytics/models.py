"""
Machine Learning Models for Predictive Analytics
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os


class RegressionModels:
    """
    Regression models for predictive analytics.
    """
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        self.trained_models = {}
        self.best_model = None
    
    def train(self, X_train, y_train):
        """Train all regression models."""
        print("\n📊 Training Regression Models...")
        print("-" * 50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        
        return self.trained_models
    
    def evaluate(self, X_test, y_test):
        """Evaluate all models."""
        results = {}
        
        print("\n📈 Model Evaluation Results:")
        print("-" * 50)
        print(f"{'Model':<25} {'R² Score':<12} {'MAE':<12} {'RMSE':<12}")
        print("-" * 50)
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'predictions': y_pred
            }
            
            print(f"{name:<25} {r2:.4f}       {mae:.4f}      {rmse:.4f}")
        
        # Find best model
        best_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = self.trained_models[best_name]
        print(f"\n🏆 Best Model: {best_name} (R² = {results[best_name]['r2']:.4f})")
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation on all models."""
        cv_results = {}
        
        print(f"\n🔄 Cross-Validation (k={cv}):")
        print("-" * 50)
        
        for name, model in self.trained_models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"{name:<25} R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None):
        """Tune hyperparameters for Random Forest."""
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        
        print("\n⚙️ Hyperparameter Tuning (Random Forest)...")
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best R² score: {grid_search.best_score_:.4f}")
        
        self.trained_models['Random Forest (Tuned)'] = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def predict(self, X, model_name='Random Forest'):
        """Make predictions using specified model."""
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        return model.predict(X)
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance from tree-based models."""
        model = self.trained_models.get(model_name)
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    
    def save(self, path='models/'):
        """Save trained models."""
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = name.lower().replace(' ', '_')
            joblib.dump(model, f'{path}regression_{filename}.pkl')
        
        print(f"✅ Models saved to '{path}'")
    
    def load(self, path='models/'):
        """Load trained models."""
        for filename in os.listdir(path):
            if filename.startswith('regression_'):
                name = filename.replace('regression_', '').replace('.pkl', '').replace('_', ' ')
                self.trained_models[name] = joblib.load(f'{path}{filename}')
        
        print(f"✅ Models loaded from '{path}'")


class ClassificationModels:
    """
    Classification models for predictive analytics.
    """
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        self.trained_models = {}
        self.best_model = None
    
    def train(self, X_train, y_train):
        """Train all classification models."""
        print("\n📊 Training Classification Models...")
        print("-" * 50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        
        return self.trained_models
    
    def evaluate(self, X_test, y_test):
        """Evaluate all models."""
        results = {}
        
        print("\n📈 Model Evaluation Results:")
        print("-" * 60)
        print(f"{'Model':<25} {'Accuracy':<12} {'F1':<12} {'AUC-ROC':<12}")
        print("-" * 60)
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # AUC-ROC if probabilities available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                auc_roc = roc_auc_score(y_test, y_proba)
            else:
                auc_roc = 0.0
            
            results[name] = {
                'accuracy': accuracy,
                'f1': f1,
                'auc_roc': auc_roc,
                'predictions': y_pred
            }
            
            print(f"{name:<25} {accuracy:.4f}       {f1:.4f}      {auc_roc:.4f}")
        
        # Find best model
        best_name = max(results, key=lambda x: results[x]['f1'])
        self.best_model = self.trained_models[best_name]
        print(f"\n🏆 Best Model: {best_name} (F1 = {results[best_name]['f1']:.4f})")
        
        return results
    
    def get_classification_report(self, X_test, y_test, model_name='Random Forest'):
        """Get detailed classification report."""
        model = self.trained_models.get(model_name)
        y_pred = model.predict(X_test)
        
        print(f"\n📋 Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))
        
        return classification_report(y_test, y_pred, output_dict=True)
    
    def predict(self, X, model_name='Random Forest'):
        """Make predictions."""
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        return model.predict(X)
    
    def predict_proba(self, X, model_name='Random Forest'):
        """Get prediction probabilities."""
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None


if __name__ == '__main__':
    print("=" * 60)
    print("ML MODELS - PREDICTIVE ANALYTICS")
    print("=" * 60)
    print("\nAvailable model classes:")
    print("1. RegressionModels - For continuous target prediction")
    print("2. ClassificationModels - For categorical target prediction")
