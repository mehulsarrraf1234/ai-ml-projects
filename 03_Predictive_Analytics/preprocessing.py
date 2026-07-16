"""
Data Preprocessing Module for Predictive Analytics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Complete data preprocessing pipeline for ML models.
    """
    
    def __init__(self):
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.scaler = None
        self.label_encoders = {}
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []
        self.fitted = False
    
    def identify_features(self, df, target_col=None):
        """Identify numeric and categorical features."""
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_col in self.numeric_features:
            self.numeric_features.remove(target_col)
        if target_col in self.categorical_features:
            self.categorical_features.remove(target_col)
        
        return self.numeric_features, self.categorical_features
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', 'most_frequent', or 'drop'
        
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if strategy in ['mean', 'median']:
                imp = SimpleImputer(strategy=strategy)
                df[numeric_cols] = imp.fit_transform(df[numeric_cols])
            else:
                imp = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imp.fit_transform(df[categorical_cols])
        
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=3):
        """
        Remove outliers from specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame without outliers
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        return df
    
    def encode_categorical(self, df, categorical_cols=None, method='onehot'):
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: Columns to encode
            method: 'onehot' or 'label'
        
        Returns:
            Encoded DataFrame and feature names
        """
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        self.feature_names = df.columns.tolist()
        return df
    
    def scale_features(self, df, numeric_cols=None, method='standard'):
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            numeric_cols: Columns to scale
            method: 'standard' (z-score) or 'minmax'
        
        Returns:
            Scaled DataFrame
        """
        df = df.copy()
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def create_features(self, df):
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with new features
        """
        df = df.copy()
        
        # Date features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['quarter'] = df['date'].dt.quarter
            df['day_of_year'] = df['date'].dt.dayofyear
        
        # Interaction features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            if f'{col1}_x_{col2}' not in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return df
    
    def full_preprocess(self, df, target_col=None, remove_outliers_flag=False):
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            remove_outliers_flag: Whether to remove outliers
        
        Returns:
            Preprocessed DataFrame
        """
        print("🔧 Running preprocessing pipeline...")
        
        # Identify features
        self.identify_features(df, target_col)
        print(f"   Numeric features: {len(self.numeric_features)}")
        print(f"   Categorical features: {len(self.categorical_features)}")
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy='mean')
        print(f"   Handled missing values")
        
        # Remove outliers if requested
        if remove_outliers_flag:
            df = self.remove_outliers(df, method='iqr', threshold=1.5)
            print(f"   Removed outliers")
        
        # Create new features
        df = self.create_features(df)
        print(f"   Created new features")
        
        # Encode categorical variables
        if self.categorical_features:
            df = self.encode_categorical(df, method='onehot')
            print(f"   Encoded categorical variables")
        
        self.fitted = True
        print("✅ Preprocessing complete!")
        
        return df
    
    def transform(self, df):
        """Transform new data using fitted preprocessor."""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call full_preprocess first.")
        return df
    
    def get_feature_importance(self, df, target, n_features=10):
        """
        Get top features based on correlation with target.
        
        Args:
            df: Input DataFrame
            target: Target column name
            n_features: Number of top features
        
        Returns:
            DataFrame with feature correlations
        """
        correlations = df.select_dtypes(include=[np.number]).corr()[target].drop(target)
        top_features = correlations.abs().sort_values(ascending=False).head(n_features)
        
        return pd.DataFrame({
            'feature': top_features.index,
            'correlation': top_features.values
        })


def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Test set proportion
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    from data_generator import get_dataset
    
    print("=" * 60)
    print("DATA PREPROCESSING DEMO")
    print("=" * 60)
    
    # Load sample data
    df = get_dataset('sales', 500)
    print(f"\n📊 Original dataset shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Full preprocessing
    df_processed = preprocessor.full_preprocess(df, target_col='sales')
    
    print(f"\n📊 Processed dataset shape: {df_processed.shape}")
