"""
Streamlit Dashboard for Predictive Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import get_dataset
from preprocessing import DataPreprocessor, split_data
from models import RegressionModels, ClassificationModels
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(dataset_name, n_samples):
    """Load and cache dataset."""
    return get_dataset(dataset_name, n_samples)


def main():
    # Header
    st.markdown('<p class="main-header">📊 Predictive Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset selection
    dataset_options = {
        'sales': 'Sales Prediction',
        'churn': 'Customer Churn',
        'demand': 'Demand Forecasting'
    }
    
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x]
    )
    
    n_samples = st.sidebar.slider("Number of Samples", 500, 5000, 1000, step=100)
    
    # Load data
    df = load_data(selected_dataset, n_samples)
    
    # Display raw data
    st.subheader(f"📁 {dataset_options[selected_dataset]} Dataset")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔧 Preprocessing", "🤖 Models", "📈 Predictions"])
    
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Missing values
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.write(missing)
        else:
            st.success("No missing values found!")
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if selected_dataset == 'sales':
            target_col = 'sales'
        elif selected_dataset == 'churn':
            target_col = 'churned'
        else:
            target_col = 'demand'
        
        col1, col2 = st.columns(2)
        
        with col1:
            if target_col in df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(df[target_col], bins=30, color='steelblue', edgecolor='black')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{target_col.title()} Distribution')
                st.pyplot(fig)
        
        with col2:
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                          alpha=0.5, c='steelblue')
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                ax.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
                st.pyplot(fig)
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', ax=ax)
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Data Preprocessing")
        
        # Preprocessing options
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.checkbox("Handle Missing Values", value=True)
            remove_outliers = st.checkbox("Remove Outliers")
        
        with col2:
            scale_data = st.checkbox("Scale Features")
            create_features = st.checkbox("Create New Features")
        
        if st.button("Run Preprocessing", type="primary"):
            preprocessor = DataPreprocessor()
            
            if handle_missing:
                df_processed = preprocessor.handle_missing_values(df)
                st.success("✅ Missing values handled")
            else:
                df_processed = df.copy()
            
            if remove_outliers:
                df_processed = preprocessor.remove_outliers(df_processed)
                st.success("✅ Outliers removed")
            
            if create_features:
                df_processed = preprocessor.create_features(df_processed)
                st.success("✅ New features created")
            
            st.info(f"Processed data shape: {df_processed.shape}")
            st.dataframe(df_processed.head(), use_container_width=True)
    
    with tab3:
        st.subheader("Machine Learning Models")
        
        # Model training
        col1, col2 = st.columns(2)
        
        with col1:
            is_classification = selected_dataset == 'churn'
            if is_classification:
                target_col = 'churned'
                st.info("🔮 Classification Task: Predicting customer churn")
            else:
                target_col = 'sales' if selected_dataset == 'sales' else 'demand'
                st.info("📈 Regression Task: Predicting continuous values")
        
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Preprocess
                preprocessor = DataPreprocessor()
                df_processed = preprocessor.full_preprocess(df, target_col=target_col)
                
                # Split data
                X_train, X_test, y_train, y_test = split_data(
                    df_processed, target_col, test_size=test_size
                )
                
                # Train models
                if is_classification:
                    model_handler = ClassificationModels()
                    model_handler.train(X_train, y_train)
                    results = model_handler.evaluate(X_test, y_test)
                else:
                    model_handler = RegressionModels()
                    model_handler.train(X_train, y_train)
                    results = model_handler.evaluate(X_test, y_test)
                
                # Display results
                st.success("✅ Training complete!")
                
                # Results visualization
                st.subheader("Model Comparison")
                
                model_names = list(results.keys())
                if is_classification:
                    metric = 'accuracy'
                else:
                    metric = 'r2'
                
                scores = [results[m][metric] for m in model_names]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(model_names, scores, color='steelblue')
                ax.set_xlabel(metric.upper())
                ax.set_title('Model Performance Comparison')
                
                for bar, score in zip(bars, scores):
                    ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{score:.4f}', va='center')
                
                st.pyplot(fig)
                
                # Store in session state
                st.session_state['model_handler'] = model_handler
                st.session_state['is_classification'] = is_classification
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
    
    with tab4:
        st.subheader("Make Predictions")
        
        if 'model_handler' not in st.session_state:
            st.warning("⚠️ Please train models first in the 'Models' tab!")
        else:
            model_handler = st.session_state['model_handler']
            is_classification = st.session_state['is_classification']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            # Select model
            model_names = list(model_handler.trained_models.keys())
            selected_model = st.selectbox("Select Model", model_names)
            
            # Make predictions
            if st.button("Predict", type="primary"):
                predictions = model_handler.predict(X_test, selected_model)
                
                # Display predictions vs actual
                results_df = pd.DataFrame({
                    'Actual': y_test.values[:20],
                    'Predicted': predictions[:20]
                })
                
                if is_classification:
                    results_df['Actual'] = results_df['Actual'].map({0: 'No', 1: 'Yes'})
                    results_df['Predicted'] = results_df['Predicted'].map({0: 'No', 1: 'Yes'})
                
                st.subheader("Sample Predictions (first 20)")
                st.dataframe(results_df, use_container_width=True)
                
                # Plot predictions vs actual
                fig, ax = plt.subplots(figsize=(10, 5))
                x_range = range(len(predictions[:50]))
                ax.plot(x_range, y_test.values[:50], 'b-', label='Actual', alpha=0.7)
                ax.plot(x_range, predictions[:50], 'r--', label='Predicted', alpha=0.7)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Value')
                ax.set_title('Actual vs Predicted Values')
                ax.legend()
                st.pyplot(fig)
                
                # Metrics
                if is_classification:
                    from sklearn.metrics import accuracy_score, f1_score
                    acc = accuracy_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions, average='weighted')
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("F1 Score", f"{f1:.4f}")
                else:
                    from sklearn.metrics import r2_score, mean_absolute_error
                    r2 = r2_score(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("R² Score", f"{r2:.4f}")
                    col2.metric("MAE", f"{mae:.4f}")


if __name__ == '__main__':
    main()
