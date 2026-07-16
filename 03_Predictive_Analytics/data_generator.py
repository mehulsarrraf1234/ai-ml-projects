"""
Data Generator for Predictive Analytics
Generates synthetic datasets for sales, churn, and demand prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_sales_data(n_samples=1000, random_state=42):
    """
    Generate synthetic sales prediction dataset.
    
    Features:
    - date: Date of sale
    - store_id: Store identifier
    - product_id: Product identifier
    - price: Product price
    - discount: Discount percentage
    - holiday: Whether it's a holiday
    - weekend: Whether it's a weekend
    - month: Month of year
    - day_of_week: Day of week
    - advertising_spend: Advertising budget
    - customer_count: Number of customers
    
    Target:
    - sales: Sales amount
    """
    np.random.seed(random_state)
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    data = {
        'date': dates,
        'store_id': np.random.randint(1, 11, n_samples),  # 10 stores
        'product_id': np.random.randint(1, 21, n_samples),  # 20 products
        'price': np.random.uniform(10, 500, n_samples),
        'discount': np.random.uniform(0, 30, n_samples),
        'holiday': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'weekend': [d.weekday() >= 5 for d in dates],
        'month': [d.month for d in dates],
        'day_of_week': [d.weekday() for d in dates],
        'advertising_spend': np.random.uniform(100, 1000, n_samples),
        'customer_count': np.random.randint(50, 500, n_samples),
    }
    
    # Generate target (sales) with some noise
    data['sales'] = (
        data['customer_count'] * 2 +
        data['advertising_spend'] * 0.5 +
        (100 - data['discount']) * 0.5 +
        data['price'] * 0.3 +
        np.where(data['holiday'], 100, 0) +
        np.where(data['weekend'], 50, 0) +
        np.random.normal(0, 50, n_samples)
    )
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    mask = np.random.random(n_samples) < 0.02
    df.loc[mask, 'discount'] = np.nan
    
    mask = np.random.random(n_samples) < 0.01
    df.loc[mask, 'advertising_spend'] = np.nan
    
    return df


def generate_churn_data(n_samples=1000, random_state=42):
    """
    Generate synthetic customer churn prediction dataset.
    
    Features:
    - customer_id: Customer identifier
    - age: Customer age
    - tenure: Months as customer
    - monthly_charges: Monthly bill amount
    - total_charges: Total amount paid
    - contract_type: Monthly/Yearly/Two-year
    - payment_method: Payment type
    - num_support_calls: Support calls made
    - num_complaints: Complaints filed
    - data_usage_gb: Data usage
    - calls_made: Calls made
    - subscription_type: Service type
    
    Target:
    - churned: Whether customer churned
    """
    np.random.seed(random_state)
    
    data = {
        'customer_id': [f'CUST{i:05d}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 70, n_samples),
        'tenure': np.random.randint(1, 73, n_samples),
        'monthly_charges': np.random.uniform(20, 200, n_samples),
        'total_charges': np.random.uniform(100, 10000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                          n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'],
                                           n_samples, p=[0.4, 0.35, 0.25]),
        'num_support_calls': np.random.poisson(2, n_samples),
        'num_complaints': np.random.poisson(0.5, n_samples),
        'data_usage_gb': np.random.uniform(0, 500, n_samples),
        'calls_made': np.random.poisson(50, n_samples),
        'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'],
                                               n_samples, p=[0.4, 0.4, 0.2]),
        'paperless_billing': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'multiple_lines': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'online_security': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'online_backup': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    # Generate churn with probability based on features
    churn_prob = (
        0.3 +  # Base churn rate
        np.where(data['contract_type'] == 'Month-to-month', 0.2, 0) +
        np.where(data['contract_type'] == 'One year', 0.05, 0) +
        np.where(data['contract_type'] == 'Two year', -0.1, 0) +
        np.where(data['num_support_calls'] > 5, 0.15, 0) +
        np.where(data['num_complaints'] > 2, 0.1, 0) +
        np.where(data['tenure'] < 12, 0.1, 0) +
        np.where(data['payment_method'] == 'Electronic check', 0.05, 0)
    )
    
    churn_prob = np.clip(churn_prob, 0, 1)
    data['churned'] = np.random.choice([0, 1], n_samples, p=[1 - churn_prob.mean(), churn_prob.mean()])
    
    # Recalculate churn based on actual probabilities
    data['churned'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return pd.DataFrame(data)


def generate_demand_data(n_samples=1000, random_state=42):
    """
    Generate synthetic demand forecasting dataset.
    
    Features:
    - date: Date
    - product_id: Product identifier
    - category: Product category
    - store_id: Store identifier
    - region: Sales region
    - price: Product price
    - competitor_price: Competitor price
    - temperature: Weather temperature
    - rainfall: Rainfall in mm
    - holiday: Holiday indicator
    - promotion: Promotion active
    
    Target:
    - demand: Units sold
    """
    np.random.seed(random_state)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'date': dates,
        'product_id': np.random.randint(1, 51, n_samples),
        'category': np.random.choice(categories, n_samples),
        'store_id': np.random.randint(1, 11, n_samples),
        'region': np.random.choice(regions, n_samples),
        'price': np.random.uniform(10, 1000, n_samples),
        'competitor_price': np.random.uniform(10, 1000, n_samples),
        'temperature': np.random.normal(25, 10, n_samples),
        'rainfall': np.random.exponential(5, n_samples),
        'holiday': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'promotion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }
    
    # Calculate demand
    price_effect = (data['competitor_price'] - data['price']) * 0.5
    weather_effect = np.where(data['temperature'] > 30, -20, np.where(data['temperature'] < 15, -10, 0))
    holiday_effect = np.where(data['holiday'], 50, 0)
    promotion_effect = np.where(data['promotion'], 30, 0)
    
    data['demand'] = (
        100 +
        price_effect +
        weather_effect +
        holiday_effect +
        promotion_effect +
        np.random.normal(0, 20, n_samples)
    )
    
    data['demand'] = np.maximum(data['demand'], 0)  # Demand can't be negative
    
    return pd.DataFrame(data)


def get_dataset(dataset_name='sales', n_samples=1000):
    """
    Get dataset by name.
    
    Args:
        dataset_name: 'sales', 'churn', or 'demand'
        n_samples: Number of samples
    
    Returns:
        pandas DataFrame
    """
    generators = {
        'sales': generate_sales_data,
        'churn': generate_churn_data,
        'demand': generate_demand_data
    }
    
    if dataset_name not in generators:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(generators.keys())}")
    
    return generators[dataset_name](n_samples)


if __name__ == '__main__':
    print("=" * 60)
    print("DATA GENERATOR - PREDICTIVE ANALYTICS")
    print("=" * 60)
    
    # Generate sample datasets
    print("\n📊 Generating Sales Dataset...")
    sales_df = generate_sales_data(100)
    print(f"   Shape: {sales_df.shape}")
    print(sales_df.head())
    
    print("\n📊 Generating Churn Dataset...")
    churn_df = generate_churn_data(100)
    print(f"   Shape: {churn_df.shape}")
    print(churn_df.head())
    
    print("\n📊 Generating Demand Dataset...")
    demand_df = generate_demand_data(100)
    print(f"   Shape: {demand_df.shape}")
    print(demand_df.head())
