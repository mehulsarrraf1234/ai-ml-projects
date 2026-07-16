"""
E-commerce Data Generator
Generates synthetic data for recommendation system and analytics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_products(n_products=100, random_state=42):
    """
    Generate product catalog.
    
    Args:
        n_products: Number of products
        random_state: Random seed
    
    Returns:
        DataFrame with product information
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 
                  'Toys', 'Beauty', 'Food', 'Automotive', 'Health']
    
    products = []
    for i in range(1, n_products + 1):
        category = random.choice(categories)
        
        # Price ranges by category
        if category == 'Electronics':
            price = np.random.uniform(50, 2000)
        elif category == 'Clothing':
            price = np.random.uniform(20, 200)
        elif category == 'Books':
            price = np.random.uniform(10, 50)
        else:
            price = np.random.uniform(10, 300)
        
        products.append({
            'product_id': f'PROD{i:05d}',
            'product_name': f'{category} Item {i}',
            'category': category,
            'price': round(price, 2),
            'stock': np.random.randint(0, 500),
            'rating': round(np.random.uniform(3.0, 5.0), 1),
            'num_reviews': np.random.randint(5, 500)
        })
    
    return pd.DataFrame(products)


def generate_customers(n_customers=500, random_state=42):
    """
    Generate customer data.
    
    Args:
        n_customers: Number of customers
        random_state: Random seed
    
    Returns:
        DataFrame with customer information
    """
    np.random.seed(random_state)
    
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    
    segments = ['Regular', 'Premium', 'VIP']
    segment_probs = [0.6, 0.3, 0.1]
    
    customers = []
    for i in range(1, n_customers + 1):
        segment = np.random.choice(segments, p=segment_probs)
        
        # Premium customers have higher lifetime value
        if segment == 'VIP':
            lifetime_value = np.random.uniform(500, 5000)
        elif segment == 'Premium':
            lifetime_value = np.random.uniform(200, 1000)
        else:
            lifetime_value = np.random.uniform(50, 300)
        
        customers.append({
            'customer_id': f'CUST{i:05d}',
            'name': f'Customer {i}',
            'age': np.random.randint(18, 70),
            'city': random.choice(cities),
            'segment': segment,
            'lifetime_value': round(lifetime_value, 2),
            'total_orders': np.random.randint(1, 50),
            'avg_order_value': round(lifetime_value / max(1, np.random.randint(1, 50)), 2)
        })
    
    return pd.DataFrame(customers)


def generate_transactions(n_transactions=5000, random_state=42):
    """
    Generate transaction data with realistic patterns.
    
    Args:
        n_transactions: Number of transactions
        random_state: Random seed
    
    Returns:
        DataFrame with transactions
    """
    np.random.seed(random_state)
    
    # Generate products and customers
    n_products = 100
    n_customers = 500
    
    products = generate_products(n_products, random_state)
    customers = generate_customers(n_customers, random_state)
    
    # Generate transactions
    transactions = []
    transaction_id = 1
    
    start_date = datetime(2023, 1, 1)
    
    for _ in range(n_transactions):
        # Random date within the year
        days_offset = np.random.randint(0, 365)
        transaction_date = start_date + timedelta(days=days_offset)
        
        customer_id = f'CUST{np.random.randint(1, n_customers + 1):05d}'
        
        # Number of items per transaction (market basket)
        n_items = np.random.poisson(3) + 1
        n_items = min(n_items, 10)  # Cap at 10 items
        
        # Select products (with some correlation - customers tend to buy from same category)
        selected_products = np.random.choice(
            products['product_id'].values, 
            size=n_items, 
            replace=False
        )
        
        for product_id in selected_products:
            price = products[products['product_id'] == product_id]['price'].values[0]
            quantity = np.random.randint(1, 4)
            
            transactions.append({
                'transaction_id': f'TXN{transaction_id:06d}',
                'date': transaction_date,
                'customer_id': customer_id,
                'product_id': product_id,
                'quantity': quantity,
                'price': price,
                'total': round(price * quantity, 2)
            })
        
        transaction_id += 1
    
    transactions_df = pd.DataFrame(transactions)
    
    # Add product category
    transactions_df = transactions_df.merge(
        products[['product_id', 'category']], 
        on='product_id', 
        how='left'
    )
    
    return transactions_df, products, customers


def generate_basket_data(transactions_df, random_state=42):
    """
    Generate one-hot encoded basket data for market basket analysis.
    
    Args:
        transactions_df: Transaction data
        random_state: Random seed
    
    Returns:
        DataFrame with one-hot encoded products
    """
    # Group by transaction and create basket
    basket = transactions_df.groupby(['transaction_id', 'product_id'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('transaction_id')
    
    # Convert to binary (purchased or not)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    return basket


def get_all_data():
    """
    Get all generated data.
    
    Returns:
        Tuple of (transactions, products, customers)
    """
    return generate_transactions()


if __name__ == '__main__':
    print("=" * 60)
    print("E-COMMERCE DATA GENERATOR")
    print("=" * 60)
    
    # Generate data
    transactions, products, customers = get_all_data()
    
    print(f"\n📊 Generated Data:")
    print(f"   Transactions: {transactions.shape[0]} rows")
    print(f"   Products: {products.shape[0]} products")
    print(f"   Customers: {customers.shape[0]} customers")
    
    print("\n📦 Sample Products:")
    print(products.head())
    
    print("\n👥 Sample Customers:")
    print(customers.head())
    
    print("\n🛒 Sample Transactions:")
    print(transactions.head(10))
