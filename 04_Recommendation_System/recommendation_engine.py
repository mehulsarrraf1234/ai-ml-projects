"""
Recommendation Engine
Collaborative Filtering for E-commerce Product Recommendations
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering Recommendation System.
    Supports both user-based and item-based approaches.
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.products = None
        self.customers = None
    
    def create_user_item_matrix(self, transactions, products):
        """
        Create user-item interaction matrix.
        
        Args:
            transactions: Transaction data
            products: Product catalog
        
        Returns:
            User-item matrix
        """
        print("🔄 Creating user-item matrix...")
        
        # Aggregate by customer and product
        interaction = transactions.groupby(['customer_id', 'product_id'])['quantity'].sum().reset_index()
        
        # Create pivot table
        self.user_item_matrix = interaction.pivot(
            index='customer_id',
            columns='product_id',
            values='quantity'
        ).fillna(0)
        
        # Scale values
        scaler = MinMaxScaler()
        self.user_item_matrix = pd.DataFrame(
            scaler.fit_transform(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )
        
        # Create item-user matrix (transpose)
        self.item_user_matrix = self.user_item_matrix.T
        
        print(f"✅ User-item matrix created: {self.user_item_matrix.shape}")
        
        return self.user_item_matrix
    
    def compute_user_similarity(self, method='cosine'):
        """
        Compute user-user similarity.
        
        Args:
            method: Similarity metric ('cosine', 'correlation')
        
        Returns:
            User similarity matrix
        """
        print(f"🔄 Computing user similarity ({method})...")
        
        if method == 'cosine':
            self.user_similarity = cosine_similarity(self.user_item_matrix)
        elif method == 'correlation':
            self.user_similarity = self.user_item_matrix.T.corr().values
        
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("✅ User similarity computed")
        return self.user_similarity
    
    def compute_item_similarity(self, method='cosine'):
        """
        Compute item-item similarity.
        
        Args:
            method: Similarity metric
        
        Returns:
            Item similarity matrix
        """
        print(f"🔄 Computing item similarity ({method})...")
        
        if method == 'cosine':
            self.item_similarity = cosine_similarity(self.item_user_matrix)
        elif method == 'correlation':
            self.item_similarity = self.item_user_matrix.corr().values
        
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.item_user_matrix.columns,
            columns=self.item_user_matrix.columns
        )
        
        print("✅ Item similarity computed")
        return self.item_similarity
    
    def recommend_user_based(self, user_id, n_recommendations=5, exclude_purchased=True):
        """
        User-based collaborative filtering recommendations.
        
        Args:
            user_id: Customer ID
            n_recommendations: Number of recommendations
            exclude_purchased: Whether to exclude already purchased items
        
        Returns:
            List of recommended product IDs with scores
        """
        if user_id not in self.user_similarity.index:
            print(f"⚠️ User {user_id} not found in the system")
            return []
        
        # Get similar users
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:11]
        
        # Get items the target user hasn't purchased
        user_purchases = self.user_item_matrix.loc[user_id]
        unpurchased_items = user_purchases[user_purchases == 0].index
        
        # Calculate scores for unpurchased items
        item_scores = {}
        
        for item in unpurchased_items:
            score = 0
            sim_sum = 0
            
            for similar_user, similarity in similar_users.items():
                similar_user_rating = self.user_item_matrix.loc[similar_user, item]
                if similar_user_rating > 0:
                    score += similarity * similar_user_rating
                    sim_sum += abs(similarity)
            
            if sim_sum > 0:
                item_scores[item] = score / sim_sum
            else:
                item_scores[item] = 0
        
        # Sort and return top recommendations
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def recommend_item_based(self, user_id, n_recommendations=5):
        """
        Item-based collaborative filtering recommendations.
        
        Args:
            user_id: Customer ID
            n_recommendations: Number of recommendations
        
        Returns:
            List of recommended product IDs with scores
        """
        if user_id not in self.user_item_matrix.index:
            print(f"⚠️ User {user_id} not found")
            return []
        
        # Get user's purchased items
        user_purchases = self.user_item_matrix.loc[user_id]
        purchased_items = user_purchases[user_purchases > 0].index
        
        # Calculate scores for all items based on similarity to purchased items
        item_scores = {}
        
        for item in self.item_similarity.columns:
            if item not in purchased_items:  # Don't recommend already purchased
                score = 0
                
                for purchased_item in purchased_items:
                    if purchased_item in self.item_similarity.index:
                        score += self.item_similarity.loc[item, purchased_item]
                
                item_scores[item] = score
        
        # Sort and return top recommendations
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_popular_items(self, n_items=10):
        """
        Get most popular items by purchase frequency.
        
        Args:
            n_items: Number of items to return
        
        Returns:
            DataFrame with popular items
        """
        purchase_counts = self.user_item_matrix.sum().sort_values(ascending=False)
        
        return purchase_counts.head(n_items)
    
    def get_similar_items(self, product_id, n_similar=5):
        """
        Get items similar to a given product.
        
        Args:
            product_id: Product ID
            n_similar: Number of similar items
        
        Returns:
            List of similar product IDs with similarity scores
        """
        if product_id not in self.item_similarity.index:
            print(f"⚠️ Product {product_id} not found")
            return []
        
        similarities = self.item_similarity[product_id].sort_values(ascending=False)
        
        # Exclude the product itself
        similarities = similarities[similarities.index != product_id]
        
        return similarities.head(n_similar).to_dict()


class ContentBasedRecommender:
    """
    Content-based recommendation using product features.
    """
    
    def __init__(self):
        self.products = None
        self.product_features = None
        self.feature_similarity = None
    
    def fit(self, products):
        """
        Fit the recommender with product data.
        
        Args:
            products: Product catalog DataFrame
        """
        self.products = products
        
        # Create feature matrix (price, rating, category encoding)
        feature_cols = ['price', 'rating']
        
        # One-hot encode categories
        category_dummies = pd.get_dummies(products['category'], prefix='cat')
        
        # Combine features
        self.product_features = pd.concat([
            products[feature_cols].reset_index(drop=True),
            category_dummies.reset_index(drop=True)
        ], axis=1)
        
        self.product_features.index = products['product_id']
        
        # Scale features
        scaler = MinMaxScaler()
        self.product_features = pd.DataFrame(
            scaler.fit_transform(self.product_features),
            index=self.product_features.index,
            columns=self.product_features.columns
        )
        
        # Compute similarity
        self.feature_similarity = cosine_similarity(self.product_features)
        self.feature_similarity = pd.DataFrame(
            self.feature_similarity,
            index=self.product_features.index,
            columns=self.product_features.index
        )
        
        print("✅ Content-based recommender fitted")
    
    def recommend_similar(self, product_id, n_recommendations=5):
        """
        Recommend products similar to given product.
        
        Args:
            product_id: Product ID
            n_recommendations: Number of recommendations
        
        Returns:
            List of similar product IDs with scores
        """
        if product_id not in self.feature_similarity.index:
            return []
        
        similarities = self.feature_similarity[product_id].sort_values(ascending=False)
        similarities = similarities[similarities.index != product_id]
        
        return list(zip(similarities.head(n_recommendations).index, 
                       similarities.head(n_recommendations).values))


def demo():
    """Demonstrate the recommendation system."""
    from data_generator import get_all_data, generate_products
    
    print("=" * 60)
    print("RECOMMENDATION ENGINE DEMO")
    print("=" * 60)
    
    # Generate data
    transactions, products, customers = get_all_data()
    
    # Initialize recommender
    recommender = CollaborativeFilteringRecommender()
    
    # Create user-item matrix
    recommender.create_user_item_matrix(transactions, products)
    
    # Compute similarities
    recommender.compute_user_similarity()
    recommender.compute_item_similarity()
    
    # Get a sample user
    sample_user = recommender.user_item_matrix.index[0]
    print(f"\n👤 Sample User: {sample_user}")
    
    # User-based recommendations
    print("\n📊 User-Based Recommendations:")
    user_recs = recommender.recommend_user_based(sample_user, n_recommendations=5)
    for product_id, score in user_recs:
        product_name = products[products['product_id'] == product_id]['product_name'].values[0]
        print(f"   {product_id}: {product_name} (score: {score:.4f})")
    
    # Item-based recommendations
    print("\n📊 Item-Based Recommendations:")
    item_recs = recommender.recommend_item_based(sample_user, n_recommendations=5)
    for product_id, score in item_recs:
        product_name = products[products['product_id'] == product_id]['product_name'].values[0]
        print(f"   {product_id}: {product_name} (score: {score:.4f})")
    
    # Popular items
    print("\n🔥 Most Popular Items:")
    popular = recommender.get_popular_items(5)
    for product_id, count in popular.items():
        product_name = products[products['product_id'] == product_id]['product_name'].values[0]
        print(f"   {product_id}: {product_name} ({int(count)} purchases)")
    
    # Content-based demo
    print("\n📦 Content-Based Recommendations:")
    content_rec = ContentBasedRecommender()
    content_rec.fit(products)
    
    sample_product = products['product_id'].iloc[0]
    similar = content_rec.recommend_similar(sample_product, n_recommendations=3)
    for product_id, score in similar:
        product_name = products[products['product_id'] == product_id]['product_name'].values[0]
        print(f"   {product_id}: {product_name} (similarity: {score:.4f})")


if __name__ == '__main__':
    demo()
