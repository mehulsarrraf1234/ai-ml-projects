"""
Market Basket Analysis
Association Rule Mining using Apriori Algorithm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')


class MarketBasketAnalyzer:
    """
    Market Basket Analysis using Association Rule Mining.
    """
    
    def __init__(self):
        self.basket = None
        self.frequent_itemsets = None
        self.rules = None
    
    def create_basket(self, transactions, min_support=0.01):
        """
        Create one-hot encoded basket and find frequent itemsets.
        
        Args:
            transactions: Transaction data
            min_support: Minimum support threshold
        
        Returns:
            Frequent itemsets DataFrame
        """
        print("🔄 Creating basket matrix...")
        
        # Group by transaction and product
        basket = transactions.groupby(['transaction_id', 'product_id'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('transaction_id')
        
        # Convert to binary (1 = purchased, 0 = not purchased)
        self.basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        print(f"✅ Basket created: {self.basket.shape[0]} transactions, {self.basket.shape[1]} products")
        
        return self.basket
    
    def find_frequent_itemsets(self, min_support=0.01, method='apriori'):
        """
        Find frequent itemsets using Apriori or FP-Growth.
        
        Args:
            min_support: Minimum support threshold
            method: 'apriori' or 'fpgrowth'
        
        Returns:
            Frequent itemsets DataFrame
        """
        print(f"🔄 Finding frequent itemsets using {method}...")
        
        if method == 'apriori':
            self.frequent_itemsets = apriori(
                self.basket,
                min_support=min_support,
                use_colnames=True
            )
        else:
            self.frequent_itemsets = fpgrowth(
                self.basket,
                min_support=min_support,
                use_colnames=True
            )
        
        # Add length column
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
        print(f"✅ Found {len(self.frequent_itemsets)} frequent itemsets")
        
        return self.frequent_itemsets
    
    def generate_rules(self, metric='confidence', min_threshold=0.5):
        """
        Generate association rules from frequent itemsets.
        
        Args:
            metric: Metric to sort by ('support', 'confidence', 'lift')
            min_threshold: Minimum threshold for the metric
        
        Returns:
            Association rules DataFrame
        """
        if self.frequent_itemsets is None:
            print("⚠️ Find frequent itemsets first!")
            return None
        
        print("🔄 Generating association rules...")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        # Filter rules with at least 2 items (antecedent + consequent)
        self.rules = self.rules[self.rules['consequents'].apply(lambda x: len(x) + len(self.rules.loc[self.rules.index[0], 'antecedents'])) >= 2]
        
        # Re-filter properly
        self.rules = self.rules[self.rules['antecedents'].apply(len) + self.rules['consequents'].apply(len) >= 2]
        
        # Sort by lift
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        print(f"✅ Generated {len(self.rules)} association rules")
        
        return self.rules
    
    def get_top_rules(self, n=10, sort_by='lift'):
        """
        Get top N association rules.
        
        Args:
            n: Number of rules to return
            sort_by: Column to sort by
        
        Returns:
            Top N rules
        """
        if self.rules is None:
            return None
        
        return self.rules.nlargest(n, sort_by)
    
    def analyze_category_rules(self, transactions):
        """
        Analyze rules at category level instead of product level.
        
        Args:
            transactions: Transaction data with category column
        
        Returns:
            Category-level association rules
        """
        print("🔄 Analyzing category-level rules...")
        
        # Create basket at category level
        category_basket = transactions.groupby(['transaction_id', 'category'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('transaction_id')
        category_basket = category_basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Find frequent itemsets
        freq_categories = apriori(category_basket, min_support=0.05, use_colnames=True)
        
        # Generate rules
        cat_rules = association_rules(freq_categories, metric='confidence', min_threshold=0.3)
        
        return freq_categories, cat_rules
    
    def visualize_rules(self, n_top=20):
        """
        Visualize top association rules.
        
        Args:
            n_top: Number of top rules to visualize
        """
        if self.rules is None or len(self.rules) == 0:
            print("⚠️ No rules to visualize. Generate rules first!")
            return
        
        top_rules = self.rules.head(n_top)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Support vs Confidence
        ax1 = axes[0, 0]
        scatter = ax1.scatter(top_rules['support'], top_rules['confidence'],
                             c=top_rules['lift'], cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Support')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Support vs Confidence (colored by Lift)')
        plt.colorbar(scatter, ax=ax1, label='Lift')
        
        # 2. Top rules by lift
        ax2 = axes[0, 1]
        top_lift = self.rules.nlargest(10, 'lift')
        y_pos = range(len(top_lift))
        ax2.barh(y_pos, top_lift['lift'], color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([str(list(x)[:2]) for x in top_lift['antecedents']], fontsize=8)
        ax2.set_xlabel('Lift')
        ax2.set_title('Top 10 Rules by Lift')
        
        # 3. Distribution of metrics
        ax3 = axes[1, 0]
        ax3.hist(self.rules['support'], bins=20, alpha=0.7, color='steelblue')
        ax3.set_xlabel('Support')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Support')
        
        # 4. Confidence distribution
        ax4 = axes[1, 1]
        ax4.hist(self.rules['confidence'], bins=20, alpha=0.7, color='coral')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Confidence')
        
        plt.tight_layout()
        plt.savefig('market_basket_analysis.png', dpi=150)
        plt.show()
        
        print("📊 Visualization saved to 'market_basket_analysis.png'")
    
    def get_product_recommendations(self, product_id, n_recommendations=3):
        """
        Get product recommendations based on association rules.
        
        Args:
            product_id: Product ID to base recommendations on
            n_recommendations: Number of recommendations
        
        Returns:
            List of recommended products
        """
        if self.rules is None:
            return []
        
        # Find rules where the product is in antecedent
        relevant_rules = self.rules[
            self.rules['antecedents'].apply(lambda x: product_id in x)
        ]
        
        # Sort by lift and get consequents
        recommendations = []
        for _, rule in relevant_rules.nlargest(n_recommendations, 'lift').iterrows():
            for product in rule['consequents']:
                if product != product_id:
                    recommendations.append({
                        'product': product,
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'support': rule['support']
                    })
        
        return recommendations[:n_recommendations]


def demo():
    """Demonstrate market basket analysis."""
    from data_generator import generate_transactions
    
    print("=" * 60)
    print("MARKET BASKET ANALYSIS DEMO")
    print("=" * 60)
    
    # Generate data
    transactions, products, customers = generate_transactions(n_transactions=1000)
    
    # Initialize analyzer
    analyzer = MarketBasketAnalyzer()
    
    # Create basket
    basket = analyzer.create_basket(transactions)
    print(f"\n📦 Basket shape: {basket.shape}")
    
    # Find frequent itemsets
    freq_itemsets = analyzer.find_frequent_itemsets(min_support=0.01)
    
    print("\n📊 Top 10 Frequent Itemsets:")
    top_itemsets = freq_itemsets.nlargest(10, 'support')
    for idx, row in top_itemsets.iterrows():
        print(f"   {list(row['itemsets'])}: support={row['support']:.4f}")
    
    # Generate rules
    rules = analyzer.generate_rules(metric='confidence', min_threshold=0.3)
    
    print("\n📋 Top 10 Association Rules:")
    top_rules = analyzer.get_top_rules(10)
    for idx, rule in top_rules.iterrows():
        print(f"   {list(rule['antecedents'])} -> {list(rule['consequents'])}")
        print(f"      support={rule['support']:.3f}, confidence={rule['confidence']:.3f}, lift={rule['lift']:.3f}")
    
    # Visualize
    print("\n📈 Generating visualizations...")
    analyzer.visualize_rules(20)
    
    # Product recommendations
    print("\n🎯 Product Recommendations based on Association Rules:")
    sample_product = products['product_id'].iloc[0]
    recs = analyzer.get_product_recommendations(sample_product, n_recommendations=3)
    print(f"\nProducts recommended with '{sample_product}':")
    for rec in recs:
        print(f"   {rec['product']}: confidence={rec['confidence']:.3f}, lift={rec['lift']:.3f}")
    
    # Category-level analysis
    print("\n📁 Category-Level Analysis:")
    cat_freq, cat_rules = analyzer.analyze_category_rules(transactions)
    
    if len(cat_rules) > 0:
        print("\nTop Category Association Rules:")
        for idx, rule in cat_rules.nlargest(5, 'lift').iterrows():
            print(f"   {list(rule['antecedents'])} -> {list(rule['consequents'])}")
            print(f"      confidence={rule['confidence']:.3f}, lift={rule['lift']:.3f}")


if __name__ == '__main__':
    demo()
