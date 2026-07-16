"""
E-commerce Analytics Dashboard
KPIs, Visualizations, and Business Insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class EcommerceAnalytics:
    """
    E-commerce Analytics Dashboard with KPIs and Visualizations.
    """
    
    def __init__(self):
        self.transactions = None
        self.products = None
        self.customers = None
        self.kpis = {}
    
    def load_data(self, transactions, products, customers):
        """
        Load data for analysis.
        
        Args:
            transactions: Transaction DataFrame
            products: Product catalog DataFrame
            customers: Customer DataFrame
        """
        self.transactions = transactions.copy()
        self.products = products.copy()
        self.customers = customers.copy()
        
        # Ensure date column is datetime
        if 'date' in self.transactions.columns:
            self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        
        print("✅ Data loaded successfully!")
    
    def calculate_kpis(self):
        """
        Calculate key performance indicators.
        
        Returns:
            Dictionary of KPIs
        """
        print("\n📊 Calculating KPIs...")
        
        # Basic metrics
        total_revenue = self.transactions['total'].sum()
        total_orders = self.transactions['transaction_id'].nunique()
        total_customers = self.transactions['customer_id'].nunique()
        total_products = self.transactions['product_id'].nunique()
        
        # Averages
        avg_order_value = total_revenue / total_orders
        avg_items_per_order = len(self.transactions) / total_orders
        
        # Customer metrics
        customer_df = self.transactions.groupby('customer_id').agg({
            'total': 'sum',
            'transaction_id': 'nunique'
        }).reset_index()
        customer_df.columns = ['customer_id', 'lifetime_value', 'order_count']
        
        repeat_rate = (customer_df['order_count'] > 1).sum() / len(customer_df) * 100
        
        # Product metrics
        top_product = self.transactions.groupby('product_id')['quantity'].sum().idxmax()
        top_product_name = self.products[self.products['product_id'] == top_product]['product_name'].values[0]
        
        # Store KPIs
        self.kpis = {
            'Total Revenue': total_revenue,
            'Total Orders': total_orders,
            'Total Customers': total_customers,
            'Total Products Sold': total_products,
            'Average Order Value': avg_order_value,
            'Items per Order': avg_items_per_order,
            'Customer Retention Rate': repeat_rate,
            'Top Selling Product': top_product_name
        }
        
        return self.kpis
    
    def get_sales_by_period(self, freq='M'):
        """
        Get sales aggregated by time period.
        
        Args:
            freq: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly)
        
        Returns:
            DataFrame with sales by period
        """
        if 'date' not in self.transactions.columns:
            return None
        
        sales = self.transactions.groupby(pd.Grouper(key='date', freq=freq))['total'].sum().reset_index()
        sales.columns = ['period', 'revenue']
        
        orders = self.transactions.groupby(pd.Grouper(key='date', freq=freq))['transaction_id'].nunique().reset_index()
        orders.columns = ['period', 'orders']
        
        result = sales.merge(orders, on='period')
        
        return result
    
    def get_sales_by_category(self):
        """
        Get sales breakdown by product category.
        
        Returns:
            DataFrame with sales by category
        """
        category_sales = self.transactions.groupby('category').agg({
            'total': 'sum',
            'quantity': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        category_sales.columns = ['category', 'revenue', 'units_sold', 'order_count']
        category_sales = category_sales.sort_values('revenue', ascending=False)
        
        return category_sales
    
    def get_top_products(self, n=10):
        """
        Get top N selling products.
        
        Args:
            n: Number of products
        
        Returns:
            DataFrame with top products
        """
        top_products = self.transactions.groupby('product_id').agg({
            'total': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        top_products.columns = ['product_id', 'revenue', 'units_sold', 'unique_customers']
        
        # Merge with product info
        top_products = top_products.merge(
            self.products[['product_id', 'product_name', 'category', 'price', 'rating']],
            on='product_id'
        )
        
        return top_products.nlargest(n, 'revenue')
    
    def get_top_customers(self, n=10):
        """
        Get top N customers by lifetime value.
        
        Args:
            n: Number of customers
        
        Returns:
            DataFrame with top customers
        """
        customer_metrics = self.transactions.groupby('customer_id').agg({
            'total': ['sum', 'mean', 'count'],
            'product_id': 'nunique'
        }).reset_index()
        customer_metrics.columns = ['customer_id', 'lifetime_value', 'avg_order_value', 'order_count', 'unique_products']
        
        # Merge with customer info
        customer_metrics = customer_metrics.merge(
            self.customers[['customer_id', 'name', 'segment', 'city']],
            on='customer_id'
        )
        
        return customer_metrics.nlargest(n, 'lifetime_value')
    
    def get_customer_segments(self):
        """
        Analyze customer segments.
        
        Returns:
            DataFrame with segment analysis
        """
        segments = self.transactions.groupby('customer_id').agg({
            'total': 'sum',
            'transaction_id': 'nunique'
        }).reset_index()
        segments.columns = ['customer_id', 'lifetime_value', 'order_count']
        
        # Merge with customer data
        segments = segments.merge(
            self.customers[['customer_id', 'segment']],
            on='customer_id'
        )
        
        # Segment analysis
        segment_analysis = segments.groupby('segment').agg({
            'customer_id': 'count',
            'lifetime_value': ['sum', 'mean'],
            'order_count': 'mean'
        }).reset_index()
        segment_analysis.columns = ['segment', 'customer_count', 'total_value', 'avg_value', 'avg_orders']
        
        return segment_analysis
    
    def visualize_kpis(self):
        """Create KPI visualization dashboard."""
        if not self.kpis:
            self.calculate_kpis()
        
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=list(self.kpis.keys())[:8],
            specs=[[{'type': 'indicator'}]*4]*2
        )
        
        # Add KPI values
        positions = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]
        
        for i, (kpi_name, kpi_value) in enumerate(self.kpis.items()):
            row, col = positions[i]
            
            if isinstance(kpi_value, (int, float)):
                if kpi_value > 1000:
                    display_value = f"{kpi_value:,.0f}"
                else:
                    display_value = f"{kpi_value:.2f}"
            else:
                display_value = str(kpi_value)
            
            fig.add_trace(
                go.Indicator(
                    mode='number',
                    value=kpi_value,
                    number={'font_size': 20},
                    title={'text': kpi_name, 'font_size': 10}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="E-commerce KPI Dashboard"
        )
        
        return fig
    
    def visualize_sales_trend(self, freq='M'):
        """
        Visualize sales trend over time.
        
        Args:
            freq: Time frequency
        
        Returns:
            Plotly figure
        """
        sales = self.get_sales_by_period(freq)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Revenue
        fig.add_trace(
            go.Bar(x=sales['period'], y=sales['revenue'], name='Revenue', marker_color='steelblue'),
            secondary_y=False
        )
        
        # Orders
        fig.add_trace(
            go.Scatter(x=sales['period'], y=sales['orders'], name='Orders', line=dict(color='coral')),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Sales Trend Over Time',
            xaxis_title='Period',
            yaxis_title='Revenue ($)',
            yaxis2_title='Number of Orders',
            hovermode='x unified'
        )
        
        return fig
    
    def visualize_category_distribution(self):
        """
        Visualize sales by category.
        
        Returns:
            Plotly figure
        """
        category_sales = self.get_sales_by_category()
        
        fig = px.pie(
            category_sales,
            values='revenue',
            names='category',
            title='Revenue by Category',
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def visualize_top_products(self, n=10):
        """
        Visualize top selling products.
        
        Args:
            n: Number of products
        
        Returns:
            Plotly figure
        """
        top_products = self.get_top_products(n)
        
        fig = px.bar(
            top_products,
            x='revenue',
            y='product_name',
            orientation='h',
            title=f'Top {n} Products by Revenue',
            color='revenue',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis_title='Product',
            xaxis_title='Revenue ($)',
            showlegend=False
        )
        
        return fig
    
    def visualize_customer_segments(self):
        """
        Visualize customer segments.
        
        Returns:
            Plotly figure
        """
        segments = self.get_customer_segments()
        
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'bar'}]])
        
        # Segment distribution
        fig.add_trace(
            go.Pie(
                labels=segments['segment'],
                values=segments['customer_count'],
                name='Customers',
                hole=0.4
            ),
            row=1, col=1
        )
        
        # Average lifetime value by segment
        fig.add_trace(
            go.Bar(
                x=segments['segment'],
                y=segments['avg_value'],
                name='Avg Lifetime Value'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Customer Segmentation Analysis',
            showlegend=True
        )
        
        return fig
    
    def generate_report(self):
        """
        Generate comprehensive analytics report.
        
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "=" * 60)
        print("E-COMMERCE ANALYTICS REPORT")
        print("=" * 60)
        
        # KPIs
        kpis = self.calculate_kpis()
        
        print("\n📊 KEY PERFORMANCE INDICATORS")
        print("-" * 40)
        for kpi, value in kpis.items():
            if isinstance(value, float):
                print(f"  {kpi}: ${value:,.2f}")
            else:
                print(f"  {kpi}: {value}")
        
        # Top categories
        print("\n📁 TOP CATEGORIES BY REVENUE")
        print("-" * 40)
        category_sales = self.get_sales_by_category()
        for _, row in category_sales.head(5).iterrows():
            print(f"  {row['category']}: ${row['revenue']:,.2f}")
        
        # Top products
        print("\n🏆 TOP 5 PRODUCTS")
        print("-" * 40)
        top_products = self.get_top_products(5)
        for _, row in top_products.iterrows():
            print(f"  {row['product_name']}: ${row['revenue']:,.2f} ({int(row['units_sold'])} sold)")
        
        # Customer segments
        print("\n👥 CUSTOMER SEGMENTS")
        print("-" * 40)
        segments = self.get_customer_segments()
        for _, row in segments.iterrows():
            print(f"  {row['segment']}: {int(row['customer_count'])} customers, ${row['avg_value']:.2f} avg value")
        
        return {
            'kpis': kpis,
            'category_sales': category_sales,
            'top_products': top_products,
            'customer_segments': segments
        }


def demo():
    """Demonstrate the analytics dashboard."""
    from data_generator import generate_transactions
    
    print("=" * 60)
    print("E-COMMERCE ANALYTICS DASHBOARD DEMO")
    print("=" * 60)
    
    # Generate data
    transactions, products, customers = generate_transactions(n_transactions=2000)
    
    # Initialize analytics
    analytics = EcommerceAnalytics()
    analytics.load_data(transactions, products, customers)
    
    # Generate report
    report = analytics.generate_report()
    
    # Visualizations (save to files for demo)
    print("\n📈 Generating visualizations...")
    
    # Sales trend
    sales_trend = analytics.visualize_sales_trend('W')
    sales_trend.write_html('sales_trend.html')
    
    # Category distribution
    cat_dist = analytics.visualize_category_distribution()
    cat_dist.write_html('category_distribution.html')
    
    # Top products
    top_prods = analytics.visualize_top_products(10)
    top_prods.write_html('top_products.html')
    
    print("✅ Visualizations saved as HTML files")
    print("   - sales_trend.html")
    print("   - category_distribution.html")
    print("   - top_products.html")


if __name__ == '__main__':
    demo()
