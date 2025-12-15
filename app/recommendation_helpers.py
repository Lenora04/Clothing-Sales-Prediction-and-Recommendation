"""
Recommendation and helper functions for the dashboard.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend(product_id, product_id_to_index, sim_matrix, df, top_n=5, include_score=False):
    """
    Get top-N product recommendations based on similarity matrix.
    
    Args:
        product_id (str): Target product ID
        product_id_to_index (dict): Mapping from product_id to DataFrame index
        sim_matrix (np.ndarray): Precomputed similarity matrix
        df (pd.DataFrame): Product dataframe
        top_n (int): Number of recommendations to return
        include_score (bool): Whether to include similarity scores
    
    Returns:
        pd.DataFrame: Recommended products with details
    """
    try:
        if product_id not in product_id_to_index:
            return None
        
        idx = product_id_to_index[product_id]
        scores = sim_matrix[idx]
        
        # Get top indices (excluding the product itself)
        top_indices = np.argsort(scores)[::-1]
        top_indices = top_indices[top_indices != idx][:top_n]
        
        results = df.iloc[top_indices].copy()
        results = results[['product_id', 'name', 'section', 'price', 'season', 'material', 'url']].reset_index(drop=True)
        
        if include_score:
            results['score'] = scores[top_indices]
        
        # Convert to string to ensure Arrow compatibility
        #results = results.astype(str)

        for col in results.columns:
            if col not in ['price', 'score']:
                results[col] = results[col].astype(str)


        return results
    except Exception as e:
        print(f"Error in recommend: {e}")
        return None


def explain_recommendation(query_product_id, rec_product_id, product_id_to_index, pipeline, df):
    """
    Generate a simple explanation for why a product is recommended.
    
    Args:
        query_product_id (str): Query product ID
        rec_product_id (str): Recommended product ID
        product_id_to_index (dict): Mapping from product_id to DataFrame index
        pipeline: Scikit-learn pipeline with TF-IDF vectorizer
        df (pd.DataFrame): Product dataframe
    
    Returns:
        str: Explanation text
    """
    try:
        query_idx = product_id_to_index[query_product_id]
        rec_idx = product_id_to_index[rec_product_id]
        
        query_product = df.iloc[query_idx]
        rec_product = df.iloc[rec_idx]
        
        # Find common attributes
        common_attrs = []
        
        attrs_to_check = ['section', 'season', 'material', 'brand']
        for attr in attrs_to_check:
            if query_product.get(attr) == rec_product.get(attr):
                common_attrs.append(f"{attr}: {query_product[attr]}")
        
        if common_attrs:
            return "Shared attributes: " + ", ".join(common_attrs)
        else:
            return "Similar text features and product metadata"
    except Exception as e:
        return f"Error generating explanation: {e}"


def visualize_products(products_df, title="Products Visualization"):
    """
    Create a simple visualization of products.
    
    Args:
        products_df (pd.DataFrame): Product dataframe
        title (str): Title for visualization
    
    Returns:
        dict: Visualization data
    """
    return {
        'total': len(products_df),
        'avg_price': products_df['price'].mean(),
        'price_range': (products_df['price'].min(), products_df['price'].max()),
        'sections': products_df['section'].value_counts().to_dict(),
        'materials': products_df['material'].value_counts().to_dict(),
    }


def predict_sales_volume(input_df, model, df):
    """
    Predict sales volume using a full sklearn pipeline.
    """
    pred_log = model.predict(input_df)
    pred_log = float(pred_log[0])
    prediction = float(np.expm1(pred_log))
    

    mean = df['sales_volume'].mean()
    median = df['sales_volume'].median()
    std = df['sales_volume'].std()

    percentile = (df['sales_volume'] <= prediction).mean() * 100

    return {
        'prediction': prediction,
        'dataset_mean': mean,
        'dataset_median': median,
        'dataset_std': std,
        'percentile': percentile
    }


def get_similar_products_by_sales(target_sales, df, tolerance=0.1, top_n=5):
    """
    Get products with similar sales volumes.
    
    Args:
        target_sales (float): Target sales volume
        df (pd.DataFrame): Product dataframe
        tolerance (float): Tolerance as % of target sales
        top_n (int): Number of products to return
    
    Returns:
        pd.DataFrame: Similar products
    """
    threshold = target_sales * tolerance
    similar = df[
        (df['sales_volume'] >= target_sales - threshold) &
        (df['sales_volume'] <= target_sales + threshold)
    ].nlargest(top_n, 'sales_volume')[['product_id', 'name', 'section', 'price', 'sales_volume', 'material']]
    
    return similar.astype(str)
