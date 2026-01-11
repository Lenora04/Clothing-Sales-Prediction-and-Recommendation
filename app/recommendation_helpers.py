"""
Recommendation and helper functions for the dashboard.
"""
import pandas as pd
import numpy as np





def explain_recommendation(query_product_id, rec_product_id, df):
    try:
        df_ids = df.copy()
        df_ids['product_id'] = df_ids['product_id'].astype(str)

        query_id = str(query_product_id)
        rec_id = str(rec_product_id)

        query_rows = df_ids[df_ids['product_id'] == query_id]
        rec_rows = df_ids[df_ids['product_id'] == rec_id]

        if query_rows.empty or rec_rows.empty:
            return "Recommended due to overall similarity across product features."

        q = query_rows.iloc[0]
        r = rec_rows.iloc[0]

        reasons = []

        #  Strong categorical matches
        if q['material'] == r['material']:
            reasons.append(f" Same material ({q['material']})")

        if q['section'] == r['section']:
            reasons.append(f" Same section ({q['section']})")

        if q['season'] == r['season']:
            reasons.append(f" Same season ({q['season']})")

        if 'brand' in df.columns and q.get('brand') == r.get('brand'):
            reasons.append(f" Same brand ({q['brand']})")

        # Text overlap (simple + fast)
        q_words = set(str(q['name']).lower().split())
        r_words = set(str(r['name']).lower().split())
        common_words = q_words & r_words

        if len(common_words) >= 2:
            reasons.append("📝 Similar product name keywords")

        # 3️Price proximity (soft signal)
        try:
            price_diff = abs(float(q['price']) - float(r['price']))
            if price_diff <= 5:
                reasons.append("💰 Similar price range")
        except:
            pass

        #  Final output
        if reasons:
            return " | ".join(reasons[:3])

        return "Recommended based on overall similarity in text and metadata."

    except Exception:
        return "Recommended based on overall product similarity."




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


def recommend(product_id, product_id_to_index, sim_matrix, df, top_n=5, include_score=False):
    try:
        if product_id not in product_id_to_index:
            return pd.DataFrame() # Return empty instead of None
        
        idx = product_id_to_index[product_id]
        
        # 🛑 SAFETY CHECK: Ensure the index is within the sliced matrix bounds
        if idx >= sim_matrix.shape[0]:
            return pd.DataFrame()

        scores = sim_matrix[idx]
        
        # Get top indices (excluding the product itself)
        top_indices = np.argsort(scores)[::-1]
        top_indices = top_indices[top_indices != idx][:top_n]
        
        results = df.iloc[top_indices].copy()
        
        if include_score:
            results['score'] = scores[top_indices]
            
        # Ensure price and score are numeric for formatting, others as strings
        for col in results.columns:
            if col not in ['price', 'score', 'sales_volume']:
                results[col] = results[col].astype(str)

        return results
    except Exception as e:
        print(f"Error in recommend: {e}")
        return pd.DataFrame()

def to_df(d):
    """
    Converts input dictionary to a DataFrame and ensures all 
    engineered features required by the model are present.
    """
    input_df = pd.DataFrame([d])
    
    # Ensure all strings are strings
    input_df['description'] = input_df['description'].astype(str)
    input_df['name'] = input_df['name'].astype(str)
    
    # Feature Engineering (MUST match the training logic exactly)
    input_df['desc_len'] = input_df['description'].str.len()
    input_df['name_len'] = input_df['name'].str.len()
    input_df['desc_word_count'] = input_df['description'].str.split().str.len()
    input_df['name_word_count'] = input_df['name'].str.split().str.len()

    # Derived Features
    input_df['price_per_word'] = input_df['price'] / (input_df['desc_word_count'] + 1)
    input_df['price_per_name_word'] = input_df['price'] / (input_df['name_word_count'] + 1)
    
    return input_df

def sales_improvement_hints(input_data, model, df):
    """
    Generate 'what-if' suggestions.
    """
    # Use to_df to ensure engineered features are calculated before prediction
    base_df = to_df(input_data)
    base_result = predict_sales_volume(base_df, model, df)
    base_pred = base_result["prediction"]

    suggestions = []

    def try_change(label, new_data):
        # Always run through to_df to recalculate engineered features like price_per_word
        test_df = to_df(new_data)
        result = predict_sales_volume(test_df, model, df)
        if result:
            delta = result["prediction"] - base_pred
            if delta > 1: # Only suggest if improvement is significant
                suggestions.append((label, delta))

    # 1. Promotion Logic
    if input_data.get("promotion") == 0:
        tmp = input_data.copy()
        tmp["promotion"] = 1
        try_change("📢 Enable Promotion", tmp)

    # 2. Price reduction (10% off)
    tmp_price = input_data.copy()
    tmp_price["price"] *= 0.9
    try_change("💰 10% Price Discount", tmp_price)

    suggestions.sort(key=lambda x: x[1], reverse=True)
    return base_pred, suggestions
