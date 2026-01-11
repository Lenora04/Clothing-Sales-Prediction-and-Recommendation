"""
Recommendation and helper functions for the dashboard.
"""
import pandas as pd
import numpy as np



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


def sales_improvement_hints(input_data, model, df):
    """
    Generate 'what-if' suggestions using the same prediction path as the app.
    """
    base_result = predict_sales_volume(to_df(input_data), model, df)
    base_pred = base_result["prediction"]

    suggestions = []

    def try_change(label, new_data):
        result = predict_sales_volume(to_df(new_data), model, df)
        if result:
            delta = result["prediction"] - base_pred
            if delta > 0:
                suggestions.append((label, delta))

    # 1️⃣ Promotion
    if input_data.get("promotion", 0) == 0:
        tmp = dict(input_data)
        tmp["promotion"] = 1
        try_change("📢 Add promotion", tmp)

    # 2️⃣ Price reduction
    if input_data.get("price", 0) > 5:
        tmp = dict(input_data)
        tmp["price"] *= 0.9
        try_change("💰 Reduce price by 10%", tmp)

    # 3️⃣ Longer description
    if input_data.get("desc_word_count", 0) < 20:
        tmp = dict(input_data)
        tmp["desc_word_count"] += 10
        try_change("📝 Improve product description", tmp)

    # 4️⃣ Popular material
    top_material = df["material"].value_counts().idxmax()
    if input_data.get("material") != top_material:
        tmp = dict(input_data)
        tmp["material"] = top_material
        try_change(f"🧵 Switch material to {top_material}", tmp)

    suggestions.sort(key=lambda x: x[1], reverse=True)
    return base_pred, suggestions

def to_df(d):
    """
    Converts input dictionary to a DataFrame and ensures all 
    engineered features required by the model are present.
    """
    input_df = pd.DataFrame([d])
    
    # Ensure engineered features exist (same logic as in app.py)
    if 'description' in input_df.columns:
        input_df['desc_len'] = input_df['description'].astype(str).str.len()
        input_df['desc_word_count'] = input_df['description'].astype(str).str.split().str.len()
    
    if 'name' in input_df.columns:
        input_df['name_len'] = input_df['name'].astype(str).str.len()
        input_df['name_word_count'] = input_df['name'].astype(str).str.split().str.len()

    # Avoid division by zero
    input_df['price_per_word'] = input_df['price'] / (input_df.get('desc_word_count', 0) + 1)
    
    return input_df
