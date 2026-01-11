import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    get_data_cached, 
    get_models_cached, 
    get_sales_predictor_cached, 
    get_sim_matrix_cached
)
from recommendation_helpers import (
    recommend, explain_recommendation, predict_sales_volume, 
    get_similar_products_by_sales, sales_improvement_hints
)

def main():
    # ==============================================================================
    # 1. INITIALIZATION & CONFIG
    # ==============================================================================
    st.set_page_config(
        page_title="Clothing Sales Recommender",
        page_icon="👕",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main { padding: 20px; }
        .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: #f9f9f9; min-height: 200px; }
        .metric-card { text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 10px; }
        .stButton > button { background-color: #5ea879; color: white; border-color: #10b981; width: 100%; }
        .stButton > button:hover { background-color: #07db54; color: white; border-color: #16a34a; }
    </style>
    """, unsafe_allow_html=True)

    # Load dataset
    df, product_id_to_index = get_data_cached()

    # Initialize Session State for heavy models
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'svd' not in st.session_state:
        st.session_state.svd = None
    if 'sales_model' not in st.session_state:
        st.session_state.sales_model = None
    if 'sim_matrix' not in st.session_state:
        st.session_state.sim_matrix = None

    # ==============================================================================
    # 2. SIDEBAR NAVIGATION
    # ==============================================================================
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Sales Volume Predictor", "Product Search & Recommendations", "Analytics Dashboard", "Market Insights"]
    )

    # ==============================================================================
    # PAGE: HOME
    # ==============================================================================
    if page == "Home":
        st.title("👕 Clothing Sales Recommendation System")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>📦 Products</h3><p style="font-size: 32px; margin: 0;">{len(df):,}</p></div>', unsafe_allow_html=True)
        with col2:
            unique_brands = [b for b in df['brand'].unique() if str(b).lower() != 'missing']
            st.markdown(f'<div class="metric-card"><h3>🏷️ Brands</h3><p style="font-size: 24px; margin: 0;">{", ".join(unique_brands[:3])}...</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>💰 Avg Price</h3><p style="font-size: 32px; margin: 0;">${df["price"].mean():.2f}</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("About This System")
        st.write("This tool provides sales forecasting and product recommendations using Hybrid Filtering and Gradient Boosting.")

    # ==============================================================================
    # PAGE: SALES VOLUME PREDICTOR
    # ==============================================================================
    elif page == "Sales Volume Predictor":
        st.title("Sales Volume Predictor")
        
        if st.session_state.sales_model is None:
            st.session_state.sales_model, _ = get_sales_predictor_cached()

        st.subheader("📝 Product Features")
        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input("💰 Price ($)", 0.0, 1000.0, 50.0)
            promotion = st.selectbox("📢 Promotion", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            seasonal = st.selectbox("🌡️ Seasonal", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            material = st.selectbox("🧵 Material", df['material'].unique().tolist())
            origin = st.selectbox("🌏 Origin", df['origin'].unique().tolist())
            # Handling potential missing column 'terms' gracefully
            terms_list = df['terms'].unique().tolist() if 'terms' in df.columns else ["Standard"]
            terms = st.selectbox("🔍 Product Term", terms_list)
            
        with col2:
            section = st.selectbox("🏗️ Section", df['section'].unique().tolist())
            season = st.selectbox("🌍 Season", df['season'].unique().tolist())
            product_position = st.selectbox("📍 Position", ['Aisle', 'End-cap', 'Front of Store'])
            product_name = st.text_input("🏷️ Product Name", "New Summer Shirt")
            product_description = st.text_area("📄 Description", "Comfortable cotton shirt.")

        if st.button("🔮 Predict Sales Volume"):
            input_df = pd.DataFrame([{
                'name': product_name, 'description': product_description, 'price': price,
                'promotion': promotion, 'seasonal': seasonal, 'section': section,
                'season': season, 'material': material, 'origin': origin, 
                'product_position': product_position, 'terms': terms
            }])
            
            # Feature engineering for prediction
            input_df['desc_len'] = input_df['description'].str.len()
            input_df['name_len'] = input_df['name'].str.len()
            input_df['desc_word_count'] = input_df['description'].str.split().str.len()
            input_df['price_per_word'] = input_df['price'] / (input_df['desc_word_count'] + 1)
            
            try:
                result = predict_sales_volume(input_df, st.session_state.sales_model, df)
                if result:
                    st.success(f"Predicted Sales Volume: {int(result['prediction'])}")
                    st.metric("Percentile Rank", f"{result['percentile']:.1f}%")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # ==============================================================================
    # PAGE: PRODUCT SEARCH & RECOMMENDATIONS
    # ==============================================================================
    elif page == "Product Search & Recommendations":
        st.title("Product Search & Recommendations")
        
        if st.session_state.sim_matrix is None:
            with st.status("Loading Recommendation Engine...", expanded=True) as status:
                st.session_state.pipeline, st.session_state.svd, _ = get_models_cached()
                st.session_state.sim_matrix = get_sim_matrix_cached()
                status.update(label="Ready!", state="complete", expanded=False)

        if 'product_names_cache' not in st.session_state:
            st.session_state.product_names_cache = df['name'].unique().tolist()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_product_name = st.selectbox("Select a Product", st.session_state.product_names_cache)
        with col2:
            top_n = st.number_input("Top N", 1, 20, 5)
        
        if selected_product_name:
            match = df[df['name'] == selected_product_name]
            if not match.empty:
                pid = match.iloc[0]['product_id']
                idx = product_id_to_index.get(pid)

                if idx is None or idx >= st.session_state.sim_matrix.shape[0]:
                    st.warning("⚠️ This product is not in the Lite Demo range. Please try products from the top of the list.")
                else:
                    try:
                        recs = recommend(pid, product_id_to_index, st.session_state.sim_matrix, df, top_n=top_n, include_score=True)
                        if not recs.empty:
                            cols = st.columns(3)
                            for i, (_, row) in enumerate(recs.iterrows()):
                                with cols[i % 3]:
                                    st.markdown(f'<div class="card"><h4>{row["name"][:30]}...</h4><p>Price: ${row["price"]:.2f}</p><p>Score: {row.get("score",0):.3f}</p></div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Recommendation Error: {e}")

    # ==============================================================================
    # PAGE: ANALYTICS & INSIGHTS
    # ==============================================================================
    elif page == "Analytics Dashboard":
        st.title("Analytics Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df['section'].value_counts(), title="Products by Section")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df, names='season', title="Seasonal Distribution")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Market Insights":
        st.title("Market Insights")
        top_brands = df.groupby('brand')['sales_volume'].sum().nlargest(10).reset_index()
        fig = px.bar(top_brands, x='sales_volume', y='brand', orientation='h', title="Top Brands by Volume")
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("Optimized for Streamlit Cloud")

# This is the crucial part that stops the error
if __name__ == "__main__":
    main()
