import sys
import os

# 1. SETUP PATHS & MODULE ALIASING (CRITICAL FOR JOBLIB)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import transformers  # Your transformers.py file
sys.modules['app'] = sys.modules[__name__]
sys.modules['app.transformers'] = transformers

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    get_data_cached, get_models_cached, 
    get_sales_predictor_cached, get_sim_matrix_cached
)
from recommendation_helpers import (
    recommend, explain_recommendation, predict_sales_volume, 
    to_df, sales_improvement_hints
)

def main():
    # ==============================================================================
    # PAGE CONFIG & CSS
    # ==============================================================================
    st.set_page_config(page_title="Clothing Sales Recommender", page_icon="👕", layout="wide")

    st.markdown("""
    <style>
        .metric-card { text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 10px; }
        .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: #f9f9f9; }
    </style>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # LOAD DATA
    # ==============================================================================
    df, product_id_to_index = get_data_cached()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
        ["Home", "Sales Volume Predictor", "Product Search & Recommendations", "Analytics Dashboard", "Market Insights"])

    # --- PAGE: HOME ---
    if page == "Home":
        st.title("👕 Clothing Sales Recommendation System")
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>📦 Products</h3><p style="font-size: 32px; margin: 0;">{len(df):,}</p></div>', unsafe_allow_html=True)
        with col2:
            unique_brands = df['brand'].unique()
            clean_brands = [str(b) for b in unique_brands if str(b).lower() != 'missing']
            st.markdown(f'<div class="metric-card"><h3>🏷️ Brands</h3><p style="font-size: 24px; margin: 0;">{", ".join(clean_brands[:3])}...</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>💰 Avg Price</h3><p style="font-size: 32px; margin: 0;">${df["price"].mean():.2f}</p></div>', unsafe_allow_html=True)
        
        st.subheader("About This System")
        st.write("This hybrid engine combines Content-Based Filtering, Text Similarity, and Gradient Boosting for Sales Prediction.")

    # --- PAGE: SALES VOLUME PREDICTOR ---
    elif page == "Sales Volume Predictor":
        st.title("🔮 Sales Volume Predictor")
        if st.session_state.get('sales_model') is None:
            st.session_state.sales_model, _ = get_sales_predictor_cached()

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Product Name", "New Summer Shirt")
                price = st.number_input("Price ($)", 0.0, 1000.0, 50.0)
                promo = st.selectbox("Promotion", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                material = st.selectbox("Material", df['material'].unique())
                terms = st.selectbox("Product Term", df['terms'].unique().tolist())
            with col2:
                section = st.selectbox("Section", df['section'].unique())
                season = st.selectbox("Season", df['season'].unique())
                product_position = st.selectbox("Position", ['Aisle', 'End-cap', 'Front of Store'])
                origin = st.selectbox("Origin", df['origin'].unique())
                desc = st.text_area("Description", "Enter product details...")
            submit = st.form_submit_button("Predict Sales Volume")

        if submit:
            input_data = {"name": name, "description": desc, "price": price, "promotion": promo, "material": material, "section": section, "season": season, "seasonal": 1 if season != "All-Season" else 0, "origin": origin, "product_position": product_position, "terms": terms}
            input_df = to_df(input_data)
            res = predict_sales_volume(input_df, st.session_state.sales_model, df)
            
            st.success(f"### Predicted Sales Volume: {int(res['prediction'])} units")
            st.metric("Percentile Rank", f"{res['percentile']:.1f}%")
            
            st.markdown("---")
            st.subheader("💡 Optimization Suggestions")
            base_pred, hints = sales_improvement_hints(input_data, st.session_state.sales_model, df)
            for label, delta in hints[:4]:
                st.success(f"{label} → **+{int(delta)} sales units**")

    # --- PAGE: RECOMMENDATIONS ---
    elif page == "Product Search & Recommendations":
        st.title("🎯 Product Recommendations")
        if st.session_state.get('sim_matrix') is None:
            st.session_state.pipeline, st.session_state.svd, _ = get_models_cached()
            st.session_state.sim_matrix = get_sim_matrix_cached()

        selected_name = st.selectbox("Select Product", df['name'].unique()[:500])
        query_row = df[df['name'] == selected_name].iloc[0]
        
        if st.button("Generate Recommendations"):
            recs = recommend(query_row['product_id'], product_id_to_index, st.session_state.sim_matrix, df, top_n=6, include_score=True)
            if not recs.empty:
                cols = st.columns(3)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % 3]:
                        st.markdown(f'<div class="card"><h4>{row["name"][:35]}...</h4><p>Price: ${row["price"]:.2f}<br>Score: {row["score"]:.3f}</p></div>', unsafe_allow_html=True)
                        st.caption(f"Reason: {explain_recommendation(query_row['product_id'], row['product_id'], df)}")
            else:
                st.warning("No recommendations found.")

    # --- PAGE: ANALYTICS DASHBOARD (RESTORED FROM YOUR ORIGINAL) ---
    elif page == "Analytics Dashboard":
        st.title("📊 Analytics Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products", len(df))
        col2.metric("Unique Brands", df['brand'].nunique())
        col3.metric("Avg Sales Vol", f"{df['sales_volume'].mean():.0f}")
        col4.metric("Price Range", f"${df['price'].min():.0f}-${df['price'].max():.0f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📦 Products by Section")
            fig = px.bar(df['section'].value_counts(), labels={'index': 'Section', 'value': 'Count'}, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("🌍 Products by Season")
            fig = px.pie(df, names='season', title="Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("💹 Price vs Sales Volume")
        fig = px.scatter(df.sample(min(1000, len(df))), x='price', y='sales_volume', color='section', hover_name='name')
        st.plotly_chart(fig, use_container_width=True)

    # --- PAGE: MARKET INSIGHTS (RESTORED FROM YOUR ORIGINAL) ---
    elif page == "Market Insights":
        st.title("📈 Market Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Top Brands by Sales")
            top_brands = df.groupby('brand')['sales_volume'].sum().nlargest(10)
            fig = px.bar(top_brands, orientation='h', labels={'value': 'Volume', 'brand': 'Brand'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("🧵 Popular Materials")
            material_sales = df.groupby('material')['sales_volume'].sum().nlargest(10)
            fig = px.bar(material_sales, labels={'value': 'Total Sales', 'material': 'Material'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌡️ Sales Trends by Season and Section")
        season_section = df.groupby(['season', 'section'])['sales_volume'].sum().reset_index()
        fig = px.bar(season_section, x='season', y='sales_volume', color='section', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
