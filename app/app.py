import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import transformers 

sys.modules['app'] = sys.modules[__name__]
sys.modules['app.transformers'] = transformers

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    get_data_cached, get_models_cached, 
    get_sales_predictor_cached, get_sim_matrix_cached
)
from recommendation_helpers import (
    recommend, explain_recommendation, predict_sales_volume, 
    to_df, sales_improvement_hints
)

def main():
    # 1. Setup
    st.set_page_config(page_title="Clothing Sales Pro", page_icon="👕", layout="wide")
    
    # Custom CSS for the cards you liked
    st.markdown("""
        <style>
        .metric-card { text-align: center; padding: 15px; background: #764ba2; color: white; border-radius: 10px; }
        .card { border: 1px solid #ddd; border-radius: 5px; padding: 10px; background: #f9f9f9; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

    # 2. Load Data (Always needed)
    df, product_id_to_index = get_data_cached()

    # 3. Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
        ["Home", "Sales Volume Predictor", "Product Search & Recommendations", "Analytics Dashboard", "Market Insights"])

    # --- PAGE: HOME ---
    if page == "Home":
        st.title("👕 Clothing Sales Recommendation System")
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><h3>📦 Products</h3><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><h3>💰 Avg Price</h3><h2>${df["price"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><h3>📈 Total Vol</h3><h2>{df["sales_volume"].sum():,}</h2></div>', unsafe_allow_html=True)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # --- PAGE: SALES VOLUME PREDICTOR ---
    elif page == "Sales Volume Predictor":
        st.title("🔮 Sales Volume Predictor")
        if st.session_state.get('sales_model') is None:
            st.session_state.sales_model, _ = get_sales_predictor_cached()

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Product Name", "New Summer Shirt")
                price = st.number_input("Price ($)", 0.0, 1000.0, 45.0)
                promo = st.selectbox("Promotion", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                material = st.selectbox("Material", df['material'].unique())
            with col2:
                section = st.selectbox("Section", df['section'].unique())
                season = st.selectbox("Season", df['season'].unique())
                origin = st.selectbox("Origin", df['origin'].unique())
                desc = st.text_area("Description", "Enter product details...")
            
            submit = st.form_submit_button("Predict Sales Volume")

        if submit:
            input_data = {
                "name": name, "description": desc, "price": price, "promotion": promo,
                "material": material, "section": section, "season": season, 
                "seasonal": 1 if season != "All-Season" else 0, "origin": origin,
                "product_position": "Aisle"
            }
            # Use to_df helper for feature engineering
            input_df = to_df(input_data)
            res = predict_sales_volume(input_df, st.session_state.sales_model, df)
            
            st.success(f"### Predicted Sales Volume: {int(res['prediction'])} units")
            
            # Show Improvement Hints
            st.markdown("---")
            st.subheader("💡 Optimization Suggestions")
            base_pred, hints = sales_improvement_hints(input_data, st.session_state.sales_model, df)
            for label, delta in hints:
                st.write(f"**{label}**: Could increase sales by ~{int(delta)} units.")

    # --- PAGE: RECOMMENDATIONS ---
    elif page == "Product Search & Recommendations":
        st.title("🎯 Product Recommendations")
        if st.session_state.get('sim_matrix') is None:
            st.session_state.pipeline, st.session_state.svd, _ = get_models_cached()
            st.session_state.sim_matrix = get_sim_matrix_cached()

        selected_name = st.selectbox("Select a Product to find matches", df['name'].unique()[:500])
        query_row = df[df['name'] == selected_name].iloc[0]
        
        if st.button("Generate Recommendations"):
            recs = recommend(query_row['product_id'], product_id_to_index, st.session_state.sim_matrix, df, top_n=6, include_score=True)
            
            if not recs.empty:
                cols = st.columns(3)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % 3]:
                        st.markdown(f"""<div class="card">
                            <h4>{row['name'][:40]}...</h4>
                            <p><b>Price:</b> ${row['price']:.2f}<br>
                            <b>Match Score:</b> {row['score']:.2f}</p>
                        </div>""", unsafe_allow_html=True)
                        # Explanation logic
                        reason = explain_recommendation(query_row['product_id'], row['product_id'], df)
                        st.caption(f"Reason: {reason}")
            else:
                st.warning("Product outside of Lite Demo range.")

    # --- PAGE: ANALYTICS DASHBOARD ---
    elif page == "Analytics Dashboard":
        st.title("📊 Analytics Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x="price", title="Price Distribution", color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.box(df, x="section", y="sales_volume", title="Sales Volume by Section")
            st.plotly_chart(fig2, use_container_width=True)

    # --- PAGE: MARKET INSIGHTS ---
    elif page == "Market Insights":
        st.title("📈 Market Insights")
        top_brands = df.groupby('brand')['sales_volume'].sum().nlargest(10).reset_index()
        fig = px.bar(top_brands, x='sales_volume', y='brand', orientation='h', title="Top 10 Brands by Total Sales")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
