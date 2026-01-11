import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, load_models, load_sales_predictor, load_similarity_matrix
from recommendation_helpers import recommend, explain_recommendation, visualize_products, predict_sales_volume, get_similar_products_by_sales, sales_improvement_hints, to_df
import plotly.express as px
import plotly.graph_objects as go
import sys
import os


# ==============================================================================
# CACHING & OPTIMIZATION (Updated for Hugging Face)
# ==============================================================================

@st.cache_data(show_spinner="Loading dataset...")
def get_data_cached():
    return load_data()

@st.cache_resource(show_spinner="Loading recommendation models...")
def get_models_cached():
    # Note: sim_matrix is now handled separately to save RAM
    pipeline, svd, _ = load_models()
    return pipeline, svd

@st.cache_resource(show_spinner="Loading sales predictor...")
def get_sales_predictor_cached():
    return load_sales_predictor()

# Initialize core models on startup
df, product_id_to_index = get_data_cached()
pipeline, svd = get_models_cached()
sales_model, sales_pipeline = get_sales_predictor_cached()

# Similarity Matrix is handled separately due to 6GB size
@st.cache_resource(show_spinner="Downloading 6GB Similarity Matrix... please wait.")
def get_sim_matrix_cached():
    return load_similarity_matrix()

# In the global scope, keep this as None
sim_matrix = None

# ==============================================================================
# Page Configuration
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
    .main {
        padding: 20px;
    }
    .card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 10px;
    }
            /* --- Button Styling for Predict Button --- */
        .stButton > button {
            /* Set the base color to green */
            background-color: #5ea879; 
            color: white; 
            border-color: #10b981;
            transition: background-color 0.3s ease;
        }

        /* Change color on hover using the standard selector */
        .stButton > button:hover {
            background-color: #07db54; 
            color: white;
            border-color: #16a34a;
            box-shadow: 0 4px 14px rgba(16,185,129,0.2);
        }
    
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Sidebar Navigation
# ==============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home","Sales Volume Predictor", "Product Search & Recommendations", "Analytics Dashboard", "Market Insights" ]
)

# ==============================================================================
# PAGE: HOME
# ==============================================================================
if page == "Home":
    st.title("👕 Clothing Sales Recommendation System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦 Products</h3>
            <p style="font-size: 32px; margin: 0;">{len(df):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 1. Get unique values (excluding 'missing' if present)
        unique_brands = df['brand'].unique()
        clean_brands = [str(b) for b in unique_brands if str(b).lower() != 'missing']
        
        # 2. Join them into a clean string
        brands_str = ", ".join(clean_brands)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏷️ Brands</h3>
            <p style="font-size: 32px; margin: 0;">{brands_str}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Avg Price</h3>
            <p style="font-size: 32px; margin: 0;">${df['price'].mean():.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("About This System")
    st.markdown("""
    This hybrid recommendation engine combines:
    - **Content-Based Filtering**: Product features (material, season, section, price)
    - **Text Similarity**: Product names and descriptions (TF-IDF vectorization)
    - **Feature Engineering**: One-hot encoding of categorical attributes
    - **Dimensionality Reduction**: SVD for efficient similarity computation
    
    **Features:**
    - 🔍 Find similar products
    - 📊 View product analytics
    - 💡 Understand recommendations with explainability
    - 📈 Explore market trends
    """)

# ==============================================================================
# PAGE: PRODUCT SEARCH & RECOMMENDATIONS
# ==============================================================================
elif page == "Product Search & Recommendations":
    st.title("Product Search & Recommendations")
    st.markdown("---")
    
    # Lazy-load similarity matrix on first access to recommendations page
    if sim_matrix is None:
        with st.status("Fetching heavy model data from Hugging Face...", expanded=True) as status:
            sim_matrix = get_sim_matrix_cached()
            status.update(label="Model Loaded!", state="complete", expanded=False)
    
    # Cache product names in session state for faster rendering
    if 'product_names_cache' not in st.session_state:
        st.session_state.product_names_cache = df['name'].unique().tolist()
    
    # Product selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_product_name = st.selectbox(
            "Select a Product",
            st.session_state.product_names_cache,
            key="product_select"
        )
    
    with col2:
        top_n = st.number_input("Top N Recommendations", min_value=1, max_value=20, value=5)
    
    if selected_product_name:
        # Get product_id from name
        matching_rows = df[df['name'] == selected_product_name]
        if len(matching_rows) > 0:
            selected_product_id = matching_rows.iloc[0]['product_id']
            
            # Display query product
            st.markdown("### Selected Product")
            query_product = df[df['product_id'] == selected_product_id].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"${query_product['price']:.2f}")
            with col2:
                st.metric("Section", query_product['section'])
            with col3:
                st.metric("Season", query_product['season'])
            with col4:
                st.metric("Material", query_product['material'])
            
            # Display product details
            with st.expander("📋 View Full Details"):
                # Convert to dict for safe display
                details_dict = query_product.to_dict()
                for key, value in details_dict.items():
                    st.write(f"**{key}**: {value}")
            
            st.markdown("---")
            
            # Get recommendations
            st.markdown("### 🎯 Recommended Products")
            try:
                recommendations = recommend(
                    selected_product_id, 
                    product_id_to_index, 
                    sim_matrix, 
                    df, 
                    top_n=top_n,
                    include_score=True
                )
                
                if recommendations is not None and len(recommendations) > 0:
                    # Display recommendations in a grid
                    cols = st.columns(min(3, len(recommendations)))
                    for idx, (_, rec) in enumerate(recommendations.iterrows()):
                        with cols[idx % 3]:
                            st.markdown(f"""
                            <div class="card">
                                <h4>{rec['name'][:30]}...</h4>
                                <p><b>Price:</b> ${rec['price']:.2f}</p>
                                <p><b>Section:</b> {rec['section']}</p>
                                <p><b>Material:</b> {rec['material']}</p>
                                <p><b>Similarity Score:</b> {rec.get('score', 0):.3f}</p>
                                <a href="{rec['url']}" target="_blank">View Product →</a>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Explainability
                    st.markdown("### 💡 Why These Recommendations?")
                    with st.expander("View Explainability"):
                        try:
                            explanation = explain_recommendation(
                                        selected_product_id,
                                        recommendations.iloc[0]['product_id'],
                                        df
                                    )
                            if explanation:
                                st.info(f"**Top matching features:** {explanation}")
                        except Exception as e:
                            st.warning(f"Could not generate explanation: {e}")
                    
                    # Show all recommendations as table
                    st.markdown("### 📋 Detailed Recommendations Table")
                    display_cols = [col for col in ['name', 'section', 'season', 'material', 'price', 'score'] 
                                   if col in recommendations.columns]
                    # Convert to Arrow-compatible types
                    rec_display = recommendations[display_cols].astype(str)
                    st.dataframe(rec_display, use_container_width=True)
                else:
                    st.warning("No recommendations found.")
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# ==============================================================================
# PAGE: ANALYTICS DASHBOARD
# ==============================================================================
elif page == "Analytics Dashboard":
    st.title("Analytics Dashboard")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Unique Brands", df['brand'].nunique())
    with col3:
        st.metric("Average Sales Volume", f"{df['sales_volume'].mean():.0f}")
    with col4:
        st.metric("Price Range", f"${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Products by Section")
        section_counts = df['section'].value_counts()
        fig = px.bar(
            x=section_counts.index,
            y=section_counts.values,
            labels={'x': 'Section', 'y': 'Count'},
            color=section_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Products by Season")
        season_counts = df['season'].value_counts()
        fig = px.pie(
            labels=season_counts.index,
            values=season_counts.values,
            title="Product Distribution by Season"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Distribution")
        fig = px.histogram(
            df,
            x='price',
            nbins=50,
            title="Price Distribution",
            labels={'price': 'Price ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Sales Volume Distribution")
        fig = px.histogram(
            df,
            x='sales_volume',
            nbins=50,
            title="Sales Volume Distribution",
            labels={'sales_volume': 'Sales Volume'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Sales Volume scatter
    st.subheader("💹 Price vs Sales Volume")
    fig = px.scatter(
        df.sample(min(1000, len(df))),  # Sample for performance
        x='price',
        y='sales_volume',
        color='section',
        size='sales_volume',
        hover_name='name',
        title="Product Price vs Sales Volume"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: MARKET INSIGHTS
# ==============================================================================
elif page == "Market Insights":
    st.title("Market Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Brands by Sales")
        top_brands = df.groupby('brand')['sales_volume'].sum().nlargest(10)
        fig = px.bar(
            x=top_brands.values,
            y=top_brands.index,
            orientation='h',
            title="Top 10 Brands by Total Sales",
            labels={'x': 'Total Sales Volume', 'y': 'Brand'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Best Selling Products")
        top_products = df.nlargest(10, 'sales_volume')[['name', 'sales_volume', 'price']]
        fig = px.bar(
            top_products,
            x='sales_volume',
            y='name',
            orientation='h',
            title="Top 10 Best Selling Products",
            labels={'sales_volume': 'Sales Volume', 'name': 'Product'},
            color='price',
            color_continuous_scale='Turbo'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Material popularity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧵 Popular Materials")
        material_sales = df.groupby('material')['sales_volume'].sum().nlargest(15)
        fig = px.bar(
            x=material_sales.index,
            y=material_sales.values,
            title="Top Materials by Sales",
            labels={'x': 'Material', 'y': 'Total Sales Volume'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌏 Origin Analysis")
        origin_sales = df.groupby('origin')['sales_volume'].sum().nlargest(15)
        fig = px.bar(
            x=origin_sales.index,
            y=origin_sales.values,
            title="Top Origins by Sales",
            labels={'x': 'Origin', 'y': 'Total Sales Volume'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal trends
    st.subheader("🌡️ Sales Trends by Season and Section")
    season_section = df.groupby(['season', 'section'])['sales_volume'].sum().reset_index()
    fig = px.bar(
        season_section,
        x='season',
        y='sales_volume',
        color='section',
        barmode='group',
        title="Sales by Season and Section",
        labels={'sales_volume': 'Total Sales Volume'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE: SALES VOLUME PREDICTOR
# ==============================================================================
elif page == "Sales Volume Predictor":
    st.title("Sales Volume Predictor")
    st.markdown("---")
    
    if sales_model is None:
        st.error("❌ Sales volume predictor model not available. Please train the model first.")
        st.info("Run notebook: `notebooks/03_multi_model_comparison.ipynb`")
    else:
        st.markdown("""
        Predict the expected sales volume for a product based on its features.
        Enter product characteristics and get an estimate of how well it will sell.
        """)
        
        # Create input form
        st.subheader("📝 Product Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price = st.number_input(
                "💰 Price ($)",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                step=1.0
            )
            
            promotion = st.selectbox(
                "📢 Promotion",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            
            seasonal = st.selectbox(
                "🌡️ Seasonal",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            material = st.selectbox(
                "🧵 Material", 
                options=df['material'].unique().tolist(), 
                key='pred_material'
            )

            origin = st.selectbox(
                "🌏 Origin", 
                options=df['origin'].unique().tolist(), 
                key='pred_origin' 
            )
            trained_terms = [t for t in df['terms'].unique() if str(t).lower() != 'missing']

            # 2. Add the 'Other' option
            selectbox_options = trained_terms + ['Other']

            terms = st.selectbox(
                "🔍 Product Term", 
                options=selectbox_options,
                key='pred_terms',
                help="The specific product category/term (e.g., jackets, shoes) used for sales modeling."
            )
        
        
        with col2:
            section = st.selectbox(
                "🏗️ Section", 
                options=df['section'].unique().tolist(), 
                key='pred_section'
            )

            season = st.selectbox(
                "🌍 Season", 
                options=df['season'].unique().tolist(), 
                key='pred_season'
            )
            
            product_position = st.selectbox(
                "📍 Product Position",
                options=['Aisle', 'End-cap', 'Front of Store'],
                key='pred_position'
            )

            # Add placeholders for the text columns (needed by TF-IDF)
            product_name = st.text_input(
                "🏷️ Product Name", 
                key='pred_name'
            )
            product_description = st.text_area(
                "📄 Product Description",
                key='pred_description'
            )
        
        # Make prediction
        if st.button("🔮 Predict Sales Volume", use_container_width=True):
            # Prepare input data
            
            input_df = pd.DataFrame([{
                'name': product_name,
                'description': product_description,
                'price': price,
                'promotion': promotion,
                'seasonal': seasonal,
                'section': section,
                'season': season,
                'material': material,
                'origin': origin,              
                'product_position': product_position,
                'terms': terms
            }])

            input_df['desc_len'] = input_df['description'].str.len()
            input_df['name_len'] = input_df['name'].str.len()

            input_df['desc_word_count'] = input_df['description'].str.split().str.len()
            input_df['name_word_count'] = input_df['name'].str.split().str.len()

            input_df['price_per_word'] = input_df['price'] / (input_df['desc_word_count'] + 1)
            input_df['price_per_name_word'] = input_df['price'] / (input_df['name_word_count'] + 1)       

            try:
                # Get prediction (pass feature pipeline if available)
                result = predict_sales_volume(input_df, sales_model, df)


                if result:
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")

                    # Emphasize predicted sales: large left card, compact stats on right
                    left, right = st.columns([3, 1])

                    with left:
                        pred = int(round(result['prediction']))
                        mean = int(round(result['dataset_mean']))
                        delta_val = int(round(result['prediction'] - result['dataset_mean']))
                        pct = result.get('percentile', 0.0)

                        # Large visual for predicted sales
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;justify-content:space-between;padding:18px;border-radius:10px;background:linear-gradient(90deg,#edf2ff,#e6fffa);">
                            <div style="flex:1">
                                <div style="font-size:28px;color:#1f2937;margin-bottom:6px;">🎯 Predicted Sales</div>
                                <div style="font-size:56px;font-weight:700;color:#0b5b8c;">{pred:,}</div>
                                <div style="font-size:14px;color:#374151;margin-top:6px;">vs catalog mean: <b>{mean:,}</b> &nbsp; (<span style="color:{'#059669' if delta_val>0 else '#b91c1c'}">{('+' if delta_val>=0 else '')}{delta_val:,}</span>)</div>
                            </div>
                            <div style="width:160px;text-align:center;padding-left:18px;border-left:1px solid rgba(0,0,0,0.06)">
                                <div style="font-size:12px;color:#6b7280">Percentile</div>
                                <div style="font-size:28px;font-weight:700;color:#111827;margin-top:6px">{pct:.1f}%</div>
                                <div style="font-size:11px;color:#6b7280;margin-top:6px">% of products below this estimate</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with right:
                        # Compact supporting metrics
                        st.markdown("<div style='padding:6px'>", unsafe_allow_html=True)
                        st.metric("📈 Dataset Average", f"{result['dataset_mean']:.0f}")
                        st.metric("📍 Dataset Median", f"{result['dataset_median']:.0f}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("---")
                    sigma = result['dataset_std']
                    
                    # Interpretation
                    st.subheader("💡 Interpretation")
                    
                    percentile = result['percentile']
                    if percentile >= 75:
                        interpretation = "🟢 **High Sales Potential** - This product is expected to perform better than most items in the catalog."
                    elif percentile >= 50:
                        interpretation = "🟡 **Average Sales Potential** - This product is expected to perform around the median."
                    else:
                        interpretation = "🔴 **Lower Sales Potential** - Consider optimizing price, description, or features."
                    
                    st.info(interpretation)

                    st.markdown("---")
                    st.subheader("🚀 What could increase sales?")

                    input_dict = input_df.iloc[0].to_dict()


                    base_pred, improvements = sales_improvement_hints(
                        input_dict,
                        sales_model,
                        df
                    )

                    if improvements:
                        for label, delta in improvements[:4]:
                            st.success(f"{label} → **+{int(delta)} sales units**")
                    else:
                        st.info("This product is already near optimal based on the model.")

                    
                    # Show similar products by sales
                    st.markdown("---")
                    st.subheader("🔗 Similar Products by Sales Volume")
                    
                    similar_prods = get_similar_products_by_sales(
                        result['prediction'],
                        df,
                        tolerance=0.15,
                        top_n=5
                    )
                    
                    if len(similar_prods) > 0:
                        st.dataframe(similar_prods, use_container_width=True)
                        st.caption("Products with similar expected sales volume (±15%)")
                    else:
                        st.info("No similar products found in the dataset.")
                    
                    st.markdown("---")
                    st.subheader("🧾 Product Summary")

                    summary_items = {
                        "Product Name": product_name,
                        "Price": f"${price:.2f}",
                        "Promotion": "Yes" if promotion else "No",
                        "Seasonal": "Yes" if seasonal else "No",
                        "Material": material,
                        "Section": section,
                        "Season": season,
                        "Product Position": product_position
                    }

                    with st.expander("Details"):
                        for k, v in summary_items.items():
                            st.write(f"**{k}**: {v}")

                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info("Ensure the model is trained and all features are provided.")

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #888;">
    <p><b>Clothing Sales Prediction & Recommendation System</b></p>
    <p>Built with Streamlit | Machine Learning </p>
</div>
""", unsafe_allow_html=True)
