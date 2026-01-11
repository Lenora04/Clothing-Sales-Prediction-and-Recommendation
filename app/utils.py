import pandas as pd
import numpy as np
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download

REPO_ID = "lenoraravindi/clothing-sales-models"

@st.cache_data(show_spinner="Loading dataset...")
def get_data_cached():
    file_path = hf_hub_download(repo_id=REPO_ID, filename="clean_clothing_sales.csv")
    df = pd.read_csv(file_path)
    # Ensure IDs are strings for matching
    df['product_id'] = df['product_id'].astype(str)
    product_id_to_index = pd.Series(df.index.values, index=df['product_id']).to_dict()
    return df, product_id_to_index

@st.cache_resource(show_spinner="Loading recommendation models...")
def get_models_cached():
    p_path = hf_hub_download(repo_id=REPO_ID, filename="hybrid_recommender_pipeline.pkl")
    s_path = hf_hub_download(repo_id=REPO_ID, filename="svd_transformer.pkl")
    return joblib.load(p_path), joblib.load(s_path), None

@st.cache_resource(show_spinner="Loading sales predictor...")
def get_sales_predictor_cached():
    m_path = hf_hub_download(repo_id=REPO_ID, filename="sales_gb_tuned_pipeline.pkl")
    return joblib.load(m_path), None

@st.cache_resource(show_spinner="Loading Similarity Matrix...")
def get_sim_matrix_cached():
    path = hf_hub_download(repo_id=REPO_ID, filename="hybrid_similarity_matrix_float32.npz")
    data = np.load(path)
    mat = data['matrix'] if 'matrix' in data else data[list(data.files)[0]]
    # Slice for RAM limits on Free Tier
    return mat[:500, :500].astype(np.float32)
