import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st
from huggingface_hub import hf_hub_download

REPO_ID = "lenoraravindi/clothing-sales-models"

# --- CACHING WRAPPERS ---

@st.cache_data(show_spinner="Loading dataset...")
def get_data_cached():
    return load_data()

@st.cache_resource(show_spinner="Loading recommendation models...")
def get_models_cached():
    return load_models()

@st.cache_resource(show_spinner="Loading sales predictor...")
def get_sales_predictor_cached():
    return load_sales_predictor()

@st.cache_resource(show_spinner="Loading Similarity Matrix...")
def get_sim_matrix_cached():
    return load_similarity_matrix()

# --- ORIGINAL LOADING FUNCTIONS ---

def load_data():
    file_path = hf_hub_download(repo_id=REPO_ID, filename="clean_clothing_sales.csv")
    df = pd.read_csv(
        file_path,
        dtype={
            'price': 'float32', 'sales_volume': 'int32', 'desc_len': 'int16',
            'name_len': 'int16', 'desc_word_count': 'int16', 'price_per_word': 'float32',
            'promotion': 'int8', 'seasonal': 'int8', 'product_position': 'category',
            'product_category': 'category', 'section': 'category', 'season': 'category',
            'material': 'category', 'origin': 'category', 'brand': 'category'
        }
    )
    df['product_id'] = df['product_id'].astype(str)

    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.add_categories("missing").fillna("missing")
        elif df[col].dtype == object:
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(0)

    product_id_to_index = pd.Series(df.index.values, index=df['product_id']).to_dict()
    return df, product_id_to_index

def load_models():
    pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="hybrid_recommender_pipeline.pkl")
    svd_path = hf_hub_download(repo_id=REPO_ID, filename="svd_transformer.pkl")
    pipeline = joblib.load(pipeline_path)
    svd = joblib.load(svd_path)
    return pipeline, svd, None # Added 3rd return to match app.py expectation

def load_similarity_matrix():
    matrix_path = hf_hub_download(repo_id=REPO_ID, filename="hybrid_similarity_matrix_float32.npz")
    sim_matrix_data = np.load(matrix_path)
    full_matrix = sim_matrix_data['matrix'] if 'matrix' in sim_matrix_data else sim_matrix_data[list(sim_matrix_data.files)[0]]
    return full_matrix[:500, :500].astype(np.float32)

def load_sales_predictor():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="sales_gb_tuned_pipeline.pkl")
    model = joblib.load(model_path)
    return model, None
