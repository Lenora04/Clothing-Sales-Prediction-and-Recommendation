"""
Utility functions for loading data and models from Hugging Face.
"""
import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st
from huggingface_hub import hf_hub_download


REPO_ID = "lenoraravindi/clothing-sales-models"

@st.cache_resource
def load_data():
    """
    Downloads and loads the processed dataset from Hugging Face.
    """
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

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.add_categories("missing").fillna("missing")
        elif df[col].dtype == object:
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(0)

    product_id_to_index = pd.Series(df.index.values, index=df['product_id']).to_dict()
    return df, product_id_to_index

@st.cache_resource
def load_models():
    """
    Loads the recommendation pipelines from Hugging Face.
    """
    pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="hybrid_recommender_pipeline.pkl")
    svd_path = hf_hub_download(repo_id=REPO_ID, filename="svd_transformer.pkl")
    
    pipeline = joblib.load(pipeline_path)
    svd = joblib.load(svd_path)
    
    return pipeline, svd, None

@st.cache_resource
def load_similarity_matrix():
    """
    Downloads and loads the float32 similarity matrix.
    Uses memory mapping to save RAM.
    """
    # Use the smaller float32 file
    matrix_path = hf_hub_download(repo_id=REPO_ID, filename="hybrid_similarity_matrix_float32.npz")
    
    # Load the compressed npz
    sim_matrix_data = np.load(matrix_path)
    
    # Extract the matrix
    if 'matrix' in sim_matrix_data:
        sim_matrix = sim_matrix_data['matrix']
    else:
        keys = list(sim_matrix_data.files)
        sim_matrix = sim_matrix_data[keys[0]]

    # Ensure it is float32 for memory efficiency
    if sim_matrix.dtype != np.float32:
        sim_matrix = sim_matrix.astype(np.float32)

    return sim_matrix

@st.cache_resource
def load_sales_predictor():
    """
    Loads the Sales Prediction Gradient Boosting model from Hugging Face.
    """
    model_path = hf_hub_download(repo_id=REPO_ID, filename="sales_gb_tuned_pipeline.pkl")
    # If you have a separate feature engineering pkl, download it here too
    # feature_path = hf_hub_download(repo_id=REPO_ID, filename="feature_engineering_pipeline.pkl")
    
    model = joblib.load(model_path)
    # feature_pipeline = joblib.load(feature_path) # Uncomment if needed
    
    return model, None # Return feature_pipeline as second arg if loaded
