"""
Utility functions for loading data and models.
"""
import pandas as pd
import numpy as np
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'clean_clothing_sales.csv')
PIPELINE_PATH = os.path.join(BASE_DIR, '..', 'models', 'hybrid_recommender_pipeline.pkl')
SVD_PATH = os.path.join(BASE_DIR, '..', 'models', 'svd_transformer.pkl')
SIM_MATRIX_PATH = os.path.join(BASE_DIR, '..', 'models', 'hybrid_similarity_matrix.npz')
SALES_PREDICTOR_PATH = os.path.join(BASE_DIR, '..', 'models', 'sales_gb_tuned_pipeline.pkl')
FEATURE_PIPELINE_PATH = os.path.join(BASE_DIR, '..', 'models', 'feature_engineering_pipeline.pkl')
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))


def load_data():
    """
    Load the processed dataset with optimized dtypes and create product ID to index mapping.
    
    Returns:
        tuple: (dataframe, product_id_to_index_dict)
    """
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            'price': 'float32',
            'sales_volume': 'int32',
            'desc_len': 'int16',
            'name_len': 'int16',
            'desc_word_count': 'int16',
            'price_per_word': 'float32',
            'promotion': 'int8',
            'seasonal': 'int8',
            'product_position': 'category',
            'product_category': 'category',
            'section': 'category',
            'season': 'category',
            'material': 'category',
            'origin': 'category',
            'brand': 'category'
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

    # Create stable product ID to index mapping
    product_id_to_index = pd.Series(df.index.values, index=df['product_id']).to_dict()
    
    return df, product_id_to_index


def load_models():
    """
    Load pre-trained pipeline and SVD transformer.
    Similarity matrix is loaded lazily on first use.
    
    Returns:
        tuple: (pipeline, svd, None)  # sim_matrix loaded later
    """
    # Load pipeline
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    
    # Load SVD
    svd = None
    if os.path.exists(SVD_PATH):
        svd = joblib.load(SVD_PATH)
    
    # Don't load similarity matrix here - lazy load it only when needed
    return pipeline, svd, None


def load_similarity_matrix():
    """
    Lazy-load the similarity matrix only when needed for recommendations.
    Call this function only in recommendation_helpers.py when sim_matrix is needed.
    
    Returns:
        np.ndarray: similarity_matrix
    """
    # Prefer a memory-mapped float32 .npy file for fast, low-memory loads
    npy_path = os.path.join(MODELS_DIR, 'hybrid_similarity_matrix_float32.npy')
    compressed_path = os.path.join(MODELS_DIR, 'hybrid_similarity_matrix_float32.npz')

    if os.path.exists(npy_path):
        # Load as memory-mapped read-only array
        return np.load(npy_path, mmap_mode='r')

    # If .npy not present, try existing NPZ and create float32 .npy for future runs
    if not os.path.exists(SIM_MATRIX_PATH):
        raise FileNotFoundError(f"Similarity matrix not found at {SIM_MATRIX_PATH}")

    sim_matrix_data = np.load(SIM_MATRIX_PATH)
    # prefer key 'matrix' or 'similarity_matrix'
    if 'matrix' in sim_matrix_data:
        sim_matrix = sim_matrix_data['matrix']
    elif 'similarity_matrix' in sim_matrix_data:
        sim_matrix = sim_matrix_data['similarity_matrix']
    else:
        # fallback: take the first array in the npz
        keys = list(sim_matrix_data.files)
        if not keys:
            raise ValueError(f"No arrays found inside {SIM_MATRIX_PATH}")
        sim_matrix = sim_matrix_data[keys[0]]

    # Convert to float32 if necessary
    if sim_matrix.dtype != np.float32:
        sim_matrix = sim_matrix.astype(np.float32)

    # Save a memory-mapped friendly .npy and a compressed .npz copy for storage
    try:
        # Save .npy (overwrites if exists)
        np.save(npy_path, sim_matrix)
        # Save compressed copy for smaller disk footprint
        np.savez_compressed(compressed_path, matrix=sim_matrix)
    except Exception:
        # If saving fails, continue and return the array in-memory
        return sim_matrix

    # Return a memory-mapped .npy for efficient future loads
    return np.load(npy_path, mmap_mode='r')


def load_sales_predictor():
    """
    Load pre-trained sales volume prediction model.
    
    Returns:
        tuple: (model, feature_pipeline)
    """
    # Discover available sales predictor files in the models directory.
    # Prefer gradient-boosting/XGBoost/GBR-named models when available.
    candidate_files = []
    if os.path.isdir(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.lower().endswith('.pkl') and fname.lower().startswith('sales'):
                candidate_files.append(os.path.join(MODELS_DIR, fname))

    # If explicit SALES_PREDICTOR_PATH exists, include it as a candidate too
    if os.path.exists(SALES_PREDICTOR_PATH):
        candidate_files.append(SALES_PREDICTOR_PATH)

    if not candidate_files:
        raise FileNotFoundError(f"No sales predictor files found in {MODELS_DIR} or {SALES_PREDICTOR_PATH}")

    # Prefer files that indicate gradient boosting or gb/xgb in the filename
    preferred_keywords = ['gradient', 'gboost', 'gb', 'xgb', 'gbr', 'boosting']
    selected = None
    for f in candidate_files:
        low = os.path.basename(f).lower()
        if any(k in low for k in preferred_keywords):
            selected = f
            break

    # If none matched preferred keywords, pick the most recently modified candidate
    if selected is None:
        candidate_files = sorted(candidate_files, key=lambda p: os.path.getmtime(p), reverse=True)
        selected = candidate_files[0]

    model = joblib.load(selected)

    # Load feature engineering pipeline if available
    feature_pipeline = None
    if os.path.exists(FEATURE_PIPELINE_PATH):
        feature_pipeline = joblib.load(FEATURE_PIPELINE_PATH)

    return model, feature_pipeline
