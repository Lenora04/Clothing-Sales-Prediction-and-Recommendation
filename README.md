# Clothing Sales Prediction & Recommendation System

Project repository containing data, notebooks, model artifacts and a Streamlit dashboard for predicting clothing sales volume and providing product recommendations.

**Contents**
- **Overview:** purpose and components of the project
- **Quick Start:** dependencies and how to run the Streamlit app locally
- **Data:** source and processed data layout
- **Models & Artifacts:** saved pipelines and similarity matrix handling
- **Application (Streamlit):** pages and usage
- **Performance & Implementation Notes:** float32 conversion, memory-mapped arrays, lazy-loading
- **Troubleshooting:** common errors and fixes
- **Development:** how to retrain and update models

---

**Overview**

This project combines a hybrid recommender (content + collaborative filtering + popularity) with a sales-volume prediction model to support analytics and product recommendations for a clothing retailer dataset. The repository contains exploratory notebooks, feature engineering code, trained model artifacts, and a Streamlit dashboard to interactively explore products and get predictions.

**Quick Start (Windows PowerShell)**

- Install dependencies:

```powershell
pip install -r requirements.txt
```

- Run the Streamlit app:

```powershell
streamlit run app/app.py
```

Open the provided local URL in your browser (usually http://localhost:8501).

**Repository Structure**

- `data/` — raw and processed datasets. See `data/processed/clean_clothing_sales.csv` for the cleaned table used by the app and models.
- `models/` — saved model pipelines and similarity matrices (binary files are ignored by git; see `.gitignore`).
- `notebooks/` — Jupyter notebooks used for EDA, feature engineering, model training and experiments.
- `app/` — Streamlit application code and helpers (`app/app.py`, `app/utils.py`, `app/recommendation_helpers.py`).

**Data**

The app expects a processed dataset at `data/processed/clean_clothing_sales.csv`. Key columns used by the models include:

- `product_id`, `name`, `description`, `price`, `brand`, `sales_volume`, and derived features such as `name_word_count`, `price_per_name_word`, and `price_per_word`.

If you generate new processed data, make sure those columns exist (or the app will compute the derived features on the fly when possible).

**Models & Artifacts**

Trained artifacts are saved in the `models/` directory. Common files include:

- `sales_gb_tuned_pipeline.pkl` (preferred sales-volume predictor)
- `sales_rf_pipeline.pkl`, `sales_rf_baseline_pipeline.pkl` (alternate predictors)
- `hybrid_recommender_pipeline.pkl`, `svd_transformer.pkl` (recommender parts)
- `hybrid_similarity_matrix.npz` (raw similarity matrix produced by notebooks)

For performance and memory efficiency, the app will convert a loaded similarity matrix to `float32` and save a memory-mapped `.npy` copy named `hybrid_similarity_matrix_float32.npy` in `models/` the first time the recommendations page is accessed. A compressed copy `hybrid_similarity_matrix_float32.npz` is also written alongside it. These generated files are ignored by `.gitignore` to avoid committing large binaries.

**Application (Streamlit)**

The Streamlit app exposes several pages:

- Home: project summary and quick links
- Product Search & Recommendations: search by product and get hybrid recommendations (lazy-loads the similarity matrix)
- Analytics Dashboard: aggregate views by brand/section/season
- Market Insights: summary metrics and plots
- Sales Volume Predictor: input product features to get a sales prediction and percentile rank

The recommender uses a hybrid approach and lazy-loads the large similarity matrix only when the recommendations page is visited. This substantially reduces startup time and memory usage.

**Performance & Implementation Notes**

- Large arrays are converted to `float32` to reduce RAM usage and improve I/O speed.
- The app creates and prefers a memory-mapped `.npy` file for the similarity matrix (loaded with `np.load(..., mmap_mode='r')`) to keep memory low.
- Many heavy artifacts are cached in-memory using Streamlit caching; the similarity matrix is lazily created and cached to prevent repeated expensive disk reads.
- The project includes a Streamlit config file at `.streamlit/config.toml` that reduces log verbosity and configures server settings for local development.

**Troubleshooting**

- Missing model files: If the app errors about a missing model (e.g., `sales_gb_tuned_pipeline.pkl`), ensure the correct pipeline is present in `models/` or change the model selection logic in `app/utils.py` to point to an available file.
- Memory/slow startup: Confirm the `models/hybrid_similarity_matrix_float32.npy` exists (created automatically on first load). If not, check write permissions for the `models/` directory.
- Streamlit Arrow errors: If you see Arrow serialization errors when rendering tables, update `pandas` and `pyarrow` to recent versions and make sure displayed columns are homogeneous types; the app already converts dataframe columns to strings in displays to avoid this.

**Development & Retraining**

Notebooks under `notebooks/` were used to preprocess data, engineer features, and train models. Typical workflow:

1. Edit and run `notebooks/02_feature_engineering.ipynb` to prepare features.
2. Use `notebooks/03_multi_model_comparison.ipynb` to train and compare models. Save the chosen pipeline to `models/` using joblib or `pickle`.
3. If the recommender similarities change, produce a new `hybrid_similarity_matrix.npz` and let the app generate the `float32` memory-mapped copy when first run.

**Commands**

Install and run the app (PowerShell):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app.py
```

If you want to regenerate the memory-mapped similarity file manually (for example after producing a new `.npz`), run a short Python snippet from the repo root:

```powershell
python - <<'PY'
import numpy as np
import os
from pathlib import Path

models = Path('models')
src = models / 'hybrid_similarity_matrix.npz'
dst_npy = models / 'hybrid_similarity_matrix_float32.npy'
dst_npz = models / 'hybrid_similarity_matrix_float32.npz'
arr = np.load(src)
key = 'matrix' if 'matrix' in arr else arr.files[0]
mat = arr[key].astype('float32')
np.save(dst_npy, mat)
np.savez_compressed(dst_npz, matrix=mat)
print('Done')
PY
```

**License & Attribution**

This repository is provided as-is for educational and prototyping purposes. Ensure you have rights to any commercial datasets you load into the app.

data source - https://www.kaggle.com/datasets/ayeshasiddiqa123/bussiness-sales-dataset