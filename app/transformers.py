from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handles pandas Series / DataFrame safely
        if isinstance(X, (pd.Series, pd.DataFrame)):
            return X.astype(str)
        return X
