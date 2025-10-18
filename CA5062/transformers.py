from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CubeRootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names if X is a pandas DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        # Apply cube root transformation
        X_transformed = np.cbrt(X)

        # If input was pandas DataFrame, return DataFrame
        if hasattr(X, 'columns'):
            import pandas as pd
            return pd.DataFrame(X_transformed, columns=self.feature_names_in_, index=X.index)
        return X_transformed

    def set_output(self, transform=None):
        # Add set_output method for pandas compatibility
        return self

