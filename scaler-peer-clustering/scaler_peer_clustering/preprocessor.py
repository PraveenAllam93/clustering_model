from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Custom Transformer for log transformation
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = np.log(X[col].clip(lower=1))  # Ensure positive numbers
        return X

# Pipeline construction
def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['ctc', 'orgyear']),
            ('cat', categorical_transformer, ['job_position', 'company_hash'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('log_transform', LogTransformer(cols=['ctc']))
    ])
    
    return pipeline


