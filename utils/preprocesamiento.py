import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

_target_encoder   = LabelEncoder()
_target_classes   = []
_target_es_string = False

def _es_categorico_dtype(serie):
    dtype_str = str(serie.dtype).lower()
    if dtype_str in ("object","category") or dtype_str.startswith("string"):
        return True
    try:
        pd.to_numeric(serie.dropna().iloc[:5]); return False
    except (ValueError, TypeError):
        return True

def preparar_datos(df, target_col):
    global _target_encoder, _target_classes, _target_es_string
    le = LabelEncoder()
    X  = df.drop(columns=[target_col]).copy()
    y  = df[target_col].copy()

    if _es_categorico_dtype(y):
        _target_es_string = True
        y = _target_encoder.fit_transform(y.astype(str).fillna("NA"))
        _target_classes   = list(_target_encoder.classes_)
    elif pd.api.types.is_integer_dtype(y.dtype):
        _target_es_string = False
        y = y.fillna(0).values.astype(int)
    else:
        _target_es_string = False
        try:
            y_num = pd.to_numeric(y, errors="coerce")
            y     = y_num.fillna(y_num.median()).values
        except TypeError:
            _target_es_string = True
            y = _target_encoder.fit_transform(y.astype(str).fillna("NA"))
            _target_classes   = list(_target_encoder.classes_)

    for col in X.columns:
        if _es_categorico_dtype(X[col]):
            X[col] = le.fit_transform(X[col].astype(str).fillna("NA"))
    for col in X.columns:
        if X[col].isnull().any():
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(X[col].median())
            except TypeError:
                X[col] = X[col].fillna(0)
    return X.fillna(0), y

def target_es_categorico(df, target_col):
    y = df[target_col]
    return _es_categorico_dtype(y) or y.nunique() <= 20

def escalar_minmax(X):
    return MinMaxScaler().fit_transform(X)

def normalizar_serie(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-9)
