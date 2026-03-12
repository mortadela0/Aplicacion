"""
utils/preprocesamiento.py
─────────────────────────────────────────────
Funciones de limpieza y preparación de datos.
Soporta target numérico Y string/categórico.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

_target_encoder   = LabelEncoder()
_target_classes   = []
_target_es_string = False


def _es_categorico_dtype(serie: pd.Series) -> bool:
    """
    Detecta si una serie debe tratarse como categórica,
    cubriendo todos los casos: object, category, StringDtype, mixto.
    """
    dtype_str = str(serie.dtype).lower()
    if dtype_str in ("object", "category"):
        return True
    if dtype_str.startswith("string"):      # pandas StringDtype
        return True
    # Intentar convertir a float — si falla es categórico
    try:
        pd.to_numeric(serie.dropna().iloc[:5])
        return False
    except (ValueError, TypeError):
        return True


def preparar_datos(df: pd.DataFrame, target_col: str):
    """
    Prepara X e y para todos los métodos de selección.
    Soporta target string, entero o float.
    """
    global _target_encoder, _target_classes, _target_es_string

    le = LabelEncoder()
    X  = df.drop(columns=[target_col]).copy()
    y  = df[target_col].copy()

    # ── Codificar target ──────────────────────────────
    if _es_categorico_dtype(y):
        # String / categórico → LabelEncoder
        _target_es_string = True
        y = _target_encoder.fit_transform(y.astype(str).fillna("NA"))
        _target_classes   = list(_target_encoder.classes_)

    elif pd.api.types.is_integer_dtype(y.dtype):
        # Entero → usar directamente
        _target_es_string = False
        y = y.fillna(0).values.astype(int)

    else:
        # Float → intentar mediana, fallback a categórico
        _target_es_string = False
        try:
            y_num = pd.to_numeric(y, errors="coerce")
            y = y_num.fillna(y_num.median()).values
        except TypeError:
            _target_es_string = True
            y = _target_encoder.fit_transform(y.astype(str).fillna("NA"))
            _target_classes   = list(_target_encoder.classes_)

    # ── Codificar features categóricas ───────────────
    for col in X.columns:
        if _es_categorico_dtype(X[col]):
            X[col] = le.fit_transform(X[col].astype(str).fillna("NA"))

    # ── Imputar nulos numéricos ───────────────────────
    for col in X.columns:
        if X[col].isnull().any():
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(X[col].median())
            except TypeError:
                X[col] = X[col].fillna(0)
    X = X.fillna(0)

    return X, y


def target_es_categorico(df: pd.DataFrame, target_col: str) -> bool:
    """True si el target debe tratarse como clasificación."""
    y = df[target_col]
    if _es_categorico_dtype(y):
        return True
    return y.nunique() <= 20


def escalar_minmax(X: pd.DataFrame) -> np.ndarray:
    """Escala features a [0,1]."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def normalizar_serie(s: pd.Series) -> pd.Series:
    """Normaliza pd.Series a [0,1]."""
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-9)
