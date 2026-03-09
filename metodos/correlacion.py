"""
metodos/correlacion.py
─────────────────────────────────────────────
MÉTODO 1 — Correlación de Pearson (|r|)

Soporta target numérico Y string/categórico.
Si el target es string se codifica automáticamente
con LabelEncoder antes de calcular la correlación.
"""

import time
import pandas as pd
import numpy as np
from utils.preprocesamiento import preparar_datos
from utils.reporte_weka     import imprimir_reporte


def ejecutar(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Calcula Pearson |r| de cada feature vs target.
    El target puede ser numérico o string — se codifica internamente.

    Retorna:
        pd.Series — correlaciones absolutas ordenadas desc.
    """
    t0 = time.time()

    X, y = preparar_datos(df, target_col)

    # Convertir y a Series para poder usar corrwith
    y_series      = pd.Series(y, index=X.index, name=target_col)
    correlaciones = X.corrwith(y_series).abs().sort_values(ascending=False)

    # Eliminar NaN (columnas constantes después del encode)
    correlaciones = correlaciones.dropna()

    elapsed      = time.time() - t0
    dataset_info = {
        "nombre":   df.attrs.get("nombre", "dataset.csv"),
        "filas":    len(df),
        "columnas": len(df.columns),
    }

    imprimir_reporte(
        scores       = correlaciones,
        metodo       = "CorrelationAttributeEval",
        nombre_score = "Pearson |r|",
        dataset_info = dataset_info,
        target_col   = target_col,
        elapsed      = elapsed,
        decimales    = 10,
        umbral_pct   = 50.0,
    )

    return correlaciones
