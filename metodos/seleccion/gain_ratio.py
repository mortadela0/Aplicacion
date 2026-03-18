"""
metodos/seleccion/gain_ratio.py
────────────────────────────────────────────────────────────
Gain Ratio nativo — sin Java, sin dependencias externas.

GainRatio = InfoGain / SplitInfo

InfoGain(X, C)  = H(C) - H(C|X)
SplitInfo(X)    = -Σ P(xi) log2 P(xi)

Normalizar por SplitInfo corrige el sesgo de InfoGain
hacia variables con muchos valores distintos.

Score cercano a 1  → feature muy discriminativa y poco ruidosa.
Score cercano a 0  → feature no aporta o tiene demasiadas categorías vacías.
"""

import time
import numpy as np
import pandas as pd
from colorama import Fore
from utils.preprocesamiento import preparar_datos, escalar_minmax
from utils.reporte_weka     import imprimir_reporte

NOMBRE = "Gain Ratio  (InfoGain / SplitInfo)"


def _entropy(y: np.ndarray) -> float:
    """Entropía de Shannon en bits."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _info_gain(x_col: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    InfoGain para una feature continua (discretiza en n_bins)
    o categórica (usa los valores directamente).
    """
    # Detectar si es continua o ya discreta
    unique_vals = np.unique(x_col)
    if len(unique_vals) > n_bins:
        # Continua: discretizar por cuantiles
        bins   = np.percentile(x_col, np.linspace(0, 100, n_bins + 1))
        bins   = np.unique(bins)
        x_disc = np.digitize(x_col, bins[1:-1])
    else:
        x_disc = x_col.astype(int)

    h_y = _entropy(y)

    # H(y | x)
    h_y_given_x = 0.0
    for val in np.unique(x_disc):
        mask   = x_disc == val
        p_val  = mask.sum() / len(y)
        h_y_given_x += p_val * _entropy(y[mask])

    return h_y - h_y_given_x


def _split_info(x_col: np.ndarray, n_bins: int = 10) -> float:
    """Entropía de la propia feature (SplitInfo)."""
    unique_vals = np.unique(x_col)
    if len(unique_vals) > n_bins:
        bins   = np.percentile(x_col, np.linspace(0, 100, n_bins + 1))
        bins   = np.unique(bins)
        x_disc = np.digitize(x_col, bins[1:-1])
    else:
        x_disc = x_col.astype(int)

    _, counts = np.unique(x_disc, return_counts=True)
    probs     = counts / len(x_col)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def ejecutar(df: pd.DataFrame, target_col: str, n_bins: int = 10) -> pd.Series:
    """
    Calcula Gain Ratio para cada feature del dataset.

    Parámetros
    ----------
    n_bins : int
        Bins para discretizar features continuas. Default 10.

    Retorna
    -------
    pd.Series — gain ratios ordenados de mayor a menor.
    """
    t0   = time.time()
    X, y = preparar_datos(df, target_col)

    print(f"\n  {Fore.YELLOW}[INFO] Gain Ratio — {len(df)} instancias, "
          f"{X.shape[1]} features, bins={n_bins}...")

    X_arr = X.values.astype(float)
    y_arr = np.asarray(y)

    gain_ratios = {}
    for j, col in enumerate(X.columns):
        ig  = _info_gain(X_arr[:, j], y_arr, n_bins)
        si  = _split_info(X_arr[:, j], n_bins)
        gain_ratios[col] = ig / si if si > 1e-9 else 0.0

    scores = pd.Series(gain_ratios).sort_values(ascending=False)

    elapsed      = time.time() - t0
    dataset_info = {
        "nombre":   df.attrs.get("nombre", "dataset"),
        "filas":    len(df),
        "columnas": len(df.columns),
    }

    imprimir_reporte(
        scores, "GainRatioAttributeEval  (nativo Python)",
        "Gain Ratio", dataset_info, target_col, elapsed,
    )

    return scores
