"""
metodos/seleccion/relief.py
────────────────────────────────────────────────────────────
ReliefF nativo — sin Java, sin dependencias externas.

Algoritmo: para cada instancia busca k vecinos de la misma clase
(hits) y k vecinos de cada clase diferente (misses) y actualiza
el peso de cada feature según si la diferencia de valor ayuda
o perjudica a separar las clases.

Score alto  → feature separa bien las clases (penaliza hits, recompensa misses).
Score bajo  → feature no discrimina o introduce ruido.

Referencia: Kononenko, 1994 — Estimating Attributes: Analysis and Extensions of RELIEF
"""

import time
import numpy as np
import pandas as pd
from colorama import Fore
from utils.preprocesamiento import preparar_datos, target_es_categorico
from utils.reporte_weka     import imprimir_reporte

NOMBRE = "ReliefF  (instancia-based)"


def _relief_scores(X: np.ndarray, y: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """
    Calcula ReliefF para clasificación multiclase.
    X debe estar normalizado [0,1] por columna.
    """
    n, m = X.shape
    classes      = np.unique(y)
    class_probs  = {c: np.mean(y == c) for c in classes}
    scores       = np.zeros(m)

    for i in range(n):
        xi, yi   = X[i], y[i]
        dists    = np.linalg.norm(X - xi, axis=1)
        dists[i] = np.inf          # excluir la propia instancia

        # k hits (misma clase)
        hit_mask = y == yi
        hit_idx  = np.where(hit_mask)[0]
        hit_idx  = hit_idx[np.argsort(dists[hit_idx])][:n_neighbors]

        for c in classes:
            if c == yi:
                continue
            miss_mask = y == c
            miss_idx  = np.where(miss_mask)[0]
            miss_idx  = miss_idx[np.argsort(dists[miss_idx])][:n_neighbors]

            if len(miss_idx) == 0:
                continue

            p_c = class_probs[c] / max(1 - class_probs[yi], 1e-9)

            for j in range(m):
                d_hit  = float(np.mean(np.abs(xi[j] - X[hit_idx,  j]))) if len(hit_idx) else 0.0
                d_miss = float(np.mean(np.abs(xi[j] - X[miss_idx, j])))
                scores[j] -= d_hit  / (n * n_neighbors)
                scores[j] += p_c * d_miss / (n * n_neighbors)

    return scores


def ejecutar(df: pd.DataFrame, target_col: str, n_neighbors: int = 10) -> pd.Series:
    """
    Ejecuta ReliefF sobre el dataset.

    Parámetros
    ----------
    n_neighbors : int
        Número de vecinos por clase. Default 10.
        Con datasets pequeños (<50 instancias) usar 5.

    Retorna
    -------
    pd.Series — scores ordenados de mayor a menor.
    """
    t0    = time.time()
    X, y  = preparar_datos(df, target_col)

    print(f"\n  {Fore.YELLOW}[INFO] ReliefF — {len(df)} instancias, "
          f"{X.shape[1]} features, k={n_neighbors}...")

    # Normalizar X a [0,1] para que las distancias sean comparables
    Xn = X.values.astype(float)
    rng = Xn.max(axis=0) - Xn.min(axis=0)
    rng[rng == 0] = 1.0
    Xn  = (Xn - Xn.min(axis=0)) / rng

    raw_scores = _relief_scores(Xn, np.asarray(y), n_neighbors)

    # Desplazar a [0, inf) para compatibilidad con imprimir_reporte
    scores = pd.Series(
        raw_scores - raw_scores.min(),
        index=X.columns
    ).sort_values(ascending=False)

    elapsed      = time.time() - t0
    dataset_info = {
        "nombre":   df.attrs.get("nombre", "dataset"),
        "filas":    len(df),
        "columnas": len(df.columns),
    }

    imprimir_reporte(
        scores, "ReliefF  (instancia-based, sin Java)",
        "ReliefF W.", dataset_info, target_col, elapsed,
    )

    return scores
