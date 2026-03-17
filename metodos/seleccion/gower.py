import time
import numpy as np
import pandas as pd
from colorama import Fore
from utils.reporte_weka import imprimir_reporte

def importancia_por_gower(df, target_col):
    features = df.drop(columns=[target_col])
    target   = df[target_col].astype(str).values
    n        = len(features)
    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = features.select_dtypes(include=["object","category"]).columns.tolist()
    rangos   = {c: max(features[c].max()-features[c].min(), 1e-9) for c in num_cols}
    idx_i, idx_j = np.triu_indices(n, k=1)
    inter_mask   = target[idx_i] != target[idx_j]
    intra_mask   = ~inter_mask
    resultados   = {}
    for col in num_cols + cat_cols:
        if col in num_cols:
            vals     = features[col].fillna(features[col].median()).values.astype(float)
            dist_col = np.abs(vals[idx_i] - vals[idx_j]) / rangos[col]
        else:
            vals     = features[col].astype(str).values
            dist_col = (vals[idx_i] != vals[idx_j]).astype(float)
        d_inter = dist_col[inter_mask].mean() if inter_mask.any() else 0.0
        d_intra = dist_col[intra_mask].mean() if intra_mask.any() else 0.0
        resultados[col] = (d_inter / (d_intra + 1e-9)) * dist_col.mean()
    return pd.Series(resultados).sort_values(ascending=False)

def ejecutar(df, target_col):
    n          = len(df)
    df_trabajo = df.sample(n=2000, random_state=42).reset_index(drop=True) if n > 2000 else df.copy()
    t0         = time.time()
    scores     = importancia_por_gower(df_trabajo, target_col)
    imprimir_reporte(scores, "GowerDistanceEval", "Gower Score",
                     {"nombre": df.attrs.get("nombre","dataset"), "filas": n, "columnas": len(df.columns)},
                     target_col, time.time()-t0, decimales=6)
    return scores
