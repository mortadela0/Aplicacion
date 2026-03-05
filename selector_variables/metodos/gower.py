"""
metodos/gower.py
─────────────────────────────────────────────
MÉTODO 4 — Distancia / Similitud de Gower

Maneja datos MIXTOS (numérico + categórico) sin encoding previo.

Distancia por variable:
  Numérica    → |xi - xj| / rango   (normalizada a [0,1])
  Categórica  → 0 si iguales, 1 si distintos

Importancia = ratio(inter_clase / intra_clase) × contribución_media
  Score alto → variable separa bien las clases del target.

Mejor para : datasets mixtos, sin encoding previo.
Limitación : O(n²) — subsampling automático si n > 2000.
"""

import time
import pandas as pd
import numpy as np
from colorama import Fore
from utils.reporte_weka import imprimir_reporte


# ─── CÁLCULO DE IMPORTANCIA POR GOWER ────────────────────────────────────────

def importancia_por_gower(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Calcula el score de importancia de cada feature usando distancia de Gower.

    Para cada variable calcula:
      - d_inter : distancia media entre pares de DISTINTA clase
      - d_intra : distancia media entre pares de MISMA clase
      - ratio   = d_inter / (d_intra + eps)
      - score   = ratio × contribución_media

    Retorna pd.Series ordenado de mayor a menor.
    """
    features = df.drop(columns=[target_col])
    target   = df[target_col].astype(str).values
    n        = len(features)

    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    todas    = num_cols + cat_cols

    if not todas:
        return pd.Series(dtype=float)

    # Pre-calcular rangos para variables numéricas
    rangos = {}
    for col in num_cols:
        r = features[col].max() - features[col].min()
        rangos[col] = r if r > 0 else 1.0

    # Índices de todos los pares (triángulo superior)
    idx_i, idx_j = np.triu_indices(n, k=1)
    misma_clase  = target[idx_i] == target[idx_j]
    inter_mask   = ~misma_clase
    intra_mask   = misma_clase

    resultados = {}

    for col in todas:
        if col in num_cols:
            vals     = features[col].fillna(features[col].median()).values.astype(float)
            dist_col = np.abs(vals[idx_i] - vals[idx_j]) / rangos[col]
        else:
            vals     = features[col].astype(str).values
            dist_col = (vals[idx_i] != vals[idx_j]).astype(float)

        d_inter      = dist_col[inter_mask].mean() if inter_mask.any() else 0.0
        d_intra      = dist_col[intra_mask].mean() if intra_mask.any() else 0.0
        ratio        = d_inter / (d_intra + 1e-9)
        contribucion = dist_col.mean()

        resultados[col] = ratio * contribucion

    return pd.Series(resultados).sort_values(ascending=False)


# ─── SIMILITUD ENTRE OBSERVACIONES ───────────────────────────────────────────

def top_similares(df: pd.DataFrame, idx_ref: int, top_n: int = 5) -> pd.DataFrame:
    """
    Dada una fila de referencia, retorna las top_n más similares
    según similitud de Gower (similitud = 1 - distancia).
    """
    features = df.copy()
    n        = len(features)
    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()

    rangos = {}
    for col in num_cols:
        r = features[col].max() - features[col].min()
        rangos[col] = r if r > 0 else 1.0

    ref_row    = features.iloc[idx_ref]
    distancias = []

    for i in range(n):
        if i == idx_ref:
            distancias.append(np.nan)
            continue
        row   = features.iloc[i]
        dists = []
        for col in num_cols:
            v1 = ref_row[col] if not pd.isna(ref_row[col]) else 0
            v2 = row[col]     if not pd.isna(row[col])     else 0
            dists.append(abs(v1 - v2) / rangos[col])
        for col in cat_cols:
            dists.append(0.0 if str(ref_row[col]) == str(row[col]) else 1.0)
        distancias.append(np.mean(dists) if dists else 0.0)

    dist_series = pd.Series(distancias, index=df.index)
    similitud   = 1 - dist_series
    top         = (similitud
                   .nlargest(top_n + 1)
                   .drop(index=df.index[idx_ref], errors="ignore")
                   .head(top_n))

    resultado = df.iloc[top.index].copy()
    resultado.insert(0, "Similitud_Gower", top.values.round(4))
    return resultado


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────

def ejecutar(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Ejecuta el análisis de importancia por Gower.
    Incluye subsampling automático si n > 2000 filas.
    Ofrece sub-menú de similitud entre observaciones.

    Retorna:
        pd.Series — importancias ordenadas de mayor a menor.
    """
    n          = len(df)
    df_trabajo = df.copy()

    # Subsampling si el dataset es grande (Gower es O(n²))
    if n > 2000:
        print(f"\n  {Fore.YELLOW}[INFO] Large dataset ({n} rows). Subsampling to 2000...")
        df_trabajo = df.sample(n=2000, random_state=42).reset_index(drop=True)

    num_cols = df_trabajo.drop(columns=[target_col]).select_dtypes(include=["number"]).columns
    cat_cols = df_trabajo.drop(columns=[target_col]).select_dtypes(include=["object", "category"]).columns

    print(f"\n  {Fore.YELLOW}[INFO] GowerDistanceEval — computing pairwise distances...")
    print(f"  {Fore.CYAN}  Numeric features    : {Fore.WHITE}{len(num_cols)}")
    print(f"  {Fore.CYAN}  Categorical features: {Fore.WHITE}{len(cat_cols)}")

    t0           = time.time()
    importancias = importancia_por_gower(df_trabajo, target_col)
    elapsed      = time.time() - t0

    dataset_info = {"nombre": "dataset.csv", "filas": len(df), "columnas": len(df.columns)}

    # ── Reporte estilo Weka ───────────────────────────
    imprimir_reporte(
        scores       = importancias,
        metodo       = "GowerDistanceEval  (inter/intra-class separation ratio)",
        nombre_score = "Gower Score",
        dataset_info = dataset_info,
        target_col   = target_col,
        elapsed      = elapsed,
        decimales    = 6,
        umbral_pct   = 50.0,
    )

    # ── Tipo de variable por atributo ─────────────────
    print(f"{Fore.WHITE}  Variable type detail:")
    print(f"{Fore.CYAN}  {'─'*50}")
    for attr in importancias.index:
        tipo  = "Numeric" if attr in num_cols else "Categorical"
        color = Fore.CYAN if tipo == "Numeric" else Fore.YELLOW
        print(f"  {color}  {attr:<35} {tipo}")
    print(f"{Fore.CYAN}  {'─'*50}\n")

    # ── Sub-menú de similitud entre observaciones ─────
    _menu_similitud(df, target_col)

    return importancias


def _menu_similitud(df: pd.DataFrame, target_col: str):
    """Sub-menú opcional para explorar similitud entre observaciones."""
    print(f"  {Fore.WHITE}Explore instance similarity? (s/N): ", end="")
    resp = input(f"{Fore.GREEN}").strip().lower()
    if resp != "s":
        return

    print(f"  {Fore.WHITE}Reference instance index (0 to {len(df)-1}): ", end="")
    try:
        idx = int(input(f"{Fore.GREEN}").strip())
        if not (0 <= idx < len(df)):
            print(f"  {Fore.RED}[ERROR] Index out of range.")
            return

        print(f"\n  {Fore.YELLOW}[INFO] Computing Gower similarity...")
        similares = top_similares(df, idx_ref=idx, top_n=5)

        print(f"\n{Fore.CYAN}{'='*65}")
        print(f"{Fore.WHITE}  Instance Similarity — Reference: index {idx}")
        print(f"{Fore.CYAN}{'='*65}")
        print(f"{Fore.WHITE}  Reference instance:")
        print(f"  {Fore.CYAN}{df.iloc[[idx]].to_string()}")
        print(f"\n{Fore.WHITE}  Top 5 most similar instances:")
        print(f"{Fore.CYAN}  {'─'*60}")
        print(f"  {Fore.GREEN}{similares.to_string()}")
        print(f"{Fore.CYAN}{'='*65}\n")

    except (ValueError, IndexError):
        print(f"  {Fore.RED}[ERROR] Invalid index.")
