"""
metodos/chi2.py
─────────────────────────────────────────────
MÉTODO 2 — SelectKBest con F-score (ANOVA F-test)

Soporta target numérico Y string/categórico.
Detecta automáticamente el tipo y usa:
  - f_classif  → target categórico / string
  - f_regression → target numérico continuo
"""

import time
import pandas as pd
import numpy as np
from colorama import Fore
from utils.preprocesamiento import preparar_datos, escalar_minmax, target_es_categorico
from utils.reporte_weka     import imprimir_reporte


def ejecutar(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Aplica SelectKBest al dataset.
    Elige f_classif o f_regression según el tipo del target.

    Retorna:
        pd.Series — F-scores ordenados desc.
    """
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    t0 = time.time()

    X, y     = preparar_datos(df, target_col)
    X_scaled = escalar_minmax(X)
    es_cat   = target_es_categorico(df, target_col)

    if es_cat:
        score_func = f_classif
        tipo_str   = "f_classif (clasificacion)"
    else:
        score_func = f_regression
        tipo_str   = "f_regression (regresion)"

    print(f"\n  {Fore.YELLOW}[INFO] Score function: {tipo_str}")

    selector = SelectKBest(score_func=score_func, k="all")
    selector.fit(X_scaled, y)

    scores  = pd.Series(selector.scores_,  index=X.columns).sort_values(ascending=False)
    pvalues = pd.Series(selector.pvalues_, index=X.columns)

    # Limpiar NaN (puede ocurrir si alguna feature es constante)
    scores  = scores.dropna()
    pvalues = pvalues.reindex(scores.index)

    elapsed      = time.time() - t0
    dataset_info = {
        "nombre":   df.attrs.get("nombre", "dataset.csv"),
        "filas":    len(df),
        "columnas": len(df.columns),
    }

    imprimir_reporte(
        scores       = scores,
        metodo       = f"Chi2/F-score — SelectKBest ({tipo_str})",
        nombre_score = "F-Score",
        dataset_info = dataset_info,
        target_col   = target_col,
        elapsed      = elapsed,
        decimales    = 10,
        umbral_pct   = 50.0,
    )

    # Tabla extra de p-valores
    print(f"{Fore.WHITE}  P-values detail:")
    print(f"{Fore.CYAN}  {'─'*60}")
    print(f"{Fore.WHITE}  {'Rank':<6} {'p-value':>12}   {'Sig':>5}   Attribute")
    print(f"{Fore.CYAN}  {'─'*60}")

    for rank, (attr, _) in enumerate(scores.items(), 1):
        pval  = pvalues.get(attr, float("nan"))
        if pd.isna(pval):
            sig, color = "n/a", Fore.WHITE
        else:
            sig   = ("***" if pval < 0.001 else
                     "**"  if pval < 0.01  else
                     "*"   if pval < 0.05  else "n.s.")
            color = (Fore.GREEN  if sig == "***" else
                     Fore.YELLOW if sig in ("**", "*") else
                     Fore.WHITE)
        print(f"  {color}{rank:<6} {str(round(pval,6)):>12}   {sig:>5}   {rank}. {attr}")

    print(f"{Fore.CYAN}  {'─'*60}")
    print(f"{Fore.WHITE}  *** p<0.001  ** p<0.01  * p<0.05  n.s. = not significant\n")

    return scores
