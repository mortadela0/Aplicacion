"""
metodos/random_forest.py
─────────────────────────────────────────────
MÉTODO 3 — Random Forest (Gini / MDI)

Soporta target numérico Y string/categórico.
Detecta automáticamente Clasificación vs Regresión:
  - String / categórico / <= 20 únicos → Classifier
  - Numérico continuo                  → Regressor
"""

import time
import pandas as pd
import numpy as np
from colorama import Fore
from utils.preprocesamiento import preparar_datos, target_es_categorico
from utils.reporte_weka     import imprimir_reporte


def ejecutar(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Entrena Random Forest y retorna importancias Gini.

    Retorna:
        pd.Series — importancias ordenadas desc.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    t0 = time.time()

    X, y   = preparar_datos(df, target_col)
    es_clf = target_es_categorico(df, target_col)

    if es_clf:
        modelo = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        tipo   = "Classifier"
    else:
        modelo = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        tipo   = "Regressor"

    print(f"\n  {Fore.YELLOW}[INFO] RandomForest{tipo} — training 150 trees...")
    modelo.fit(X, y)

    importancias = (pd.Series(modelo.feature_importances_, index=X.columns)
                      .sort_values(ascending=False))

    elapsed      = time.time() - t0
    dataset_info = {
        "nombre":   df.attrs.get("nombre", "dataset.csv"),
        "filas":    len(df),
        "columnas": len(df.columns),
    }

    imprimir_reporte(
        scores       = importancias,
        metodo       = f"RandomForest{tipo} — Gini Importance (MDI)",
        nombre_score = "Gini Imp.",
        dataset_info = dataset_info,
        target_col   = target_col,
        elapsed      = elapsed,
        decimales    = 10,
        umbral_pct   = 50.0,
    )

    # Tabla de importancia acumulada
    acumulado = 0.0
    umbral_80 = None
    umbral_95 = None

    print(f"{Fore.WHITE}  Cumulative importance:")
    print(f"{Fore.CYAN}  {'─'*60}")
    print(f"{Fore.WHITE}  {'Rank':<6} {'Importance':>12}   {'Cumulative':>12}   Attribute")
    print(f"{Fore.CYAN}  {'─'*60}")

    for rank, (attr, val) in enumerate(importancias.items(), 1):
        acumulado += val
        color  = (Fore.GREEN  if acumulado <= 0.80 else
                  Fore.YELLOW if acumulado <= 0.95 else
                  Fore.WHITE)
        marca = ""
        if umbral_80 is None and acumulado >= 0.80:
            umbral_80 = rank
            marca = f"  {Fore.YELLOW}<-- 80%"
        elif umbral_95 is None and acumulado >= 0.95:
            umbral_95 = rank
            marca = f"  {Fore.WHITE}<-- 95%"

        print(f"  {color}{rank:<6} {val:>12.6f}   {acumulado:>12.6f}   {rank}. {attr}{marca}")

    print(f"{Fore.CYAN}  {'─'*60}")
    if umbral_80:
        print(f"{Fore.WHITE}  Top {umbral_80} attribute(s) explain >= 80% of total importance.")
    if umbral_95:
        print(f"{Fore.WHITE}  Top {umbral_95} attribute(s) explain >= 95% of total importance.\n")

    return importancias
