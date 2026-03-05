"""
validacion.py
─────────────────────────────────────────────────────────────
Validaciones de calidad del dataset antes del análisis.

validar_dataset(df):
    1. Dimensiones mínimas (filas y columnas)
    2. Valores nulos por columna (% y alertas)
    3. Tipos de datos presentes
    4. Filas duplicadas
    5. Columnas sin varianza (valor constante)
    6. Alta cardinalidad en categóricas

validar_target(df, target_col):
    1. Tipo detectado (categórico vs numérico continuo)
    2. Distribución de clases y ratio de desbalance
    3. Estadísticas descriptivas + asimetría si es continuo
"""

import pandas as pd
import numpy as np
from colorama import Fore
from utils.consola import separador


# ─── VALIDACIÓN DEL DATASET ──────────────────────────────────────────────────

def validar_dataset(df: pd.DataFrame):
    """
    Valida que el DataFrame sea apto para análisis.

    Retorna:
        es_valido    (bool)       — False si hay errores críticos
        advertencias (list[str])  — problemas no bloqueantes
        errores      (list[str])  — problemas que impiden el análisis
    """
    errores      = []
    advertencias = []

    separador("DATASET VALIDATION")

    filas, cols = df.shape

    # ── 1. DIMENSIONES ───────────────────────────────────────────────────────
    print(f"\n  {Fore.CYAN}[1] Dimensions:")
    print(f"      {Fore.WHITE}{filas} instances  x  {cols} attributes")

    if filas < 10:
        errores.append(f"Too few rows ({filas}). Minimum recommended: 30.")
        print(f"  {Fore.RED}  x Too few instances ({filas}). Minimum recommended: 30.")
    elif filas < 30:
        advertencias.append(f"Few instances ({filas}). Models may not generalize well.")
        print(f"  {Fore.YELLOW}  ! Only {filas} instances — results may be unreliable.")
    else:
        print(f"  {Fore.GREEN}  v Sufficient instances.")

    if cols < 2:
        errores.append("Need at least 2 columns (1 feature + 1 target).")
        print(f"  {Fore.RED}  x Insufficient columns.")
    else:
        print(f"  {Fore.GREEN}  v Sufficient columns.")

    # ── 2. VALORES NULOS ─────────────────────────────────────────────────────
    print(f"\n  {Fore.CYAN}[2] Missing values:")
    nulos         = df.isnull().sum()
    pct_nulos     = (nulos / filas * 100).round(2)
    cols_con_nulos = nulos[nulos > 0]

    if len(cols_con_nulos) == 0:
        print(f"  {Fore.GREEN}  v No missing values.")
    else:
        for col, n in cols_con_nulos.items():
            pct   = pct_nulos[col]
            color = Fore.RED if pct > 40 else Fore.YELLOW
            print(f"  {color}  ! {col:<35} {n:>5} missing  ({pct:.1f}%)")
            if pct > 40:
                advertencias.append(f"'{col}' has {pct:.1f}% missing — consider dropping.")
            else:
                advertencias.append(f"'{col}' has {pct:.1f}% missing — will impute with median.")

    # ── 3. TIPOS DE DATOS ────────────────────────────────────────────────────
    n_num = len(df.select_dtypes(include=["number"]).columns)
    n_cat = len(df.select_dtypes(include=["object", "category"]).columns)
    print(f"\n  {Fore.CYAN}[3] Column types:")
    print(f"      {Fore.WHITE}Numeric    : {Fore.GREEN}{n_num}")
    print(f"      {Fore.WHITE}Categorical: {Fore.YELLOW}{n_cat}")

    if n_num == 0:
        advertencias.append("No numeric columns — Pearson correlation will not be useful.")
        print(f"  {Fore.YELLOW}  ! No numeric columns — Correlation method limited.")

    # ── 4. DUPLICADOS ────────────────────────────────────────────────────────
    n_dup = df.duplicated().sum()
    print(f"\n  {Fore.CYAN}[4] Duplicate rows:")
    if n_dup == 0:
        print(f"  {Fore.GREEN}  v No duplicates.")
    else:
        pct_dup = round(n_dup / filas * 100, 2)
        color   = Fore.RED if pct_dup > 20 else Fore.YELLOW
        print(f"  {color}  ! {n_dup} duplicate rows ({pct_dup:.1f}%)")
        if pct_dup > 20:
            advertencias.append(f"{pct_dup:.1f}% duplicate rows — may bias results.")

    # ── 5. COLUMNAS SIN VARIANZA ─────────────────────────────────────────────
    sin_varianza = [c for c in df.columns if df[c].nunique() <= 1]
    print(f"\n  {Fore.CYAN}[5] Zero-variance columns:")
    if not sin_varianza:
        print(f"  {Fore.GREEN}  v All columns have variance.")
    else:
        for c in sin_varianza:
            print(f"  {Fore.RED}  x '{c}' — constant value (no information)")
        errores.append(f"Constant columns detected: {sin_varianza}")

    # ── 6. ALTA CARDINALIDAD ─────────────────────────────────────────────────
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns
    alta_card = [(col, df[col].nunique()) for col in cat_cols if df[col].nunique() > 50]
    print(f"\n  {Fore.CYAN}[6] Categorical cardinality:")
    if not alta_card:
        print(f"  {Fore.GREEN}  v Cardinality OK in all categorical columns.")
    else:
        for col, uniq in alta_card:
            print(f"  {Fore.YELLOW}  ! '{col}' — {uniq} unique values (high cardinality)")
            advertencias.append(f"'{col}' has {uniq} unique categories — high cardinality.")

    # ── RESUMEN ──────────────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}  {'─'*60}")
    if errores:
        print(f"  {Fore.RED}CRITICAL ERRORS ({len(errores)}):")
        for e in errores:
            print(f"    {Fore.RED}• {e}")
    if advertencias:
        print(f"  {Fore.YELLOW}WARNINGS ({len(advertencias)}):")
        for a in advertencias:
            print(f"    {Fore.YELLOW}• {a}")
    if not errores and not advertencias:
        print(f"  {Fore.GREEN}Dataset is VALID. No issues found.")
    elif not errores:
        print(f"\n  {Fore.GREEN}Dataset is SUITABLE for analysis (check warnings).")
    else:
        print(f"\n  {Fore.RED}Dataset is NOT SUITABLE. Fix critical errors first.")
    print(f"{Fore.CYAN}  {'─'*60}\n")

    return len(errores) == 0, advertencias, errores


# ─── VALIDACIÓN DEL TARGET ────────────────────────────────────────────────────

def validar_target(df: pd.DataFrame, target_col: str):
    """
    Valida la columna target:
      - Categórico  → distribución de clases + ratio desbalance
      - Continuo    → estadísticas descriptivas + asimetría
    """
    separador(f"TARGET VALIDATION: '{target_col}'")

    y        = df[target_col]
    n_unique = y.nunique()
    es_cat   = (y.dtype == object) or (str(y.dtype) == "category") or (n_unique <= 20)

    tipo_str = "Categorical / Classification" if es_cat else "Numeric / Regression"
    print(f"\n  {Fore.CYAN}Detected type : {Fore.WHITE}{tipo_str}")
    print(f"  {Fore.CYAN}Unique values : {Fore.WHITE}{n_unique}")
    print(f"  {Fore.CYAN}Missing       : {Fore.WHITE}{y.isnull().sum()}")

    if es_cat:
        _validar_clases(y)
    else:
        _validar_continuo(y)

    print(f"{Fore.CYAN}  {'─'*60}\n")


def _validar_clases(y: pd.Series):
    """Muestra distribución de clases y detecta desbalance."""
    conteo = y.value_counts()
    total  = len(y)

    print(f"\n  {Fore.WHITE}  Class distribution:")
    print(f"  {Fore.CYAN}  {'Class':<25} {'N':>6}  {'%':>6}  Bar")
    print(f"  {Fore.CYAN}  {'─'*55}")

    for clase, cnt in conteo.items():
        pct   = cnt / total * 100
        bar   = chr(9608) * int(pct / 2.5)
        color = Fore.GREEN if pct > 30 else Fore.YELLOW if pct > 10 else Fore.RED
        print(f"  {color}  {str(clase):<25} {cnt:>6}  {pct:>5.1f}%  {Fore.CYAN}{bar}")

    if conteo.min() > 0:
        ratio = conteo.max() / conteo.min()
        print(f"\n  {Fore.CYAN}  Imbalance ratio: {Fore.WHITE}{ratio:.1f}:1")
        if ratio > 10:
            print(f"  {Fore.RED}  x SEVERE imbalance — consider SMOTE or oversampling.")
        elif ratio > 3:
            print(f"  {Fore.YELLOW}  ! Moderate imbalance — monitor per-class metrics.")
        else:
            print(f"  {Fore.GREEN}  v Classes are balanced.")


def _validar_continuo(y: pd.Series):
    """Muestra estadísticas descriptivas del target continuo."""
    print(f"\n  {Fore.WHITE}  Descriptive statistics:")
    stats = {
        "Mean":     y.mean(),
        "Median":   y.median(),
        "Std":      y.std(),
        "Min":      y.min(),
        "Max":      y.max(),
        "Skewness": y.skew(),
        "Kurtosis": y.kurt(),
    }
    for k, v in stats.items():
        print(f"    {Fore.CYAN}{k:<12} {Fore.WHITE}{v:.4f}")

    skew = abs(y.skew())
    if skew > 2:
        print(f"\n  {Fore.YELLOW}  ! High skewness ({y.skew():.2f}) — consider log/sqrt transform.")
    elif skew > 1:
        print(f"\n  {Fore.YELLOW}  ! Moderate skewness ({y.skew():.2f}).")
    else:
        print(f"\n  {Fore.GREEN}  v Distribution looks acceptable.")
