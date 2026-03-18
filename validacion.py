"""
validacion.py — Validaciones de calidad del dataset.
"""
import pandas as pd
import numpy as np
from colorama import Fore

def _sep(titulo=""):
    print(f"\n{Fore.CYAN}{'─'*70}")
    if titulo:
        print(f"{Fore.WHITE}  {titulo}")
        print(f"{Fore.CYAN}{'─'*70}")

def validar_dataset(df):
    errores, advertencias = [], []
    _sep("DATASET VALIDATION")
    filas, cols = df.shape
    print(f"\n  {Fore.CYAN}[1] Dimensiones: {Fore.WHITE}{filas} instancias x {cols} atributos")
    if filas < 10:
        errores.append(f"Pocas filas ({filas}). Mínimo recomendado: 30.")
        print(f"  {Fore.RED}  x Muy pocas instancias.")
    elif filas < 30:
        advertencias.append(f"Solo {filas} instancias.")
        print(f"  {Fore.YELLOW}  ! Pocas instancias.")
    else:
        print(f"  {Fore.GREEN}  v OK")
    if cols < 2:
        errores.append("Se necesitan al menos 2 columnas.")
        print(f"  {Fore.RED}  x Columnas insuficientes.")

    nulos = df.isnull().sum()
    print(f"\n  {Fore.CYAN}[2] Valores nulos:")
    cols_nulos = nulos[nulos > 0]
    if cols_nulos.empty:
        print(f"  {Fore.GREEN}  v Sin nulos")
    else:
        for col, n in cols_nulos.items():
            pct = n/filas*100
            color = Fore.RED if pct > 40 else Fore.YELLOW
            print(f"  {color}  ! {col:<35} {n:>5} ({pct:.1f}%)")
            advertencias.append(f"{col}: {pct:.1f}% nulos")

    dup = df.duplicated().sum()
    print(f"\n  {Fore.CYAN}[3] Duplicados: {Fore.WHITE}{dup}")
    if dup > 0:
        advertencias.append(f"{dup} filas duplicadas")
        print(f"  {Fore.YELLOW}  ! {dup} filas duplicadas")

    sin_var = [c for c in df.columns if df[c].nunique() <= 1]
    print(f"\n  {Fore.CYAN}[4] Columnas constantes: {Fore.WHITE}{len(sin_var)}")
    for c in sin_var:
        errores.append(f"Columna constante: {c}")
        print(f"  {Fore.RED}  x {c}")

    print(f"\n{Fore.CYAN}  {'─'*60}")
    if errores:
        print(f"  {Fore.RED}ERRORES CRITICOS ({len(errores)}):")
        for e in errores: print(f"    {Fore.RED}• {e}")
    if advertencias:
        print(f"  {Fore.YELLOW}ADVERTENCIAS ({len(advertencias)}):")
        for a in advertencias: print(f"    {Fore.YELLOW}• {a}")
    if not errores and not advertencias:
        print(f"  {Fore.GREEN}Dataset VÁLIDO. Sin problemas.")
    print(f"{Fore.CYAN}  {'─'*60}\n")
    return len(errores) == 0, advertencias, errores

def validar_target(df, target_col):
    _sep(f"TARGET VALIDATION: '{target_col}'")
    y        = df[target_col]
    n_unique = y.nunique()
    es_cat   = (y.dtype == object) or (n_unique <= 20)
    print(f"\n  {Fore.CYAN}Tipo     : {Fore.WHITE}{'Categórico' if es_cat else 'Numérico continuo'}")
    print(f"  {Fore.CYAN}Únicos   : {Fore.WHITE}{n_unique}")
    print(f"  {Fore.CYAN}Nulos    : {Fore.WHITE}{y.isnull().sum()}")
    if y.dtype == object or str(y.dtype) == 'category':
        print(f"\n  {Fore.YELLOW}  [AVISO] El target es STRING. Se recomienda usar entero (0,1,2...) para evitar errores en SGD/SVM.")
        print(f"  {Fore.YELLOW}          El CSV de prediccion debe usar los MISMOS valores string del training.")
    if es_cat:
        conteo = y.value_counts()
        total  = len(y)
        print(f"\n  {Fore.WHITE}  Distribución de clases:")
        for clase, cnt in conteo.items():
            pct  = cnt/total*100
            bar  = '█' * int(pct/3)
            color = Fore.GREEN if pct > 30 else Fore.YELLOW if pct > 10 else Fore.RED
            print(f"  {color}  {str(clase):<25} {cnt:>6}  ({pct:5.1f}%)  {Fore.CYAN}{bar}")
        if len(conteo) > 1:
            ratio = conteo.max()/conteo.min()
            print(f"\n  {Fore.CYAN}  Desbalance: {Fore.WHITE}{ratio:.1f}:1")
    print(f"{Fore.CYAN}  {'─'*60}\n")
