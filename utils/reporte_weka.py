"""
utils/reporte_weka.py
─────────────────────────────────────────────
Formateador de salida estilo Weka.
"""
import datetime
import pandas as pd
import numpy as np
from colorama import Fore, Style, init
init(autoreset=True)

ANCHO = 70

def _linea(char="="):
    return char * ANCHO

def _fmt(val, decimales=10):
    if val == 0.0: return "0"
    if abs(val) < 1e-6: return f"{val:.6e}"
    s = f"{val:.{decimales}f}".rstrip("0").rstrip(".")
    if "." in s:
        if len(s.split(".")[1]) < 4:
            s = f"{val:.4f}"
    else:
        s = s + ".0"
    return s

def _cabecera(metodo, dataset_info, target_col):
    ahora  = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    filas  = dataset_info.get("filas","?")
    cols   = dataset_info.get("columnas","?")
    nombre = dataset_info.get("nombre","dataset.csv")
    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Attribute Evaluator  (supervised, Class: {target_col})")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  Evaluator   : {Fore.YELLOW}{metodo}")
    print(f"{Fore.WHITE}  Relation    : {nombre}")
    print(f"{Fore.WHITE}  Instances   : {filas}  |  Attributes: {cols}")
    print(f"{Fore.WHITE}  Timestamp   : {ahora}")
    print(f"{Fore.CYAN}{_linea('=')}")

def _tabla_ranked(scores, nombre_score, decimales=10):
    print(f"\n{Fore.WHITE}  Ranked attributes:")
    print(f"{Fore.CYAN}  {_linea('-')}")
    vals_fmt = [_fmt(v, decimales) for v in scores.values]
    col_w    = max(max((len(s) for s in vals_fmt), default=12), len(nombre_score), 12)
    print(f"{Fore.WHITE}  {'Rank':<6} {nombre_score:>{col_w}}   {'Merit %':>8}   Attribute")
    print(f"{Fore.CYAN}  {_linea('-')}")
    max_val = scores.max() if scores.max() > 0 else 1.0
    for rank, ((attr, val), val_str) in enumerate(zip(scores.items(), vals_fmt), start=1):
        merit_pct = (val / max_val) * 100
        color = Fore.GREEN if rank<=3 else Fore.YELLOW if rank<=max(3,int(len(scores)*0.5)) else Fore.WHITE
        print(f"  {color}{rank:<6} {val_str:>{col_w}}   {merit_pct:>7.2f}%   {rank}. {attr}")
    print(f"{Fore.CYAN}  {_linea('-')}")

def _tabla_selected(scores, umbral_pct=50.0):
    max_val = scores.max() if scores.max() > 0 else 1.0
    sel     = [attr for attr,val in scores.items() if (val/max_val)*100 >= umbral_pct]
    print(f"\n{Fore.WHITE}  Selected attributes ({Fore.YELLOW}{len(sel)}{Fore.WHITE} / {len(scores)})  [merit >= {umbral_pct:.0f}%]")
    print(f"{Fore.CYAN}  {_linea('-')}")
    for i, attr in enumerate(sel, 1):
        print(f"  {Fore.GREEN}  {i}. {attr}")
    if not sel: print(f"  {Fore.YELLOW}  (ninguno supera el umbral)")
    print(f"{Fore.CYAN}  {_linea('-')}")
    return sel

def _pie(elapsed, scores, decimales=10):
    print(f"\n{Fore.WHITE}  Evaluation time  : {Fore.YELLOW}{elapsed:.4f}s")
    print(f"{Fore.WHITE}  Total attributes : {len(scores)}")
    print(f"{Fore.WHITE}  Mean merit       : {_fmt(scores.mean(), decimales)}")
    print(f"{Fore.WHITE}  Max merit        : {_fmt(scores.max(),  decimales)}")
    print(f"{Fore.CYAN}{_linea('=')}\n")

def imprimir_reporte(scores, metodo, nombre_score, dataset_info, target_col,
                     elapsed, decimales=10, umbral_pct=50.0):
    _cabecera(metodo, dataset_info, target_col)
    _tabla_ranked(scores, nombre_score, decimales)
    _tabla_selected(scores, umbral_pct)
    _pie(elapsed, scores, decimales)

def imprimir_reporte_comparativo(resumen, dataset_info, target_col):
    from utils.preprocesamiento import normalizar_serie
    ahora  = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    nombre = dataset_info.get("nombre","dataset.csv")
    filas  = dataset_info.get("filas","?")
    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Attribute Selection Summary")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  Relation  : {nombre}  |  Instances: {filas}  |  Target: {target_col}")
    print(f"{Fore.WHITE}  Timestamp : {ahora}")
    print(f"{Fore.CYAN}{_linea('=')}")
    cols_m = [c for c in resumen.columns if c != "SCORE_FINAL"]
    header = f"\n  {'Rank':<5} {'Attribute':<28}"
    for c in cols_m: header += f" {c[:10]:>12}"
    header += f" {'FINAL':>12}   Bar"
    print(f"{Fore.WHITE}{header}")
    print(f"{Fore.CYAN}  {_linea('-')}")
    for i, (attr, row) in enumerate(resumen.iterrows(), 1):
        final = row["SCORE_FINAL"]
        barra = "*" * int(final * 30)
        color = Fore.GREEN if i<=3 else Fore.YELLOW if i<=max(3,int(len(resumen)*0.5)) else Fore.WHITE
        fila  = f"  {color}{i:<5} {attr:<28}"
        for c in cols_m: fila += f" {_fmt(row[c],8):>12}"
        fila += f" {_fmt(final,8):>12}   {Fore.CYAN}{barra}"
        print(fila)
    print(f"{Fore.CYAN}  {_linea('-')}")
    sel = resumen[resumen["SCORE_FINAL"] >= 0.5].index.tolist()
    print(f"\n{Fore.WHITE}  Selected (SCORE >= 0.5): {Fore.YELLOW}{len(sel)}{Fore.WHITE} / {len(resumen)}")
    for i, attr in enumerate(sel, 1):
        print(f"  {Fore.GREEN}    {i}. {attr}")
    print(f"{Fore.CYAN}{_linea('=')}\n")
