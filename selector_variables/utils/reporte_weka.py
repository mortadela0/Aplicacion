"""
utils/reporte_weka.py
─────────────────────────────────────────────
Formateador de salida estilo Weka.
Usa precisión completa (sin redondeo) para comparar con Weka Explorer.
"""

import datetime
import pandas as pd
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

ANCHO = 70


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _linea(char: str = "=") -> str:
    return char * ANCHO


def _fmt(val: float, decimales: int = 10) -> str:
    """
    Formatea un número con precisión máxima, igual que Weka Explorer:
      - Nunca trunca por debajo de lo que devuelve el evaluador
      - Notación científica si el valor es < 1e-6
      - Elimina ceros finales innecesarios
      - Garantiza al menos 4 decimales si hay parte decimal

    Ejemplos:
      0.857142857142857  →  "0.8571428571"
      0.0               →  "0"
      1.23456789e-9     →  "1.234568e-09"
      1.0               →  "1.0"
    """
    if val == 0.0:
        return "0"
    if abs(val) < 1e-6:
        return f"{val:.6e}"
    # Formato con todos los decimales, eliminar ceros finales
    s = f"{val:.{decimales}f}".rstrip("0").rstrip(".")
    # Garantizar mínimo 4 decimales
    if "." in s:
        parte_decimal = s.split(".")[1]
        if len(parte_decimal) < 4:
            s = f"{val:.4f}"
    else:
        # Número entero resultado — añadir .0 para claridad
        s = s + ".0"
    return s


def _cabecera(metodo: str, dataset_info: dict, target_col: str):
    """Cabecera idéntica a Weka Explorer: evaluador, relación, instancias, timestamp."""
    ahora  = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y")
    filas  = dataset_info.get("filas",    "?")
    cols   = dataset_info.get("columnas", "?")
    nombre = dataset_info.get("nombre",   "dataset.csv")

    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Attribute Evaluator  (supervised, Class: {target_col})")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  Evaluator   : {Fore.YELLOW}{metodo}")
    print(f"{Fore.WHITE}  Relation    : {Fore.WHITE}{nombre}")
    print(f"{Fore.WHITE}  Instances   : {Fore.WHITE}{filas}")
    print(f"{Fore.WHITE}  Attributes  : {Fore.WHITE}{cols}")
    print(f"{Fore.WHITE}  Target      : {Fore.WHITE}{target_col}")
    print(f"{Fore.WHITE}  Timestamp   : {Fore.WHITE}{ahora}")
    print(f"{Fore.CYAN}{_linea('=')}")


def _tabla_ranked(scores: pd.Series, nombre_score: str, decimales: int = 10):
    """
    Tabla ranked con precisión completa — sin redondeo, igual que Weka.

    Columnas:  Rank | Score (full precision) | Merit % | Attribute
    """
    print(f"\n{Fore.WHITE}  Ranked attributes:")
    print(f"{Fore.CYAN}  {_linea('-')}")

    # Ancho de columna score basado en el valor más largo
    vals_fmt = [_fmt(v, decimales) for v in scores.values]
    col_w    = max(max((len(s) for s in vals_fmt), default=12),
                  len(nombre_score), 12)

    print(f"{Fore.WHITE}  {'Rank':<6} {nombre_score:>{col_w}}   {'Merit %':>8}   Attribute")
    print(f"{Fore.CYAN}  {_linea('-')}")

    max_val = scores.max() if scores.max() > 0 else 1.0

    for rank, ((attr, val), val_str) in enumerate(
            zip(scores.items(), vals_fmt), start=1):

        merit_pct = (val / max_val) * 100

        if rank <= 3:
            color = Fore.GREEN
        elif rank <= max(3, int(len(scores) * 0.5)):
            color = Fore.YELLOW
        else:
            color = Fore.WHITE

        print(
            f"  {color}{rank:<6} {val_str:>{col_w}}"
            f"   {merit_pct:>7.2f}%"
            f"   {rank}. {attr}"
        )

    print(f"{Fore.CYAN}  {_linea('-')}")


def _tabla_selected(scores: pd.Series, umbral_pct: float = 50.0) -> list:
    """Atributos cuyo merit supera el umbral — 'Selected attributes' de Weka."""
    max_val       = scores.max() if scores.max() > 0 else 1.0
    seleccionados = [
        attr for attr, val in scores.items()
        if (val / max_val) * 100 >= umbral_pct
    ]

    print(
        f"\n{Fore.WHITE}  Selected attributes "
        f"({Fore.YELLOW}{len(seleccionados)}{Fore.WHITE} / {len(scores)})"
        f"  [merit >= {umbral_pct:.0f}%]"
    )
    print(f"{Fore.CYAN}  {_linea('-')}")

    for i, attr in enumerate(seleccionados, 1):
        print(f"  {Fore.GREEN}  {i}. {attr}")

    if not seleccionados:
        print(f"  {Fore.YELLOW}  (ninguno supera el umbral)")

    print(f"{Fore.CYAN}  {_linea('-')}")
    return seleccionados


def _pie(elapsed: float, scores: pd.Series, decimales: int = 10):
    """Pie de reporte con estadísticas globales en precisión completa."""
    print(f"\n{Fore.WHITE}  Evaluation time  : {Fore.YELLOW}{elapsed:.4f}s")
    print(f"{Fore.WHITE}  Total attributes : {Fore.WHITE}{len(scores)}")
    print(f"{Fore.WHITE}  Mean merit       : {Fore.WHITE}{_fmt(scores.mean(), decimales)}")
    print(f"{Fore.WHITE}  Max merit        : {Fore.WHITE}{_fmt(scores.max(),  decimales)}")
    print(f"{Fore.WHITE}  Min merit        : {Fore.WHITE}{_fmt(scores.min(),  decimales)}")
    print(f"{Fore.WHITE}  Std merit        : {Fore.WHITE}{_fmt(scores.std(),  decimales)}")
    print(f"{Fore.CYAN}{_linea('=')}\n")


# ─── FUNCIONES PÚBLICAS ───────────────────────────────────────────────────────

def imprimir_reporte(
    scores:       pd.Series,
    metodo:       str,
    nombre_score: str,
    dataset_info: dict,
    target_col:   str,
    elapsed:      float,
    decimales:    int   = 10,
    umbral_pct:   float = 50.0,
):
    """
    Reporte completo estilo Weka para un método individual.
    Por defecto usa 10 decimales (comparable con Weka Explorer).
    """
    _cabecera(metodo, dataset_info, target_col)
    _tabla_ranked(scores, nombre_score, decimales)
    _tabla_selected(scores, umbral_pct)
    _pie(elapsed, scores, decimales)


def imprimir_reporte_comparativo(
    resumen:      pd.DataFrame,
    dataset_info: dict,
    target_col:   str,
):
    """
    Tabla comparativa de todos los métodos, estilo Weka Search Results.
    Scores normalizados [0,1] con precisión completa.
    """
    ahora  = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y")
    filas  = dataset_info.get("filas",  "?")
    nombre = dataset_info.get("nombre", "dataset.csv")

    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Attribute Selection Summary")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  Relation  : {nombre}")
    print(f"{Fore.WHITE}  Instances : {filas}")
    print(f"{Fore.WHITE}  Target    : {target_col}")
    metodos_str = " | ".join(c for c in resumen.columns if c != "SCORE_FINAL")
    print(f"{Fore.WHITE}  Methods   : {metodos_str}")
    print(f"{Fore.WHITE}  Timestamp : {ahora}")
    print(f"{Fore.CYAN}{_linea('=')}")

    cols_metodos = [c for c in resumen.columns if c != "SCORE_FINAL"]

    # Encabezado con columnas fijas de 12 caracteres (como Weka)
    header = f"\n  {'Rank':<5} {'Attribute':<28}"
    for c in cols_metodos:
        header += f" {c[:10]:>12}"
    header += f" {'FINAL':>12}   Merit bar"
    print(f"{Fore.WHITE}{header}")
    print(f"{Fore.CYAN}  {_linea('-')}")

    for i, (attr, row) in enumerate(resumen.iterrows(), 1):
        final = row["SCORE_FINAL"]
        barra = "*" * int(final * 30)

        if i <= 3:
            color = Fore.GREEN
        elif i <= max(3, int(len(resumen) * 0.5)):
            color = Fore.YELLOW
        else:
            color = Fore.WHITE

        fila = f"  {color}{i:<5} {attr:<28}"
        for c in cols_metodos:
            fila += f" {_fmt(row[c], 8):>12}"
        fila += f" {_fmt(final, 8):>12}   {Fore.CYAN}{barra}"
        print(fila)

    print(f"{Fore.CYAN}  {_linea('-')}")

    # Top 3
    print(f"\n{Fore.WHITE}  Best 3 attributes (by SCORE_FINAL):")
    medallas = {1: "***", 2: "** ", 3: "*  "}
    for i, (attr, val) in enumerate(resumen["SCORE_FINAL"].nlargest(3).items(), 1):
        print(f"  {Fore.GREEN}    {medallas[i]}  {i}. {attr:<35} {_fmt(val, 8)}")

    # Selected >= 0.5
    sel = resumen[resumen["SCORE_FINAL"] >= 0.5].index.tolist()
    print(f"\n{Fore.WHITE}  Selected attributes (SCORE_FINAL >= 0.5): "
          f"{Fore.YELLOW}{len(sel)}{Fore.WHITE} / {len(resumen)}")
    for i, attr in enumerate(sel, 1):
        print(f"  {Fore.GREEN}    {i}. {attr}")

    print(f"{Fore.CYAN}{_linea('=')}\n")


def imprimir_validacion_vs_weka(
    resumen_propio:  pd.DataFrame,
    resultados_weka: dict,
    target_col:      str,
):
    """
    Compara rankings propios contra evaluadores de Weka.
    Todos los scores se muestran con precisión completa.
    """
    from utils.preprocesamiento import normalizar_serie

    if not resultados_weka.get("disponible"):
        print(f"\n  {Fore.YELLOW}[SKIP] Weka not available.")
        return

    ahora = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Cross-Validation: Own Methods  vs  Weka Evaluators")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  Target    : {target_col}")
    print(f"{Fore.WHITE}  Timestamp : {ahora}")
    print(f"{Fore.CYAN}{_linea('=')}")

    # Promedio Weka
    weka_series = []
    for key in ["InfoGain", "GainRatio", "Correlation", "ReliefF"]:
        if key in resultados_weka:
            weka_series.append(normalizar_serie(resultados_weka[key]))

    if weka_series:
        weka_avg = pd.concat(weka_series, axis=1).mean(axis=1).sort_values(ascending=False)
        top_weka = set(weka_avg.nlargest(5).index.tolist())
    else:
        weka_avg = pd.Series(dtype=float)
        top_weka = set()

    top_propios = set(resumen_propio["SCORE_FINAL"].nlargest(5).index.tolist())
    cfs_subset  = set(resultados_weka.get("CFS_subset", []))

    todos_attrs = sorted(resumen_propio.index.tolist())

    print(f"\n{Fore.WHITE}  {'Attribute':<32} {'Own':>12} {'Weka':>12} {'CFS':>5}  Status")
    print(f"{Fore.CYAN}  {_linea('-')}")

    acuerdos    = []
    desacuerdos = []

    for attr in todos_attrs:
        own_score  = resumen_propio.loc[attr, "SCORE_FINAL"] if attr in resumen_propio.index else 0.0
        weka_score = weka_avg[attr] if (len(weka_avg) > 0 and attr in weka_avg.index) else 0.0
        en_cfs     = "YES" if attr in cfs_subset else "no"
        en_top_own = attr in top_propios
        en_top_wk  = attr in top_weka

        if en_top_own and en_top_wk:
            status = "CONFIRMED"
            color  = Fore.GREEN
            acuerdos.append(attr)
        elif en_top_own and not en_top_wk:
            status = "own only"
            color  = Fore.YELLOW
            desacuerdos.append(attr)
        elif not en_top_own and en_top_wk:
            status = "weka only"
            color  = Fore.CYAN
        else:
            status = "-"
            color  = Fore.WHITE

        cfs_c = Fore.GREEN if en_cfs == "YES" else Fore.WHITE
        print(
            f"  {color}{attr:<32}"
            f" {_fmt(own_score,  6):>12}"
            f" {_fmt(weka_score, 6):>12}"
            f" {cfs_c}{en_cfs:>5}{color}  {status}"
        )

    print(f"{Fore.CYAN}  {_linea('-')}")

    # Resumen
    print(f"\n{Fore.WHITE}  Agreement Summary:")
    print(f"{Fore.CYAN}  {_linea('-')}")

    conf_cfs = [a for a in acuerdos if a in cfs_subset]
    print(f"  {Fore.GREEN}  CONFIRMED top-5 + CFS ({len(conf_cfs)}):")
    for a in conf_cfs:
        print(f"  {Fore.GREEN}    * {a}")

    conf_no_cfs = [a for a in acuerdos if a not in cfs_subset]
    print(f"\n  {Fore.GREEN}  CONFIRMED top-5, not in CFS ({len(conf_no_cfs)}):")
    for a in conf_no_cfs:
        print(f"  {Fore.GREEN}    * {a}")

    print(f"\n  {Fore.YELLOW}  Only in own top-5 — revisar ({len(desacuerdos)}):")
    for a in desacuerdos:
        print(f"  {Fore.YELLOW}    ? {a}")

    # Recomendadas
    recomendadas = list(set(acuerdos) | cfs_subset)
    recomendadas_sorted = (
        resumen_propio.reindex(recomendadas)["SCORE_FINAL"]
        .dropna()
        .sort_values(ascending=False)
        .index.tolist()
    )

    print(f"\n{Fore.CYAN}  {_linea('-')}")
    print(f"{Fore.WHITE}  RECOMMENDED ATTRIBUTES")
    print(f"{Fore.WHITE}  (confirmed by own methods AND Weka):")
    print(f"{Fore.CYAN}  {_linea('-')}")
    for i, attr in enumerate(recomendadas_sorted, 1):
        score = resumen_propio.loc[attr, "SCORE_FINAL"] if attr in resumen_propio.index else 0
        print(f"  {Fore.GREEN}  {i}. {attr:<35} {_fmt(score, 8)}")

    print(f"{Fore.CYAN}{_linea('=')}\n")
