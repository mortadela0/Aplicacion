"""
╔══════════════════════════════════════════════════════════╗
║   SELECTOR DE VARIABLES v2.1                             ║
║   Correlacion | Chi2 | RandomForest | Gower | Weka API  ║
╚══════════════════════════════════════════════════════════╝

Uso:
    python selector.py

Estructura:
    selector.py              <- Punto de entrada (este archivo)
    validacion.py            <- Validacion de dataset y target
    metodos/
        correlacion.py       <- Metodo 1: Pearson |r|
        chi2.py              <- Metodo 2: F-score SelectKBest
        random_forest.py     <- Metodo 3: Gini importance
        gower.py             <- Metodo 4: Distancia de Gower
        weka_attsel.py       <- Metodo 5: Weka native evaluators
    utils/
        consola.py           <- Banner, menu, helpers UI
        preprocesamiento.py  <- Encoding, imputacion, escalado
        reporte_weka.py      <- Formateador salida estilo Weka
    datos/                   <- Coloca aqui tus CSV
    resultados/              <- Salida de exportaciones
"""

import os
import sys
import pandas as pd
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)


# ─── VERIFICAR DEPENDENCIAS BASE ─────────────────────────────────────────────
def _safe_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def verificar_dependencias():
    base = {
        "sklearn":    "scikit-learn",
        "pandas":     "pandas",
        "numpy":      "numpy",
        "colorama":   "colorama",
        "matplotlib": "matplotlib",
    }
    faltan = [pip for mod, pip in base.items() if not _safe_import(mod)]
    if faltan:
        print(f"\n{Fore.RED}[ERROR] Faltan dependencias obligatorias:")
        for p in faltan:
            print(f"  {Fore.YELLOW}pip install {p}")
        print(f"\n{Fore.YELLOW}  O ejecuta: pip install -r requirements.txt")
        sys.exit(1)


def crear_estructura():
    for d in ["datos", "resultados"]:
        os.makedirs(d, exist_ok=True)


# ─── ARRANQUE ─────────────────────────────────────────────────────────────────
verificar_dependencias()
crear_estructura()

from utils.consola          import (banner, separador, menu_principal,
                                    pausar, elegir_target, limpiar_pantalla)
from utils.preprocesamiento import normalizar_serie
from utils.reporte_weka     import (imprimir_reporte_comparativo,
                                    imprimir_validacion_vs_weka)
from validacion             import validar_dataset, validar_target

import metodos.correlacion   as m_correlacion
import metodos.chi2          as m_chi2
import metodos.random_forest as m_rf
import metodos.gower         as m_gower
import metodos.weka_attsel   as m_weka
import metodos.IBk           as m_ibk


# ─── CARGA DE CSV ─────────────────────────────────────────────────────────────
def cargar_csv():
    """
    Carga un CSV desde disco o usa el dataset demo (Iris).
    Retorna (DataFrame, nombre_archivo) o (None, None) si falla.
    """
    print(f"\n  {Fore.CYAN}Carpeta recomendada: {Fore.WHITE}datos/")
    ruta = input(f"\n  {Fore.WHITE}Ruta del CSV (Enter = demo Iris): {Fore.GREEN}").strip()

    # ── Dataset demo ──────────────────────────────────
    if not ruta:
        print(f"  {Fore.YELLOW}[INFO] Cargando dataset demo (Iris)...")
        try:
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df   = iris.frame
            df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
            # Convertir target numérico a nombre de especie (string) para demo
            nombres = {0: "setosa", 1: "versicolor", 2: "virginica"}
            df["target"] = df["target"].map(nombres)
            print(f"  {Fore.GREEN}[OK] {df.shape[0]} instancias, {df.shape[1]} atributos.")
            print(f"  {Fore.CYAN}  target: 'setosa' / 'versicolor' / 'virginica'  (STRING)")
            return df, "iris.arff"
        except Exception as e:
            print(f"  {Fore.RED}[ERROR] No se pudo cargar el demo: {e}")
            return None, None

    # ── Archivo real ──────────────────────────────────
    if not os.path.exists(ruta):
        print(f"  {Fore.RED}[ERROR] Archivo no encontrado: '{ruta}'")
        return None, None

    if not ruta.lower().endswith(".csv"):
        print(f"  {Fore.YELLOW}[WARN] Sin extension .csv — intentando de todas formas...")

    sep_i = input(f"  {Fore.WHITE}Separador (Enter=coma / ';' / 'tab'): {Fore.GREEN}").strip()
    sep   = "\t" if sep_i in ["\\t", "tab"] else (sep_i or ",")

    try:
        df = pd.read_csv(ruta, sep=sep)
        if df.shape[1] == 1:
            print(f"  {Fore.YELLOW}[WARN] Solo 1 columna — separador correcto?")
        print(f"  {Fore.GREEN}[OK] {df.shape[0]} instancias, {df.shape[1]} atributos.")
        return df, os.path.basename(ruta)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] {e}")
        return None, None


# ─── COMPARAR TODOS LOS MÉTODOS ───────────────────────────────────────────────
def comparar_metodos(df, target_col, nombre_archivo, resultados_weka):
    """
    Ejecuta los 4 métodos propios + integra Weka si disponible.
    Genera SCORE_FINAL normalizado y reporte comparativo.
    """
    separador("RUNNING ALL EVALUATORS")

    print(f"\n  {Fore.YELLOW}[1/4] CorrelationAttributeEval...")
    corr = m_correlacion.ejecutar(df, target_col)

    print(f"\n  {Fore.YELLOW}[2/4] Chi2 / F-score...")
    chi  = m_chi2.ejecutar(df, target_col)

    print(f"\n  {Fore.YELLOW}[3/4] RandomForest Gini...")
    rf   = m_rf.ejecutar(df, target_col)

    print(f"\n  {Fore.YELLOW}[4/4] Gower Distance...")
    gwr  = m_gower.ejecutar(df, target_col)

    # ── Unificar índices ──────────────────────────────
    todos_idx = (corr.index
                     .union(chi.index)
                     .union(rf.index)
                     .union(gwr.index))

    resumen = pd.DataFrame(index=todos_idx)
    resumen["Correlacion"]  = normalizar_serie(corr.reindex(todos_idx).fillna(0))
    resumen["Chi2_F"]       = normalizar_serie(chi.reindex(todos_idx).fillna(0))
    resumen["RandomForest"] = normalizar_serie(rf.reindex(todos_idx).fillna(0))
    resumen["Gower"]        = normalizar_serie(gwr.reindex(todos_idx).fillna(0))

    # ── Incluir Weka si disponible ────────────────────
    if resultados_weka and resultados_weka.get("disponible"):
        weka_score = m_weka.consolidar_scores_weka(resultados_weka, todos_idx)
        resumen["Weka_Avg"] = weka_score
        cols_score = ["Correlacion", "Chi2_F", "RandomForest", "Gower", "Weka_Avg"]
        print(f"\n  {Fore.GREEN}[INFO] Weka incluido en SCORE_FINAL (5 metodos).")
    else:
        cols_score = ["Correlacion", "Chi2_F", "RandomForest", "Gower"]
        print(f"\n  {Fore.YELLOW}[INFO] Weka no disponible — SCORE_FINAL con 4 metodos.")

    resumen["SCORE_FINAL"] = resumen[cols_score].mean(axis=1)
    resumen = resumen.sort_values("SCORE_FINAL", ascending=False)

    # ── Reporte comparativo ───────────────────────────
    dataset_info = {
        "nombre":   nombre_archivo,
        "filas":    len(df),
        "columnas": len(df.columns),
    }
    imprimir_reporte_comparativo(resumen, dataset_info, target_col)

    # ── Validacion cruzada vs Weka ────────────────────
    if resultados_weka and resultados_weka.get("disponible"):
        separador("CROSS-VALIDATION: OWN METHODS vs WEKA")
        imprimir_validacion_vs_weka(resumen, resultados_weka, target_col)

    return resumen


# ─── EXPORTAR ─────────────────────────────────────────────────────────────────
def exportar_resultados(resumen: pd.DataFrame):
    separador("EXPORT RESULTS")
    nombre   = input(f"  {Fore.WHITE}Nombre del archivo (sin .csv): {Fore.GREEN}").strip()
    nombre   = nombre if nombre else "resultados_seleccion"
    ruta_out = os.path.join("resultados", f"{nombre}.csv")
    resumen.to_csv(ruta_out)
    print(f"\n  {Fore.GREEN}[OK] Guardado en: {ruta_out}")
    cols = " | ".join(resumen.columns.tolist())
    print(f"  {Fore.CYAN}Columnas: {cols}")


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    limpiar_pantalla()
    banner()

    df              = None
    target_col      = None
    nombre_archivo  = None
    ultimo_resumen  = None
    resultados_weka = None

    while True:
        opcion = menu_principal(weka_ok=m_weka.weka_disponible())

        # ── [1] CARGAR CSV ───────────────────────────────────────────────────
        if opcion == "1":
            df_nuevo, nombre_nuevo = cargar_csv()
            if df_nuevo is not None:
                print(f"\n{Fore.CYAN}  Vista previa (3 filas):")
                print(df_nuevo.head(3).to_string())

                es_valido, _, _ = validar_dataset(df_nuevo)
                if not es_valido:
                    r = input(
                        f"\n  {Fore.RED}Errores criticos. ¿Continuar de todas formas? (s/N): "
                        f"{Fore.GREEN}"
                    ).strip().lower()
                    if r != "s":
                        print(f"  {Fore.YELLOW}[INFO] Carga cancelada.")
                        pausar(); limpiar_pantalla(); banner()
                        continue

                df              = df_nuevo
                nombre_archivo  = nombre_nuevo
                target_col      = elegir_target(df)
                ultimo_resumen  = None
                resultados_weka = None
                if target_col:
                    validar_target(df, target_col)

        # ── [2] CORRELACIÓN ──────────────────────────────────────────────────
        elif opcion == "2":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_correlacion.ejecutar(df, target_col)

        # ── [3] CHI2 ─────────────────────────────────────────────────────────
        elif opcion == "3":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_chi2.ejecutar(df, target_col)

        # ── [4] RANDOM FOREST ────────────────────────────────────────────────
        elif opcion == "4":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_rf.ejecutar(df, target_col)

        # ── [5] GOWER ────1────────────────────────────────────────────────────
        elif opcion == "5":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_gower.ejecutar(df, target_col)

        # ── [6] WEKA ─────────────────────────────────────────────────────────
        elif opcion == "6":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                print(f"\n  {Fore.YELLOW}[INFO] Iniciando evaluadores de Weka...")
                resultados_weka = m_weka.ejecutar(df, target_col)
                if not resultados_weka.get("disponible"):
                    print(f"\n  {Fore.YELLOW}[TIP] Para activar Weka:")
                    print(f"  {Fore.WHITE}    1. Instala Java JDK 8+  →  https://adoptium.net")
                    print(f"  {Fore.WHITE}    2. pip install python-weka-wrapper3 jpype1")
                    print(f"  {Fore.WHITE}    3. Configura JAVA_HOME y reinicia")
        
        # ── [7] IBk ─────────────────────────────────────────────────────────
        elif opcion == "7":
            m_ibk.ejecutar(df, target_col)

        # ── [8] COMPARAR TODOS ───────────────────────────────────────────────
        elif opcion == "8":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                if resultados_weka is None and m_weka.weka_disponible():
                    r = input(
                        f"\n  {Fore.CYAN}¿Ejecutar Weka para cross-validation? (S/n): "
                        f"{Fore.GREEN}"
                    ).strip().lower()
                    if r != "n":
                        resultados_weka = m_weka.ejecutar(df, target_col)

                ultimo_resumen = comparar_metodos(
                    df, target_col, nombre_archivo, resultados_weka
                )

        # ── [8] EXPORTAR ─────────────────────────────────────────────────────
        elif opcion == "8":
            if ultimo_resumen is None:
                print(f"\n  {Fore.RED}[ERROR] Primero ejecuta la comparacion (opcion 7).")
            else:
                exportar_resultados(ultimo_resumen)

        # ── [0] SALIR ────────────────────────────────────────────────────────
        elif opcion == "0":
            m_weka.detener_jvm()
            print(f"\n{Fore.CYAN}  Hasta luego.\n")
            break

        else:
            print(f"\n  {Fore.RED}[ERROR] Opcion no valida (0-8).")

        pausar()
        limpiar_pantalla()
        banner()


if __name__ == "__main__":
    main()