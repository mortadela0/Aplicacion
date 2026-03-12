"""
╔══════════════════════════════════════════════════════════╗
║   SELECTOR DE VARIABLES v2.2                             ║
║   Correlation | Chi2 | RF | Gower | Weka | Prediccion   ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import sys
import pandas as pd
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)


def _safe_import(mod):
    try:
        __import__(mod); return True
    except ImportError:
        return False


def verificar_dependencias():
    base = {"sklearn": "scikit-learn", "pandas": "pandas",
            "numpy": "numpy", "colorama": "colorama", "matplotlib": "matplotlib"}
    faltan = [pip for mod, pip in base.items() if not _safe_import(mod)]
    if faltan:
        print(f"\n{Fore.RED}[ERROR] Faltan dependencias:")
        for p in faltan:
            print(f"  {Fore.YELLOW}pip install {p}")
        sys.exit(1)


def crear_estructura():
    for d in ["datos", "resultados"]:
        os.makedirs(d, exist_ok=True)


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
import metodos.prediccion    as m_pred

try:
    import metodos.weka_attsel as m_weka
    _WEKA_OK = True
except (ImportError, ModuleNotFoundError):
    m_weka  = None
    _WEKA_OK = False


def _weka_disponible():
    return _WEKA_OK and m_weka is not None and m_weka.weka_disponible()


# ─── CARGA DE CSV ─────────────────────────────────────────────────────────────
def cargar_csv():
    print(f"\n  {Fore.CYAN}Carpeta recomendada: {Fore.WHITE}datos/")
    ruta = input(f"\n  {Fore.WHITE}Ruta del CSV (Enter = demo Iris): {Fore.GREEN}").strip()

    if not ruta:
        print(f"  {Fore.YELLOW}[INFO] Cargando dataset demo (Iris)...")
        try:
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df   = iris.frame
            df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
            nombres = {0: "setosa", 1: "versicolor", 2: "virginica"}
            df["target"] = df["target"].map(nombres)
            print(f"  {Fore.GREEN}[OK] {df.shape[0]} instancias, {df.shape[1]} atributos.")
            return df, "iris.arff"
        except Exception as e:
            print(f"  {Fore.RED}[ERROR] {e}"); return None, None

    if not os.path.exists(ruta):
        print(f"  {Fore.RED}[ERROR] No encontrado: '{ruta}'"); return None, None

    sep_i = input(f"  {Fore.WHITE}Separador (Enter=coma / ';' / 'tab'): {Fore.GREEN}").strip()
    sep   = "\t" if sep_i in ["\\t", "tab"] else (sep_i or ",")
    try:
        df = pd.read_csv(ruta, sep=sep)
        if df.shape[1] == 1:
            print(f"  {Fore.YELLOW}[WARN] Solo 1 columna — separador correcto?")
        print(f"  {Fore.GREEN}[OK] {df.shape[0]} instancias, {df.shape[1]} atributos.")
        return df, os.path.basename(ruta)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] {e}"); return None, None


# ─── COMPARAR TODOS ───────────────────────────────────────────────────────────
def comparar_metodos(df, target_col, nombre_archivo, resultados_weka):
    separador("RUNNING ALL EVALUATORS")

    print(f"\n  {Fore.YELLOW}[1/4] CorrelationAttributeEval...")
    corr = m_correlacion.ejecutar(df, target_col)
    print(f"\n  {Fore.YELLOW}[2/4] Chi2 / F-score...")
    chi  = m_chi2.ejecutar(df, target_col)
    print(f"\n  {Fore.YELLOW}[3/4] RandomForest Gini...")
    rf   = m_rf.ejecutar(df, target_col)
    print(f"\n  {Fore.YELLOW}[4/4] Gower Distance...")
    gwr  = m_gower.ejecutar(df, target_col)

    todos_idx = corr.index.union(chi.index).union(rf.index).union(gwr.index)

    resumen = pd.DataFrame(index=todos_idx)
    resumen["Correlacion"]  = normalizar_serie(corr.reindex(todos_idx).fillna(0))
    resumen["Chi2_F"]       = normalizar_serie(chi.reindex(todos_idx).fillna(0))
    resumen["RandomForest"] = normalizar_serie(rf.reindex(todos_idx).fillna(0))
    resumen["Gower"]        = normalizar_serie(gwr.reindex(todos_idx).fillna(0))

    if resultados_weka and resultados_weka.get("disponible"):
        weka_score = m_weka.consolidar_scores_weka(resultados_weka, todos_idx)
        resumen["Weka_Avg"] = weka_score
        cols_score = ["Correlacion", "Chi2_F", "RandomForest", "Gower", "Weka_Avg"]
        print(f"\n  {Fore.GREEN}[INFO] Weka incluido en SCORE_FINAL (5 metodos).")
    else:
        cols_score = ["Correlacion", "Chi2_F", "RandomForest", "Gower"]
        print(f"\n  {Fore.YELLOW}[INFO] SCORE_FINAL con 4 metodos propios.")

    resumen["SCORE_FINAL"] = resumen[cols_score].mean(axis=1)
    resumen = resumen.sort_values("SCORE_FINAL", ascending=False)

    dataset_info = {"nombre": nombre_archivo, "filas": len(df), "columnas": len(df.columns)}
    imprimir_reporte_comparativo(resumen, dataset_info, target_col)

    if resultados_weka and resultados_weka.get("disponible"):
        separador("CROSS-VALIDATION: OWN METHODS vs WEKA")
        imprimir_validacion_vs_weka(resumen, resultados_weka, target_col)

    return resumen


# ─── EXPORTAR ─────────────────────────────────────────────────────────────────
def exportar_resultados(resumen):
    separador("EXPORT RESULTS")
    nombre   = input(f"  {Fore.WHITE}Nombre (sin .csv): {Fore.GREEN}").strip() or "resultados_seleccion"
    ruta_out = os.path.join("resultados", f"{nombre}.csv")
    resumen.to_csv(ruta_out)
    print(f"\n  {Fore.GREEN}[OK] Guardado: {ruta_out}")


# ─── MENÚ PREDICCIÓN ─────────────────────────────────────────────────────────
def menu_prediccion(df, target_col, ultimo_resumen, modelos_entrenados):
    """Sub-menú de predicción con opciones a / b."""
    separador("PREDICCIÓN")

    # Obtener top features del resumen si existe
    top_features = None
    if ultimo_resumen is not None:
        n = ultimo_resumen["SCORE_FINAL"].gt(0.5).sum()
        n = max(n, 3)
        top_features = ultimo_resumen["SCORE_FINAL"].nlargest(int(n)).index.tolist()
        print(f"  {Fore.CYAN}  Top features (SCORE_FINAL > 0.5): "
              f"{Fore.WHITE}{', '.join(top_features)}")
    else:
        print(f"  {Fore.YELLOW}  [INFO] Sin resumen previo — se usan todas las features.")
        print(f"  {Fore.YELLOW}         Ejecuta opcion [7] primero para usar top features.")

    print(f"\n  {Fore.WHITE}[a]  Evaluar modelos      (train/test split + cross-validation)")
    print(f"  {Fore.WHITE}[b]  Predecir CSV nuevo   (usa el mejor modelo entrenado)")
    print(f"  {Fore.WHITE}[0]  Volver al menu principal")
    sub = input(f"\n  {Fore.GREEN}> ").strip().lower()

    if sub == "a":
        # Pedir K y test_size
        k_str = input(f"  {Fore.WHITE}  K vecinos para KNN (Enter=5): {Fore.GREEN}").strip()
        k = int(k_str) if k_str.isdigit() else 5

        ts_str = input(f"  {Fore.WHITE}  % test (Enter=20): {Fore.GREEN}").strip()
        try:
            test_size = float(ts_str) / 100 if ts_str else 0.2
            test_size = max(0.1, min(0.4, test_size))
        except ValueError:
            test_size = 0.2

        nuevos = m_pred.evaluar(df, target_col,
                                top_features=top_features,
                                test_size=test_size,
                                k_vecinos=k)
        modelos_entrenados.update(nuevos)

    elif sub == "b":
        if not modelos_entrenados:
            print(f"\n  {Fore.RED}[ERROR] Primero evalúa los modelos (opcion 9 → a).")
        else:
            m_pred.predecir_nuevo(modelos_entrenados, df, target_col, top_features)

    elif sub != "0":
        print(f"  {Fore.RED}[ERROR] Opcion no válida.")

    return modelos_entrenados


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    limpiar_pantalla()
    banner()

    df               = None
    target_col       = None
    nombre_archivo   = None
    ultimo_resumen   = None
    resultados_weka  = None
    modelos_entrenados = {}

    while True:
        opcion = menu_principal(weka_ok=_weka_disponible())

        if opcion == "1":
            df_nuevo, nombre_nuevo = cargar_csv()
            if df_nuevo is not None:
                print(f"\n{Fore.CYAN}  Vista previa (3 filas):")
                print(df_nuevo.head(3).to_string())
                es_valido, _, _ = validar_dataset(df_nuevo)
                if not es_valido:
                    r = input(f"\n  {Fore.RED}Errores criticos. ¿Continuar? (s/N): {Fore.GREEN}").strip().lower()
                    if r != "s":
                        pausar(); limpiar_pantalla(); banner(); continue
                df                 = df_nuevo
                nombre_archivo     = nombre_nuevo
                target_col         = elegir_target(df)
                ultimo_resumen     = None
                resultados_weka    = None
                modelos_entrenados = {}
                if target_col:
                    validar_target(df, target_col)

        elif opcion == "2":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_correlacion.ejecutar(df, target_col)

        elif opcion == "3":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_chi2.ejecutar(df, target_col)

        elif opcion == "4":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_rf.ejecutar(df, target_col)

        elif opcion == "5":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                m_gower.ejecutar(df, target_col)

        elif opcion == "6":
            if not _WEKA_OK:
                print(f"\n  {Fore.RED}[ERROR] Modulo weka_attsel.py no encontrado.")
            elif df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                resultados_weka = m_weka.ejecutar(df, target_col)

        elif opcion == "7":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                if resultados_weka is None and _weka_disponible():
                    r = input(f"\n  {Fore.CYAN}¿Ejecutar Weka? (S/n): {Fore.GREEN}").strip().lower()
                    if r != "n":
                        resultados_weka = m_weka.ejecutar(df, target_col)
                ultimo_resumen = comparar_metodos(
                    df, target_col, nombre_archivo, resultados_weka)

        elif opcion == "8":
            if ultimo_resumen is None:
                print(f"\n  {Fore.RED}[ERROR] Primero ejecuta comparacion (opcion 7).")
            else:
                exportar_resultados(ultimo_resumen)

        elif opcion == "9":
            if df is None or target_col is None:
                print(f"\n  {Fore.RED}[ERROR] Primero carga un CSV (opcion 1).")
            else:
                modelos_entrenados = menu_prediccion(
                    df, target_col, ultimo_resumen, modelos_entrenados)

        elif opcion == "0":
            if _WEKA_OK and m_weka:
                m_weka.detener_jvm()
            print(f"\n{Fore.CYAN}  Hasta luego.\n")
            break

        else:
            print(f"\n  {Fore.RED}[ERROR] Opcion no valida (0-9).")

        pausar()
        limpiar_pantalla()
        banner()


if __name__ == "__main__":
    main()
