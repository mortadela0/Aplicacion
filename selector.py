"""
selector.py — Punto de entrada CLI.
Ejecutar: python selector.py
"""
import os, sys
import pandas as pd
from colorama import Fore, init
init(autoreset=True)

# ── Verificar dependencias ────────────────────────────────────────────────
def _check(mod):
    try: __import__(mod); return True
    except ImportError: return False

DEPS = {"sklearn":"scikit-learn","pandas":"pandas","numpy":"numpy",
        "colorama":"colorama","matplotlib":"matplotlib","joblib":"joblib"}
faltan = [pip for mod,pip in DEPS.items() if not _check(mod)]
if faltan:
    print("\n[ERROR] Faltan:", " ".join(faltan))
    sys.exit(1)

for d in ["datos","resultados","modelos"]:
    os.makedirs(d, exist_ok=True)

# ── Imports del proyecto ─────────────────────────────────────────────────
from utils.consola        import banner, separador, menu_principal, pausar, elegir_target, limpiar_pantalla
from utils.exportar       import exportar_csv
from utils.entrenador     import evaluar, imprimir_comparacion
from utils.predictor      import guardar_modelo, predecir_nuevo, listar_modelos
from utils.dataset_editor import (menu_eliminar_por_scores,
                                   menu_eliminar_por_score_final,
                                   guardar_dataset)
from validacion           import validar_dataset, validar_target
import metodos.seleccion as sel
from metodos.supervisado   import CATALOGO as CAT_SUP,   ALIAS as ALIAS_SUP,   get_clasificador as get_sup
from metodos.no_supervisado import CATALOGO as CAT_NOSUP, ALIAS as ALIAS_NOSUP, get_clasificador as get_nosup

try:
    from metodos.seleccion.weka_attsel import weka_disponible, detener_jvm
    import metodos.seleccion.weka_attsel as m_weka
    _WEKA = True
except Exception:
    m_weka = None; _WEKA = False
    def weka_disponible(): return False
    def detener_jvm(): pass


# ── Carga de CSV ─────────────────────────────────────────────────────────
def cargar_csv():
    ruta = input(f"\n  {Fore.WHITE}Ruta CSV (Enter=demo Iris): {Fore.GREEN}").strip()
    if not ruta:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df   = iris.frame
        df.columns = [c.replace(" (cm)","").replace(" ","_") for c in df.columns]
        df["target"] = df["target"].map({0:"setosa",1:"versicolor",2:"virginica"})
        print(f"  {Fore.GREEN}[OK] Iris demo: {df.shape[0]} x {df.shape[1]}")
        return df, "iris.csv"
    if not os.path.exists(ruta):
        print(f"  {Fore.RED}[ERROR] No encontrado: {ruta}"); return None, None
    sep_i = input(f"  {Fore.WHITE}Separador (Enter=coma / ';' / 'tab'): {Fore.GREEN}").strip()
    sep   = "\t" if sep_i in ["\\t","tab"] else (sep_i or ",")
    try:
        df = pd.read_csv(ruta, sep=sep)
        print(f"  {Fore.GREEN}[OK] {df.shape[0]} x {df.shape[1]}")
        return df, os.path.basename(ruta)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] {e}"); return None, None


# ── Sub-menú genérico de clasificadores ──────────────────────────────────
def menu_clasificadores(df, target_col, modelos, catalogo, alias, get_fn, titulo):
    separador(titulo)
    print(f"\n  {Fore.WHITE}Clasificadores:")
    for num, (clave, mod) in enumerate(catalogo.items(), 1):
        print(f"  {Fore.CYAN}    [{num}] {clave:<16} — {getattr(mod,'NOMBRE',clave)}")
    print(f"  {Fore.CYAN}    [todos / Enter] — todos")
    print(f"\n  {Fore.WHITE}[a] Entrenar/evaluar   [b] Predecir CSV   [c] Listar modelos   [0] Volver")
    sub = input(f"\n  {Fore.GREEN}> ").strip().lower()

    if sub == "a":
        sel_str = input(f"  {Fore.WHITE}  Clasificadores (num/nombre/todos): {Fore.GREEN}").strip().lower()
        validos = set(catalogo.keys())
        if sel_str in ("","todos","all"):
            modelos_sel = list(catalogo.keys())
        else:
            tokens = sel_str.replace(",","").split()
            modelos_sel = list(dict.fromkeys(
                alias.get(t) or t for t in tokens if alias.get(t) or t in validos
            ))
            if not modelos_sel:
                print(f"  {Fore.YELLOW}  Entrada no reconocida — usando todos.")
                modelos_sel = list(catalogo.keys())

        ts_str = input(f"  {Fore.WHITE}  % test (Enter=20): {Fore.GREEN}").strip()
        try:    test_size = max(0.1, min(0.4, float(ts_str)/100)) if ts_str else 0.2
        except: test_size = 0.2

        params  = {}
        for nom in modelos_sel:
            mod   = catalogo[nom]
            pinfo = getattr(mod,"PARAMS_INFO",{})
            if pinfo:
                print(f"  {Fore.CYAN}  Parámetros para {nom}:")
                p = {}
                for pname, pdesc in pinfo.items():
                    val = input(f"    {Fore.WHITE}{pname} ({pdesc}, Enter=default): {Fore.GREEN}").strip()
                    if val:
                        try:    p[pname] = int(val)
                        except:
                            try:    p[pname] = float(val)
                            except: p[pname] = val
                params[nom] = p

        resultados = {}
        for nom in modelos_sel:
            n_cl   = df[target_col].nunique()
            clf    = get_fn(nom, params.get(nom,{}), n_cl)
            if clf is None:
                print(f"  {Fore.RED}[ERROR] {nom} no reconocido"); continue
            mod     = catalogo[nom]
            display = getattr(mod,"NOMBRE", nom)
            skip_cv = (getattr(mod,"TIPO","") == "no_supervisado" and len(df) > 500)
            res     = evaluar(df, target_col, clf, display, test_size, skip_cv)
            resultados[nom] = res

        imprimir_comparacion(resultados, catalogo)

        # Guardar modelos
        guardar = input(f"  {Fore.WHITE}  ¿Guardar modelos entrenados? (s/N): {Fore.GREEN}").strip().lower()
        if guardar == "s":
            for nom, r in resultados.items():
                guardar_modelo(r["clf"], nom, r["X_columns"], target_col,
                               {"accuracy": r["accuracy"], "f1": r["f1"]})
        modelos.update(resultados)

    elif sub == "b":
        if not modelos:
            print(f"  {Fore.RED}[ERROR] Entrena primero (opcion a).")
        else:
            mejor = max(modelos, key=lambda m: modelos[m]["cv_mean"])
            entry = dict(modelos[mejor], nombre=mejor)
            predecir_nuevo(entry, target_col)

    elif sub == "c":
        pkls = listar_modelos()
        if pkls:
            sel_m = input(f"  {Fore.WHITE}  Número de modelo para predecir (Enter=cancelar): {Fore.GREEN}").strip()
            if sel_m.isdigit():
                idx = int(sel_m)-1
                if 0 <= idx < len(pkls):
                    import joblib
                    entry = joblib.load(os.path.join("modelos", pkls[idx]))
                    predecir_nuevo(entry, entry.get("target_col", target_col))

    return modelos


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    limpiar_pantalla(); banner()
    df              = None
    target_col      = None
    nombre_archivo  = None
    ultimo_resumen  = None
    resultados_weka = None
    modelos_sup     = {}
    modelos_nosup   = {}

    while True:
        opcion = menu_principal(weka_ok=weka_disponible())

        if opcion == "1":
            df_nuevo, nombre_nuevo = cargar_csv()
            if df_nuevo is not None:
                es_valido, _, _ = validar_dataset(df_nuevo)
                if not es_valido:
                    r = input(f"  {Fore.RED}Errores criticos. ¿Continuar? (s/N): {Fore.GREEN}").strip().lower()
                    if r != "s":
                        pausar(); limpiar_pantalla(); banner(); continue
                df              = df_nuevo
                nombre_archivo  = nombre_nuevo
                target_col      = elegir_target(df)
                ultimo_resumen  = None
                resultados_weka = None
                modelos_sup     = {}
                modelos_nosup   = {}
                if target_col:
                    validar_target(df, target_col)

        elif opcion == "2":
            if df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                scores = sel.correlacion(df, target_col)
                df = menu_eliminar_por_scores(df, target_col, scores, "Correlación de Pearson |r|")

        elif opcion == "3":
            if df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                scores = sel.chi2(df, target_col)
                df = menu_eliminar_por_scores(df, target_col, scores, "Chi² / F-score")

        elif opcion == "4":
            if df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                scores = sel.random_forest(df, target_col)
                df = menu_eliminar_por_scores(df, target_col, scores, "Random Forest Gini")

        elif opcion == "5":
            if df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                from metodos.seleccion.gower import configurar_pesos
                pesos_gower = configurar_pesos(df, target_col)
                scores = sel.gower(df, target_col, weights=pesos_gower)
                df = menu_eliminar_por_scores(df, target_col, scores, "Gower Distance")

        elif opcion == "6":
            if not _WEKA:
                print(f"  {Fore.RED}[ERROR] Weka no disponible.")
            elif df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                try:
                    resultados_weka = m_weka.ejecutar(df, target_col)
                except Exception as e:
                    print(f"\n  {Fore.RED}[ERROR] Weka falló: {e}")
                    print(f"  {Fore.YELLOW}  Puedes continuar con las opciones 2-5 y 7 sin Weka.")
                    resultados_weka = {"disponible": False}

        elif opcion == "7":
            if df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                # Weka es opcional: solo se incluye si el usuario ya lo ejecutó (opción 6)
                # ReliefF y GainRatio son nativos — siempre disponibles sin Java
                if resultados_weka is None and weka_disponible():
                    print(f"  {Fore.CYAN}  Weka disponible. Ejecuta la opción [6] primero para"
                          f" incluir Weka_Avg en el SCORE_FINAL.")
                    print(f"  {Fore.CYAN}  Continuando con los 6 métodos nativos "
                          f"(Pearson, Chi², RF, Gower, ReliefF, GainRatio)...")

                try:
                    ultimo_resumen = sel.comparar_todos(
                        df, target_col, nombre_archivo, resultados_weka, m_weka)
                    # Ofrecer eliminar variables por SCORE_FINAL + acuerdo entre métodos
                    if ultimo_resumen is not None:
                        df = menu_eliminar_por_score_final(df, target_col, ultimo_resumen)
                except Exception as e:
                    print(f"\n  {Fore.RED}[ERROR] Error en comparación: {e}")
                    import traceback; traceback.print_exc()

        elif opcion == "8":
            if ultimo_resumen is None:
                print(f"  {Fore.RED}[ERROR] Ejecuta opcion 7 primero.")
            else:
                nombre = input(f"  {Fore.WHITE}Nombre del archivo (sin .csv): {Fore.GREEN}").strip() or "resultados"
                exportar_csv(ultimo_resumen, nombre)

        elif opcion == "9":
            if df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                try:
                    modelos_sup = menu_clasificadores(
                        df, target_col, modelos_sup,
                        CAT_SUP, ALIAS_SUP, get_sup,
                        "CLASIFICADORES SUPERVISADOS — J48 | NB | KNN | SVM | SGD")
                except Exception as e:
                    print(f"\n  {Fore.RED}[ERROR] {e}")
                    import traceback; traceback.print_exc()

        elif opcion == "10":
            if df is None or target_col is None:
                print(f"  {Fore.RED}[ERROR] Carga CSV primero.")
            else:
                try:
                    modelos_nosup = menu_clasificadores(
                        df, target_col, modelos_nosup,
                        CAT_NOSUP, ALIAS_NOSUP, get_nosup,
                        "CLASIFICADORES NO SUPERVISADOS — Hierarchical | KMeans | DBSCAN | GMM | KMedoids")
                except Exception as e:
                    print(f"\n  {Fore.RED}[ERROR] {e}")
                    import traceback; traceback.print_exc()

        elif opcion == "0":
            detener_jvm()
            print(f"\n{Fore.CYAN}  Hasta luego.\n"); break
        else:
            print(f"  {Fore.RED}[ERROR] Opcion no valida (0-10).")

        pausar(); limpiar_pantalla(); banner()

if __name__ == "__main__":
    main()
