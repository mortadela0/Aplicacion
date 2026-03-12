"""
metodos/weka_attsel.py
─────────────────────────────────────────────
MÉTODO 5 — Evaluadores nativos de Weka
via python-weka-wrapper3 + JPype1
"""

import os
import time
import tempfile
import pandas as pd
import numpy as np
from colorama import Fore
from utils.consola import separador

# ─── ESTADO DE LA JVM ────────────────────────────────
_WEKA_OK    = False
_JVM_ACTIVA = False

# ─── RUTAS CONOCIDAS DE jvm.dll EN WINDOWS ───────────
_RUTAS_JVM = [
    r"C:\Program Files\Eclipse Adoptium\jdk-25.0.2.10-hotspot\bin\server\jvm.dll",  # <-- tu JDK
    r"C:\Program Files\Java\jdk-17\bin\server\jvm.dll",
    r"C:\Program Files\Java\jdk-17.0.10\bin\server\jvm.dll",
    r"C:\Program Files\Java\jdk-17.0.9\bin\server\jvm.dll",
    r"C:\Program Files\Java\jdk-17.0.8\bin\server\jvm.dll",
    r"C:\Program Files\Java\jdk-21\bin\server\jvm.dll",
    r"C:\Program Files\Eclipse Adoptium\jdk-17\bin\server\jvm.dll",
    r"C:\Program Files\Microsoft\jdk-17\bin\server\jvm.dll",
    r"C:\Program Files\Amazon Corretto\jdk17\bin\server\jvm.dll",
]


def weka_disponible() -> bool:
    global _WEKA_OK, _JVM_ACTIVA
    if not _WEKA_OK:
        iniciar_jvm()
    return _WEKA_OK


def _buscar_jvm_dll() -> str | None:
    """
    Busca jvm.dll en este orden:
      1. JAVA_HOME env var
      2. Rutas conocidas hardcodeadas
      3. Búsqueda automática en C:\\Program Files\\Java
    """
    # 1. Desde JAVA_HOME
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home:
        candidatos = [
            os.path.join(java_home, "bin", "server", "jvm.dll"),
            os.path.join(java_home, "jvm.dll"),
        ]
        for c in candidatos:
            if os.path.exists(c):
                return c

    # 2. Rutas hardcodeadas conocidas
    for ruta in _RUTAS_JVM:
        if os.path.exists(ruta):
            return ruta

    # 3. Búsqueda automática en Program Files
    for base in [r"C:\Program Files\Java", r"C:\Program Files\Eclipse Adoptium",
                 r"C:\Program Files\Microsoft"]:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            if "jvm.dll" in files:
                return os.path.join(root, "jvm.dll")

    return None


def iniciar_jvm() -> bool:
    global _WEKA_OK, _JVM_ACTIVA

    if _JVM_ACTIVA:
        return _WEKA_OK

    # Verificar librerías
    try:
        import jpype
        import weka.core.jvm as jvm
    except ImportError as e:
        print(f"  {Fore.RED}[ERROR] Libreria no instalada: {e}")
        print(f"  {Fore.YELLOW}         pip install python-weka-wrapper3 jpype1")
        _WEKA_OK = False
        return False

    errores = []

    # Buscar jvm.dll y forzar JAVA_HOME ANTES de cualquier intento
    jvm_dll = _buscar_jvm_dll()
    if jvm_dll:
        jvm_dir = os.path.dirname(jvm_dll)
        os.environ["JAVA_HOME"] = jvm_dir
        print(f"  {Fore.YELLOW}[JVM] jvm.dll encontrado: {jvm_dll}")
        print(f"  {Fore.YELLOW}[JVM] JAVA_HOME = {jvm_dir}")
    else:
        print(f"  {Fore.YELLOW}[JVM] jvm.dll no encontrado — intentando con JAVA_HOME del sistema")

    # Estrategia 1: jvm.start normal (ahora con JAVA_HOME seteado)
    try:
        print(f"  {Fore.YELLOW}[JVM] Intentando arranque...")
        jvm.start(packages=False)
        _JVM_ACTIVA = True
        _WEKA_OK    = True
        print(f"  {Fore.GREEN}[JVM] Weka JVM iniciada OK.")
        return True
    except Exception as e:
        errores.append(f"jvm.start: {e}")

    # Estrategia 2: jpype directo con ruta explícita
    if jvm_dll:
        try:
            import jpype
            if not jpype.isJVMStarted():
                print(f"  {Fore.YELLOW}[JVM] Intentando jpype.startJVM directo...")
                jpype.startJVM(jvm_dll, convertStrings=True)
                # Inicializar clases base de Weka manualmente
                jpype.JClass("weka.core.Version")
                jvm._started = True
                _JVM_ACTIVA = True
                _WEKA_OK    = True
                print(f"  {Fore.GREEN}[JVM] Weka JVM iniciada OK (jpype directo).")
                return True
        except Exception as e:
            errores.append(f"jpype directo: {e}")

    # Todas fallaron
    _WEKA_OK = False
    print(f"\n  {Fore.RED}[JVM] No se pudo iniciar la JVM.")
    print(f"  {Fore.RED}  Errores:")
    for e in errores:
        print(f"  {Fore.RED}    • {e}")
    print(f"\n  {Fore.CYAN}  SOLUCION: Corre esto en PowerShell ANTES de python selector.py:")
    print(f"  {Fore.WHITE}  $env:JAVA_HOME = \"C:\\Program Files\\Java\\jdk-17\\bin\\server\"")
    print(f"\n  {Fore.CYAN}  O dile al codigo donde está tu jvm.dll:")
    print(f"  {Fore.WHITE}  Abre metodos/weka_attsel.py y agrega tu ruta")
    print(f"  {Fore.WHITE}  al inicio de la lista _RUTAS_JVM\n")
    return False


def detener_jvm():
    global _JVM_ACTIVA
    if _JVM_ACTIVA:
        try:
            import weka.core.jvm as jvm
            jvm.stop()
            _JVM_ACTIVA = False
        except Exception:
            pass


# ─── CONVERSIÓN DataFrame → ARFF → Weka Instances ────

def df_a_instancias_weka(df: pd.DataFrame, target_col: str):
    from weka.core.converters import Loader
    arff_path = _df_a_arff_temporal(df, target_col)
    loader    = Loader(classname="weka.core.converters.ArffLoader")
    data      = loader.load_file(arff_path)
    data.class_is_last()
    os.unlink(arff_path)
    return data


def _col_es_nominal(serie: pd.Series, col: str, target_col: str) -> bool:
    """
    Decide si una columna debe declararse como NOMINAL (string) en el ARFF.
    Cubre: object, category, StringDtype, y columnas numéricas con strings mezclados.
    """
    if col == target_col:
        return True
    dtype_str = str(serie.dtype).lower()
    if dtype_str in ("object", "category") or dtype_str.startswith("string"):
        return True
    # Columna declarada numérica pero puede tener strings mezclados
    try:
        pd.to_numeric(serie.dropna().iloc[:10], errors="raise")
        return False
    except (ValueError, TypeError):
        return True


def _df_a_arff_temporal(df: pd.DataFrame, target_col: str) -> str:
    cols_feat  = [c for c in df.columns if c != target_col]
    df_ordered = df[cols_feat + [target_col]].copy()
    lines      = ["@relation dataset\n\n"]

    # Pre-calcular si cada columna es nominal o numérica
    es_nominal = {col: _col_es_nominal(df_ordered[col], col, target_col)
                  for col in df_ordered.columns}

    for col in df_ordered.columns:
        name = _arff_name(col)
        if es_nominal[col]:
            vals     = df_ordered[col].dropna().astype(str).unique()
            vals_str = ",".join(f"\'{v}\'" for v in sorted(vals))
            lines.append(f"@attribute {name} {{{vals_str}}}\n")
        else:
            lines.append(f"@attribute {name} numeric\n")

    lines.append("\n@data\n")
    for _, row in df_ordered.iterrows():
        vals = []
        for col in df_ordered.columns:
            v = row[col]
            if pd.isna(v):
                vals.append("?")
            elif es_nominal[col]:
                # Escapar comillas simples dentro del valor
                v_str = str(v).replace("'", "\\'")
                vals.append(f"\'{v_str}\'")
            else:
                vals.append(str(v))
        lines.append(",".join(vals) + "\n")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".arff", delete=False, encoding="utf-8")
    tmp.writelines(lines)
    tmp.close()
    return tmp.name


def _arff_name(col: str) -> str:
    safe = col.replace(" ", "_").replace("'", "").replace('"', "")
    return f"'{safe}'" if any(c in safe for c in ",-{}%") else safe


# ─── EVALUADORES NATIVOS ──────────────────────────────

def _ejecutar_evaluador_ranking(data, classname: str, options: list = None) -> dict:
    from weka.attribute_selection import ASEvaluation, AttributeSelection, ASSearch
    evaluator = ASEvaluation(classname=classname, options=options or [])
    search    = ASSearch(classname="weka.attributeSelection.Ranker",
                         options=["-T", "-1.7976931348623157E308", "-N", "-1"])
    attsel = AttributeSelection()
    attsel.evaluator(evaluator)
    attsel.search(search)
    attsel.select_attributes(data)
    return {data.attribute(int(i)).name: float(m)
            for i, m in zip(attsel.ranked_attributes[:, 0],
                            attsel.ranked_attributes[:, 1])}


def _ejecutar_cfs_bestfirst(data):
    from weka.attribute_selection import ASEvaluation, AttributeSelection, ASSearch
    evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval",
                             options=["-P", "1", "-E", "1"])
    search    = ASSearch(classname="weka.attributeSelection.BestFirst",
                         options=["-D", "1", "-N", "5"])
    attsel = AttributeSelection()
    attsel.evaluator(evaluator)
    attsel.search(search)
    attsel.select_attributes(data)
    selected = [data.attribute(int(i)).name for i in attsel.selected_attributes
                if int(i) != data.class_index]
    return selected, attsel.results_string


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────

def ejecutar(df: pd.DataFrame, target_col: str) -> dict:
    separador("WEKA NATIVE EVALUATORS")

    if not iniciar_jvm():
        return {"disponible": False}

    try:
        print(f"  {Fore.YELLOW}[INFO] Convirtiendo DataFrame a Weka Instances...")
        data   = df_a_instancias_weka(df, target_col)
        n_attr = data.num_attributes - 1
        print(f"  {Fore.GREEN}[OK] {data.num_instances} instancias, {n_attr} atributos.\n")

        resultados = {"disponible": True}

        print(f"  {Fore.YELLOW}[1/5] InfoGainAttributeEval...")
        t0 = time.time()
        ig = _ejecutar_evaluador_ranking(data, "weka.attributeSelection.InfoGainAttributeEval")
        resultados["InfoGain"] = pd.Series(ig).sort_values(ascending=False)
        _imprimir_ranking_weka(resultados["InfoGain"], "InfoGainAttributeEval", "Info Gain", time.time()-t0)

        print(f"  {Fore.YELLOW}[2/5] GainRatioAttributeEval...")
        t0 = time.time()
        gr = _ejecutar_evaluador_ranking(data, "weka.attributeSelection.GainRatioAttributeEval")
        resultados["GainRatio"] = pd.Series(gr).sort_values(ascending=False)
        _imprimir_ranking_weka(resultados["GainRatio"], "GainRatioAttributeEval", "Gain Ratio", time.time()-t0)

        print(f"  {Fore.YELLOW}[3/5] CorrelationAttributeEval...")
        t0 = time.time()
        cr = _ejecutar_evaluador_ranking(data, "weka.attributeSelection.CorrelationAttributeEval")
        resultados["Correlation"] = pd.Series(cr).sort_values(ascending=False)
        _imprimir_ranking_weka(resultados["Correlation"], "CorrelationAttributeEval", "Pearson r", time.time()-t0)

        print(f"  {Fore.YELLOW}[4/5] ReliefFAttributeEval...")
        t0 = time.time()
        rf = _ejecutar_evaluador_ranking(data, "weka.attributeSelection.ReliefFAttributeEval",
                                         options=["-M", "-1", "-D", "1", "-K", "10"])
        resultados["ReliefF"] = pd.Series(rf).sort_values(ascending=False)
        _imprimir_ranking_weka(resultados["ReliefF"], "ReliefFAttributeEval", "ReliefF W.", time.time()-t0)

        print(f"  {Fore.YELLOW}[5/5] CfsSubsetEval + BestFirst...")
        t0 = time.time()
        cfs_selected, cfs_report = _ejecutar_cfs_bestfirst(data)
        resultados["CFS_subset"] = cfs_selected
        resultados["CFS_report"] = cfs_report
        elapsed_cfs = time.time() - t0

        print(f"\n{Fore.CYAN}{'='*65}")
        print(f"{Fore.WHITE}  CfsSubsetEval + BestFirst — Weka output:")
        print(f"{Fore.CYAN}{'─'*65}")
        for line in cfs_report.split("\n"):
            print(f"  {Fore.WHITE}{line}")
        print(f"{Fore.CYAN}{'='*65}")
        print(f"{Fore.WHITE}  Tiempo: {Fore.YELLOW}{elapsed_cfs:.3f}s\n")

        return resultados

    except Exception as e:
        import traceback
        print(f"\n  {Fore.RED}[ERROR] Weka evaluation failed: {e}")
        traceback.print_exc()
        return {"disponible": False}


def _imprimir_ranking_weka(scores: pd.Series, evaluador: str, nombre_col: str, elapsed: float):
    from utils.reporte_weka import imprimir_reporte
    imprimir_reporte(
        scores       = scores,
        metodo       = evaluador,
        nombre_score = nombre_col,
        dataset_info = {"nombre": "dataset", "filas": "?", "columnas": len(scores)},
        target_col   = "(class)",
        elapsed      = elapsed,
        decimales    = 10,
        umbral_pct   = 50.0,
    )


def consolidar_scores_weka(resultados: dict, todos_idx) -> pd.Series:
    from utils.preprocesamiento import normalizar_serie
    if not resultados.get("disponible"):
        return pd.Series(0.0, index=todos_idx)
    capas = []
    for key in ["InfoGain", "GainRatio", "Correlation", "ReliefF"]:
        if key in resultados:
            s = normalizar_serie(resultados[key].reindex(todos_idx).fillna(0))
            capas.append(s)
    if not capas:
        return pd.Series(0.0, index=todos_idx)
    return pd.concat(capas, axis=1).mean(axis=1)


# ─── AUTO-ARRANQUE AL IMPORTAR ───────────────────────
try:
    iniciar_jvm()
except Exception:
    pass
