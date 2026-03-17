"""
metodos/seleccion/weka_attsel.py
Evaluadores nativos de Weka via python-weka-wrapper3.
Requiere Java JDK 8+ y: pip install python-weka-wrapper3 jpype1
"""
import os, time, tempfile
import pandas as pd
import numpy as np
from colorama import Fore

_WEKA_OK    = False
_JVM_ACTIVA = False
_RUTAS_JVM  = [
    r"C:\Program Files\Eclipse Adoptium\jdk-25.0.2.10-hotspot\bin\server\jvm.dll",
    r"C:\Program Files\Java\jdk-17\bin\server\jvm.dll",
    r"C:\Program Files\Java\jdk-21\bin\server\jvm.dll",
]

def weka_disponible():
    global _WEKA_OK
    if not _WEKA_OK:
        iniciar_jvm()
    return _WEKA_OK

def iniciar_jvm():
    global _WEKA_OK, _JVM_ACTIVA
    if _JVM_ACTIVA:
        return _WEKA_OK
    try:
        import jpype
        import weka.core.jvm as jvm

        # JVM ya corriendo (import anterior o llamada previa via JPype)
        if jpype.isJVMStarted():
            _JVM_ACTIVA = True
            _WEKA_OK    = True
            return _WEKA_OK

        jvm.start(packages=False)
        _JVM_ACTIVA = True
        _WEKA_OK    = True
        print(f"  {Fore.GREEN}[JVM] Weka iniciada OK.")
    except Exception as e:
        msg = str(e).lower()
        if "already started" in msg:
            # JPype lanza esto si otra ruta ya inicio la JVM
            _JVM_ACTIVA = True
            _WEKA_OK    = True
        else:
            _WEKA_OK = False
            print(f"  {Fore.RED}[JVM] No disponible: {e}")
    return _WEKA_OK

def detener_jvm():
    global _JVM_ACTIVA
    if _JVM_ACTIVA:
        try:
            import weka.core.jvm as jvm
            jvm.stop()
            _JVM_ACTIVA = False
        except Exception:
            pass

def ejecutar(df, target_col):
    if not iniciar_jvm():
        return {"disponible": False}
    from weka.core.converters import Loader
    from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
    try:
        arff = _df_a_arff(df, target_col)
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data   = loader.load_file(arff)
        data.class_is_last()
        os.unlink(arff)
        res = {"disponible": True}
        for key, cls in [("InfoGain","weka.attributeSelection.InfoGainAttributeEval"),
                          ("GainRatio","weka.attributeSelection.GainRatioAttributeEval"),
                          ("Correlation","weka.attributeSelection.CorrelationAttributeEval"),
                          ("ReliefF","weka.attributeSelection.ReliefFAttributeEval")]:
            ev = ASEvaluation(classname=cls)
            sr = ASSearch(classname="weka.attributeSelection.Ranker",
                          options=["-T","-1.7976931348623157E308","-N","-1"])
            at = AttributeSelection()
            at.evaluator(ev); at.search(sr)
            at.select_attributes(data)
            res[key] = pd.Series({data.attribute(int(i)).name: float(m)
                for i, m in zip(at.ranked_attributes[:,0], at.ranked_attributes[:,1])
            }).sort_values(ascending=False)
        return res
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] Weka: {e}")
        return {"disponible": False}

def consolidar_scores_weka(res, idx):
    from utils.preprocesamiento import normalizar_serie
    if not res.get("disponible"):
        return pd.Series(0.0, index=idx)
    capas = [normalizar_serie(res[k].reindex(idx).fillna(0))
             for k in ["InfoGain","GainRatio","Correlation","ReliefF"] if k in res]
    return pd.concat(capas, axis=1).mean(axis=1) if capas else pd.Series(0.0, index=idx)

def _df_a_arff(df, target_col):
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df2  = df[cols].copy()
    lines = ["@relation dataset\n"]
    for col in df2.columns:
        if col == target_col or df2[col].dtype == object:
            vals = ",".join(f"\'{v}\'" for v in sorted(df2[col].dropna().astype(str).unique()))
            lines.append(f"@attribute {col} {{{vals}}}\n")
        else:
            lines.append(f"@attribute {col} numeric\n")
    lines.append("\n@data\n")
    for _, row in df2.iterrows():
        vals = []
        for col in df2.columns:
            v = row[col]
            if pd.isna(v): vals.append("?")
            elif df2[col].dtype == object: vals.append(f"\'{str(v)}\'")
            else: vals.append(str(v))
        lines.append(",".join(vals) + "\n")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".arff", delete=False, encoding="utf-8")
    tmp.writelines(lines); tmp.close()
    return tmp.name

try:
    iniciar_jvm()
except Exception:
    pass
