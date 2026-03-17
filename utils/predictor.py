"""
utils/predictor.py
Persistencia de modelos en modelos/ y prediccion sobre CSV nuevo.
"""
import os
import time
import numpy as np
import pandas as pd
import joblib
from colorama import Fore

MODELOS_DIR = "modelos"

def guardar_modelo(clf, nombre, X_columns, target_col, metadata=None):
    """Guarda clf + metadata en modelos/{nombre}_{timestamp}.pkl"""
    os.makedirs(MODELOS_DIR, exist_ok=True)
    ts       = int(time.time())
    filename = f"{nombre}_{ts}.pkl"
    ruta     = os.path.join(MODELOS_DIR, filename)
    payload  = {
        "clf":       clf,
        "X_columns": X_columns,
        "target_col": target_col,
        "nombre":    nombre,
        "timestamp": ts,
        "metadata":  metadata or {},
    }
    joblib.dump(payload, ruta)
    print(f"  \033[32m[OK]\033[0m Modelo guardado: {ruta}")
    return ruta

def cargar_modelo(ruta):
    """Carga un modelo guardado con guardar_modelo()."""
    if not os.path.exists(ruta):
        print(f"  \033[31m[ERROR]\033[0m No encontrado: {ruta}")
        return None
    return joblib.load(ruta)

def listar_modelos():
    """Lista archivos .pkl en modelos/ con metadata basica."""
    os.makedirs(MODELOS_DIR, exist_ok=True)
    pkls = sorted([f for f in os.listdir(MODELOS_DIR) if f.endswith(".pkl")])
    if not pkls:
        print(f"  \033[33m[INFO]\033[0m No hay modelos guardados en {MODELOS_DIR}/")
        return []
    print(f"\n  Modelos disponibles en {MODELOS_DIR}/:")
    for i, fn in enumerate(pkls, 1):
        try:
            p = joblib.load(os.path.join(MODELOS_DIR, fn))
            print(f"  [{i}] {fn}  —  {p.get('nombre','?')}  target={p.get('target_col','?')}  cols={len(p.get('X_columns',[]))}")
        except Exception:
            print(f"  [{i}] {fn}  —  (no se pudo leer)")
    return pkls

def predecir_nuevo(entry, target_col):
    """
    entry: dict con keys clf, X_columns (de guardar_modelo o de evaluar)
    Pide ruta de CSV, predice y exporta a resultados/.
    """
    from utils.preprocesamiento import preparar_datos, _target_encoder, _target_es_string

    clf    = entry["clf"]
    X_cols = entry["X_columns"]
    nombre = entry.get("nombre", "modelo")

    ruta = input(f"  {Fore.WHITE}  Ruta CSV nuevo (sin etiqueta): {Fore.GREEN}").strip()
    if not os.path.exists(ruta):
        print(f"  {Fore.RED}[ERROR] No encontrado: {ruta}"); return

    sep_i = input(f"  {Fore.WHITE}  Separador (Enter=coma / ';' / 'tab'): {Fore.GREEN}").strip()
    sep   = "\t" if sep_i in ["\\t","tab"] else (sep_i or ",")

    try:
        df_nuevo = pd.read_csv(ruta, sep=sep)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] {e}"); return

    df_tmp = df_nuevo.copy()
    if target_col not in df_tmp.columns:
        df_tmp[target_col] = 0

    try:
        X_nuevo, _, _ = _prep_X(df_tmp, target_col, X_cols)
        y_pred        = clf.predict(X_nuevo)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] Prediccion: {e}"); return

    if _target_es_string:
        try:
            y_labels = _target_encoder.inverse_transform(y_pred.astype(int))
        except Exception:
            y_labels = y_pred
    else:
        y_labels = y_pred

    os.makedirs("resultados", exist_ok=True)
    base     = os.path.splitext(os.path.basename(ruta))[0]
    ruta_out = os.path.join("resultados", f"pred_{nombre}_{base}.csv")
    df_out   = df_nuevo.copy()
    df_out[f"PRED_{target_col}"] = y_labels
    df_out.to_csv(ruta_out, index=False)

    print(f"\n  Primeras 10 predicciones:")
    for i, v in enumerate(y_labels[:10], 1):
        print(f"  {Fore.GREEN}  {i:3d}.  {v}")
    if len(y_labels) > 10:
        print(f"  {Fore.WHITE}  ... y {len(y_labels)-10} más")

    unique, counts = np.unique(y_labels, return_counts=True)
    print(f"\n  Distribución:")
    for u, c in zip(unique, counts):
        pct = c/len(y_labels)*100
        print(f"  {Fore.CYAN}  {str(u):<22} {c:>5}  ({pct:5.1f}%)  {'█'*int(pct/3)}")
    print(f"\n  {Fore.GREEN}Guardado: {ruta_out}")

def _prep_X(df_tmp, target_col, X_cols):
    from utils.preprocesamiento import preparar_datos
    X, y = preparar_datos(df_tmp, target_col)
    return X.reindex(columns=X_cols, fill_value=0).values, y, None
