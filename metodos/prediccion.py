"""
metodos/prediccion.py
─────────────────────────────────────────────
MÓDULO DE PREDICCIÓN — KNN + Regresión Logística/Lineal

Flujos:
  evaluar()        — train/test split 80/20 + cross-validation 5-fold
                     métricas completas estilo Weka
  predecir_nuevo() — carga CSV sin etiqueta y exporta predicciones

Detecta automáticamente:
  Clasificación  → KNN Classifier + Logistic Regression
  Regresión      → KNN Regressor  + Linear Regression
"""

import time
import os
import pandas as pd
import numpy as np
from colorama import Fore, init

init(autoreset=True)

ANCHO = 70


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _linea(c="="):
    return c * ANCHO


def _fmt(v):
    if not isinstance(v, (int, float, np.floating, np.integer)):
        return str(v)
    if np.isnan(v):
        return "NaN"
    s = f"{float(v):.10f}".rstrip("0").rstrip(".")
    if "." in s and len(s.split(".")[1]) < 4:
        s = f"{float(v):.4f}"
    return s


# ─── PREPARACIÓN ─────────────────────────────────────────────────────────────

def _preparar(df, target_col, top_features=None):
    from utils.preprocesamiento import preparar_datos, target_es_categorico
    X, y = preparar_datos(df, target_col)
    es_clf = target_es_categorico(df, target_col)
    if top_features:
        cols_ok = [c for c in top_features if c in X.columns]
        if cols_ok:
            X = X[cols_ok]
            print(f"  {Fore.CYAN}  Top features usadas ({len(cols_ok)}): "
                  f"{Fore.WHITE}{', '.join(cols_ok)}")
        else:
            print(f"  {Fore.YELLOW}  [WARN] Top features no coinciden — usando todas.")
    return X, y, es_clf


# ─── MODELOS ─────────────────────────────────────────────────────────────────

def _get_modelos(es_clf, k):
    from sklearn.neighbors    import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline      import Pipeline

    if es_clf:
        return {
            "KNN_Classifier": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  KNeighborsClassifier(n_neighbors=k))
            ]),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  LogisticRegression(max_iter=1000, random_state=42))
            ]),
        }
    else:
        return {
            "KNN_Regressor": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  KNeighborsRegressor(n_neighbors=k))
            ]),
            "LinearRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  LinearRegression())
            ]),
        }


# ─── EVALUACIÓN TRAIN/TEST ───────────────────────────────────────────────────

def evaluar(df, target_col, top_features=None, test_size=0.2, k_vecinos=5):
    """
    Entrena y evalúa modelos con split 80/20 + cross-validation 5-fold.
    Retorna dict {nombre_modelo: {modelo, métricas, ...}}
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report,
        r2_score, mean_absolute_error, mean_squared_error
    )

    X, y, es_clf = _preparar(df, target_col, top_features)

    print(f"\n{Fore.CYAN}{_linea()}")
    print(f"{Fore.WHITE}  Model Evaluation  —  "
          f"{'Classification' if es_clf else 'Regression'}")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  Target     : {target_col}")
    print(f"{Fore.WHITE}  Features   : {X.shape[1]}")
    print(f"{Fore.WHITE}  Instances  : {len(df)}")
    print(f"{Fore.WHITE}  Split      : {int((1-test_size)*100)}% train / "
          f"{int(test_size*100)}% test")
    print(f"{Fore.WHITE}  K vecinos  : {k_vecinos}")
    print(f"{Fore.CYAN}{_linea()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if es_clf else None
    )

    modelos    = _get_modelos(es_clf, k_vecinos)
    resultados = {}

    for nombre, pipeline in modelos.items():

        print(f"\n{Fore.CYAN}{_linea('-')}")
        print(f"{Fore.WHITE}  Evaluator : {Fore.YELLOW}{nombre}")
        print(f"{Fore.CYAN}{_linea('-')}")

        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        y_pred  = pipeline.predict(X_te)
        elapsed = time.time() - t0

        cv_metric = "accuracy" if es_clf else "r2"
        cv_scores = cross_val_score(pipeline, X, y, cv=5,
                                    scoring=cv_metric, n_jobs=-1)

        if es_clf:
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            cm  = confusion_matrix(y_te, y_pred)
            cr  = classification_report(y_te, y_pred,
                                        zero_division=0, output_dict=True)

            print(f"{Fore.WHITE}  Accuracy       : {Fore.GREEN}{_fmt(acc)}")
            print(f"{Fore.WHITE}  F1 (weighted)  : {Fore.GREEN}{_fmt(f1)}")
            print(f"{Fore.WHITE}  CV-5 accuracy  : {Fore.CYAN}"
                  f"{_fmt(cv_scores.mean())}  ±  {_fmt(cv_scores.std())}")
            print(f"{Fore.WHITE}  Tiempo         : {Fore.YELLOW}{elapsed:.4f}s")

            _print_confusion(cm, y_te, y_pred)
            _print_clf_report(cr)

            resultados[nombre] = dict(
                pipeline=pipeline, accuracy=acc, f1=f1,
                cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                tipo="clasificacion"
            )

        else:
            r2   = r2_score(y_te, y_pred)
            mae  = mean_absolute_error(y_te, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

            print(f"{Fore.WHITE}  R²             : {Fore.GREEN}{_fmt(r2)}")
            print(f"{Fore.WHITE}  MAE            : {Fore.GREEN}{_fmt(mae)}")
            print(f"{Fore.WHITE}  RMSE           : {Fore.GREEN}{_fmt(rmse)}")
            print(f"{Fore.WHITE}  CV-5 R²        : {Fore.CYAN}"
                  f"{_fmt(cv_scores.mean())}  ±  {_fmt(cv_scores.std())}")
            print(f"{Fore.WHITE}  Tiempo         : {Fore.YELLOW}{elapsed:.4f}s")

            resultados[nombre] = dict(
                pipeline=pipeline, r2=r2, mae=mae, rmse=rmse,
                cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                tipo="regresion"
            )

    _print_resumen(resultados, es_clf)
    return resultados


# ─── REPORTES ────────────────────────────────────────────────────────────────

def _print_confusion(cm, y_te, y_pred):
    from utils.preprocesamiento import _target_encoder, _target_es_string

    print(f"\n{Fore.WHITE}  Confusion Matrix:")
    print(f"{Fore.CYAN}  {_linea('-')}")

    clases = sorted(set(list(y_te) + list(y_pred)))

    if _target_es_string:
        try:
            etiquetas = [str(_target_encoder.classes_[int(c)]) for c in clases]
        except Exception:
            etiquetas = [str(c) for c in clases]
    else:
        etiquetas = [str(c) for c in clases]

    max_lbl = max(len(e) for e in etiquetas)
    col_w   = max(max_lbl, 5)

    header = f"  {Fore.WHITE}  {'':>{max_lbl}}"
    for e in etiquetas:
        header += f"  {e:>{col_w}}"
    print(header)
    print(f"  {Fore.CYAN}  {'─' * (max_lbl + (col_w + 2) * len(clases) + 2)}")

    for i, (fila, etiq) in enumerate(zip(cm, etiquetas)):
        fila_str = f"  {Fore.WHITE}  {etiq:>{max_lbl}}"
        for j, val in enumerate(fila):
            color = Fore.GREEN if i == j and val > 0 else \
                    Fore.RED   if i != j and val > 0 else Fore.WHITE
            fila_str += f"  {color}{val:>{col_w}}"
        fila_str += f"  {Fore.WHITE}| {etiq}"
        print(fila_str)

    print(f"  {Fore.CYAN}  {_linea('-')}")


def _print_clf_report(cr):
    print(f"\n{Fore.WHITE}  Per-class metrics:")
    print(f"{Fore.CYAN}  {_linea('-')}")
    print(f"{Fore.WHITE}  {'Class':<22} {'Precision':>10} {'Recall':>10} "
          f"{'F1-score':>10} {'Support':>9}")
    print(f"{Fore.CYAN}  {'─'*62}")

    for clase, vals in cr.items():
        if clase in {"accuracy", "macro avg", "weighted avg"} \
                or not isinstance(vals, dict):
            continue
        f1v = vals.get("f1-score", 0)
        color = Fore.GREEN if f1v >= 0.8 else \
                Fore.YELLOW if f1v >= 0.5 else Fore.RED
        print(f"  {color}  {str(clase):<22} "
              f"{_fmt(vals['precision']):>10} "
              f"{_fmt(vals['recall']):>10} "
              f"{_fmt(vals['f1-score']):>10} "
              f"{int(vals['support']):>9}")

    if "weighted avg" in cr:
        w = cr["weighted avg"]
        print(f"{Fore.CYAN}  {'─'*62}")
        print(f"{Fore.WHITE}  {'Weighted avg':<22} "
              f"{_fmt(w['precision']):>10} "
              f"{_fmt(w['recall']):>10} "
              f"{_fmt(w['f1-score']):>10}")

    print(f"  {Fore.CYAN}{_linea('-')}")


def _print_resumen(resultados, es_clf):
    print(f"\n{Fore.CYAN}{_linea()}")
    print(f"{Fore.WHITE}  Model Comparison Summary")
    print(f"{Fore.CYAN}{_linea('-')}")

    mejor = max(resultados, key=lambda m: resultados[m]["cv_mean"])

    if es_clf:
        print(f"{Fore.WHITE}  {'Model':<25} {'Accuracy':>12} {'F1':>12} "
              f"{'CV-5 mean':>12} {'CV±std':>10}")
        print(f"{Fore.CYAN}  {'─'*72}")
        for nombre, r in resultados.items():
            color = Fore.GREEN if nombre == mejor else Fore.WHITE
            marca = "  ◄ BEST" if nombre == mejor else ""
            print(f"  {color}  {nombre:<25} "
                  f"{_fmt(r['accuracy']):>12} "
                  f"{_fmt(r['f1']):>12} "
                  f"{_fmt(r['cv_mean']):>12} "
                  f"±{_fmt(r['cv_std']):>9}"
                  f"{Fore.YELLOW}{marca}")
    else:
        print(f"{Fore.WHITE}  {'Model':<25} {'R²':>12} {'MAE':>12} "
              f"{'RMSE':>12} {'CV-5 R²':>12}")
        print(f"{Fore.CYAN}  {'─'*72}")
        for nombre, r in resultados.items():
            color = Fore.GREEN if nombre == mejor else Fore.WHITE
            marca = "  ◄ BEST" if nombre == mejor else ""
            print(f"  {color}  {nombre:<25} "
                  f"{_fmt(r['r2']):>12} "
                  f"{_fmt(r['mae']):>12} "
                  f"{_fmt(r['rmse']):>12} "
                  f"{_fmt(r['cv_mean']):>12}"
                  f"{Fore.YELLOW}{marca}")

    print(f"{Fore.CYAN}{_linea()}\n")


# ─── PREDICCIÓN SOBRE CSV NUEVO ───────────────────────────────────────────────

def predecir_nuevo(resultados_modelos: dict, df_train: pd.DataFrame,
                   target_col: str, top_features: list = None):
    """
    Carga un CSV nuevo sin etiqueta, usa el mejor modelo entrenado
    y exporta un CSV con la columna PRED_<target> añadida.
    """
    from utils.preprocesamiento import (_target_encoder, _target_es_string,
                                         preparar_datos)

    if not resultados_modelos:
        print(f"  {Fore.RED}[ERROR] No hay modelos entrenados. "
              f"Ejecuta primero la opcion Evaluar (opcion 9a).")
        return

    # Elegir el mejor modelo por CV
    mejor_nombre = max(resultados_modelos,
                       key=lambda m: resultados_modelos[m]["cv_mean"])
    pipeline = resultados_modelos[mejor_nombre]["pipeline"]

    print(f"\n{Fore.CYAN}{_linea()}")
    print(f"{Fore.WHITE}  Predict on new data")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  Modelo seleccionado : {Fore.YELLOW}{mejor_nombre} "
          f"{Fore.WHITE}(mejor CV-5)")
    print(f"{Fore.CYAN}{_linea('-')}")

    ruta = input(f"  {Fore.WHITE}  Ruta del CSV nuevo (sin etiqueta): "
                 f"{Fore.GREEN}").strip()
    if not os.path.exists(ruta):
        print(f"  {Fore.RED}[ERROR] Archivo no encontrado: '{ruta}'")
        return

    sep_i = input(f"  {Fore.WHITE}  Separador (Enter=coma / ';' / 'tab'): "
                  f"{Fore.GREEN}").strip()
    sep   = "\t" if sep_i in ["\\t", "tab"] else (sep_i or ",")

    try:
        df_nuevo = pd.read_csv(ruta, sep=sep)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] {e}")
        return

    print(f"  {Fore.GREEN}[OK] {df_nuevo.shape[0]} instancias, "
          f"{df_nuevo.shape[1]} columnas cargadas.")

    # Preparar features del CSV nuevo con la misma lógica que el train
    df_tmp = df_nuevo.copy()
    if target_col not in df_tmp.columns:
        df_tmp[target_col] = 0   # columna dummy para que preparar_datos funcione

    try:
        X_nuevo, _, _ = _preparar(df_tmp, target_col, top_features)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] Preparando features: {e}")
        return

    # Alinear columnas exactamente con el entrenamiento
    X_train_cols = pipeline[:-1].feature_names_in_ \
        if hasattr(pipeline[:-1], "feature_names_in_") else X_nuevo.columns

    try:
        # Reindexar con las columnas del train (rellena con 0 si falta alguna)
        X_nuevo = X_nuevo.reindex(columns=X_train_cols, fill_value=0)
        y_pred  = pipeline.predict(X_nuevo)
    except Exception as e:
        print(f"  {Fore.RED}[ERROR] Prediccion fallida: {e}")
        print(f"  {Fore.YELLOW}  Asegurate de que el CSV nuevo tiene las mismas "
              f"columnas que el dataset de entrenamiento.")
        return

    # Decodificar etiquetas si el target original era string
    if _target_es_string:
        try:
            y_labels = _target_encoder.inverse_transform(y_pred.astype(int))
        except Exception:
            y_labels = y_pred
    else:
        y_labels = y_pred

    # Exportar
    df_out = df_nuevo.copy()
    df_out[f"PRED_{target_col}"] = y_labels

    os.makedirs("resultados", exist_ok=True)
    nombre_base = os.path.splitext(os.path.basename(ruta))[0]
    ruta_out    = os.path.join("resultados",
                               f"predicciones_{mejor_nombre}_{nombre_base}.csv")
    df_out.to_csv(ruta_out, index=False)

    # Mostrar muestra de predicciones
    print(f"\n{Fore.CYAN}{_linea()}")
    print(f"{Fore.WHITE}  Predictions — first 10 rows:")
    print(f"{Fore.CYAN}{_linea('-')}")
    for i, val in enumerate(y_labels[:10], 1):
        print(f"  {Fore.GREEN}  {i:3d}.  {val}")
    if len(y_labels) > 10:
        print(f"  {Fore.WHITE}  ... y {len(y_labels) - 10} mas")

    # Distribución de predicciones
    unique, counts = np.unique(y_labels, return_counts=True)
    print(f"\n{Fore.WHITE}  Distribución de predicciones:")
    for u, c in zip(unique, counts):
        pct  = c / len(y_labels) * 100
        barra = "█" * int(pct / 3)
        print(f"  {Fore.CYAN}  {str(u):<20} {c:>5}  ({pct:5.1f}%)  {barra}")

    print(f"\n{Fore.WHITE}  Total predichos : {Fore.GREEN}{len(y_labels)}")
    print(f"{Fore.WHITE}  Guardado en     : {Fore.YELLOW}{ruta_out}")
    print(f"{Fore.CYAN}{_linea()}\n")
