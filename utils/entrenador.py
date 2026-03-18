"""
utils/entrenador.py
Lógica de entrenamiento, evaluación y salida estilo Weka.
Usada tanto por el CLI (selector.py) como por la web (web/routes/api.py).
"""
import time
import numpy as np
import pandas as pd
from colorama import Fore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_recall_fscore_support,
                              confusion_matrix, cohen_kappa_score,
                              mean_absolute_error)

ANCHO = 70

def _linea(c="="):
    return c * ANCHO

def _decodificar(labels, encoder):
    if encoder is not None:
        try:
            return [str(encoder.classes_[int(c)]) for c in labels]
        except Exception:
            pass
    return [str(c) for c in labels]

# ── Salida Weka ───────────────────────────────────────────────────────────────

def imprimir_accuracy_by_class(y_test, y_pred, clases, encoder=None):
    labels = list(clases)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    n  = len(labels)
    tp_r, fp_r = [], []
    for i in range(n):
        tp = cm[i,i]; fn = cm[i,:].sum()-tp
        fp = cm[:,i].sum()-tp; tn = cm.sum()-tp-fn-fp
        tp_r.append(tp/(tp+fn+1e-9)); fp_r.append(fp/(fp+tn+1e-9))
    nombres = _decodificar(labels, encoder)
    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  === Accuracy By Class ===")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  {'TP Rate':>8}  {'FP Rate':>8}  {'Precision':>10}  {'Recall':>8}  {'F-Measure':>10}  Class")
    print(f"{Fore.CYAN}  {'─'*62}")
    for i, nom in enumerate(nombres):
        color = Fore.GREEN if f1[i]>=0.8 else Fore.YELLOW if f1[i]>=0.5 else Fore.RED
        print(f"  {color}{tp_r[i]:>8.4f}  {fp_r[i]:>8.4f}  {prec[i]:>10.4f}  {rec[i]:>8.4f}  {f1[i]:>10.4f}  {nom}")
    print(f"{Fore.CYAN}{_linea('=')}")

def imprimir_confusion_matrix(y_test, y_pred, clases, encoder=None):
    labels  = list(clases)
    cm      = confusion_matrix(y_test, y_pred, labels=labels)
    nombres = _decodificar(labels, encoder)
    col_w   = max(max(len(n) for n in nombres), 5) + 2
    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  === Confusion Matrix ===")
    print(f"{Fore.CYAN}{_linea('-')}")
    header = f"  {'':>{col_w}}"
    for n in nombres: header += f"  {n:>{col_w}}"
    print(f"{Fore.WHITE}{header}")
    print(f"{Fore.CYAN}  {'─'*(col_w+(col_w+2)*len(nombres)+4)}")
    for i, (fila, nom) in enumerate(zip(cm, nombres)):
        s = f"  {Fore.WHITE}{nom:>{col_w}}"
        for j, val in enumerate(fila):
            color = Fore.GREEN if i==j and val>0 else Fore.RED if i!=j and val>0 else Fore.WHITE
            s += f"  {color}{val:>{col_w}}"
        print(s + f"  {Fore.WHITE}| {nom}")
    print(f"{Fore.CYAN}{_linea('=')}")

def imprimir_summary(y_test, y_pred, nombre_modelo, elapsed, cv_scores=None):
    acc = accuracy_score(y_test, y_pred)
    n   = len(y_test)
    ok  = int(round(acc*n)); fail = n-ok
    try:    kappa = cohen_kappa_score(y_test, y_pred)
    except: kappa = 0.0
    try:    mae = mean_absolute_error(y_test.astype(float), y_pred.astype(float))
    except: mae = 0.0
    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  === Summary ===  [{Fore.YELLOW}{nombre_modelo}{Fore.WHITE}]")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  {'Correctly Classified Instances':<42} {ok:>5}     {acc*100:>10.4f} %")
    print(f"{Fore.WHITE}  {'Incorrectly Classified Instances':<42} {fail:>5}     {(1-acc)*100:>10.4f} %")
    print(f"{Fore.WHITE}  {'Kappa statistic':<42} {kappa:>19.4f}")
    print(f"{Fore.WHITE}  {'Mean absolute error':<42} {mae:>19.4f}")
    print(f"{Fore.WHITE}  {'Total Number of Instances':<42} {n:>5}")
    if cv_scores is not None:
        print(f"{Fore.CYAN}{_linea('-')}")
        print(f"{Fore.WHITE}  {'Cross-Validation accuracy (5-fold)':<42} {cv_scores.mean():>10.4f} ± {cv_scores.std():.4f}")
    print(f"{Fore.WHITE}  {'Evaluation time':<42} {elapsed:>16.4f}s")
    print(f"{Fore.CYAN}{_linea('=')}\n")

# ── Evaluación principal ──────────────────────────────────────────────────────

def evaluar(df, target_col, clf, nombre_display,
            test_size=0.2, skip_cv=False):
    """
    Entrena clf, imprime salida Weka y retorna dict de métricas.
    clf debe ser un estimator sklearn ya configurado.
    """
    from utils.preprocesamiento import (preparar_datos, target_es_categorico,
                                        _target_encoder, _target_es_string)
    X, y    = preparar_datos(df, target_col)
    encoder = _target_encoder if _target_es_string else None
    X_arr   = X.values
    es_clf  = target_es_categorico(df, target_col)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y, test_size=test_size, random_state=42,
        stratify=y if es_clf else None)

    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Classifier : {Fore.YELLOW}{nombre_display}")
    print(f"{Fore.WHITE}  Train {int((1-test_size)*100)}%  |  Test {int(test_size*100)}%  |  Instancias: {len(df)}")
    print(f"{Fore.CYAN}{_linea('=')}")

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    elapsed = time.time() - t0

    cv_scores = None
    if not skip_cv:
        try:
            cv_scores = cross_val_score(clf, X_arr, y, cv=5,
                                        scoring="accuracy", n_jobs=-1)
        except Exception:
            pass

    clases = np.unique(np.concatenate([y_tr, y_te]))
    imprimir_accuracy_by_class(y_te, y_pred, clases, encoder)
    imprimir_confusion_matrix(y_te, y_pred, clases, encoder)
    imprimir_summary(y_te, y_pred, nombre_display, elapsed, cv_scores)

    acc = accuracy_score(y_te, y_pred)
    f1v = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    return {
        "clf":       clf,
        "X_columns": list(X.columns),
        "accuracy":  acc,
        "f1":        f1v,
        "cv_mean":   float(cv_scores.mean()) if cv_scores is not None else acc,
        "cv_std":    float(cv_scores.std())  if cv_scores is not None else 0.0,
    }

def imprimir_comparacion(resultados, catalogo):
    if len(resultados) < 2:
        return
    mejor = max(resultados, key=lambda m: resultados[m]["cv_mean"])
    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Comparison Summary")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  {'Classifier':<40} {'Accuracy':>10}  {'F1':>10}  {'CV mean':>10}  {'CV±std':>9}")
    print(f"{Fore.CYAN}  {'─'*75}")
    for nom, r in resultados.items():
        display = catalogo.get(nom, {})
        nombre  = getattr(display, "NOMBRE", nom) if hasattr(display,"NOMBRE") else nom
        color   = Fore.GREEN if nom==mejor else Fore.WHITE
        marca   = "  ◄ BEST" if nom==mejor else ""
        print(f"  {color}{nombre:<40} {r['accuracy']:>10.6f}  {r['f1']:>10.6f}"
              f"  {r['cv_mean']:>10.6f}  ±{r['cv_std']:.6f}{Fore.YELLOW}{marca}")
    print(f"{Fore.CYAN}{_linea('=')}\n")
