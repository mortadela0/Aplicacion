"""
web/routes/api.py
Todas las rutas REST del proyecto ML Studio v3.
"""
import io, base64, traceback, pickle
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, session
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

api_bp = Blueprint("api", __name__)

# ─────────────────────────────────────────────────────────────
# Helpers de sesión
# ─────────────────────────────────────────────────────────────
def get_df():
    raw = session.get("df_pickle")
    return pickle.loads(raw) if raw else None

def set_df(df):
    session["df_pickle"] = pickle.dumps(df)

def get_target():
    return session.get("target_col")

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=110)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def _dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0d1e35")
    ax.tick_params(colors="#a0c4ff", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#1e3a5f")
    return fig, ax

# ─────────────────────────────────────────────────────────────
# CARGAR CSV
# ─────────────────────────────────────────────────────────────
@api_bp.route("/api/cargar", methods=["POST"])
def cargar_dataset():
    try:
        f   = request.files.get("archivo")
        sep = request.form.get("separador", ",")
        if sep == "tab": sep = "\t"
        if f is None:
            return jsonify({"error": "Sin archivo"}), 400
        df = pd.read_csv(f, sep=sep)
        if df.shape[1] <= 1:
            return jsonify({"error": "Solo 1 columna — revisa el separador"}), 400
        set_df(df)
        session.update({
            "target_col": None, "nombre": f.filename,
            "ultimo_resumen": None, "clf_resultados": [],
            "metodos_ejecutados": [], "modelos_entrenados": {}
        })
        columnas = []
        for i, col in enumerate(df.columns):
            nu = int(df[col].nunique())
            dt = str(df[col].dtype)
            if dt == "object":
                tipo, icono = "STRING", "✓"
            elif "int" in dt and nu <= 20:
                tipo, icono = f"INT {nu} clases", "~"
            elif "int" in dt:
                tipo, icono = "INT continuo", "·"
            else:
                tipo, icono = "FLOAT continuo", "·"
            columnas.append({"idx": i, "nombre": col, "tipo": tipo,
                              "icono": icono,
                              "nulos": int(df[col].isnull().sum())})
        return jsonify({
            "ok": True, "nombre": f.filename,
            "filas": len(df), "columnas_count": len(df.columns),
            "columnas": columnas,
            "preview": df.head(5).fillna("").to_dict("records"),
            "headers": list(df.columns)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/set_target", methods=["POST"])
def set_target():
    try:
        df  = get_df()
        col = (request.json or {}).get("columna")
        if df is None:
            return jsonify({"error": "Sin dataset"}), 400
        if col not in df.columns:
            return jsonify({"error": f"Columna '{col}' no existe"}), 400
        session["target_col"] = col
        if df[col].dtype == object:
            print(f"  [AVISO] Target '{col}' es STRING. "
                  f"Se recomienda entero para evitar errores en SGD/SVM.")
        muestra = [str(c) for c in df[col].dropna().unique()[:10]]
        return jsonify({"ok": True, "target": col,
                        "n_clases": int(df[col].nunique()),
                        "muestra": muestra})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# SELECTORES
# ─────────────────────────────────────────────────────────────
@api_bp.route("/api/selector/<metodo>", methods=["POST"])
def ejecutar_selector(metodo):
    try:
        df = get_df(); tc = get_target()
        if df is None or tc is None:
            return jsonify({"error": "Carga dataset y selecciona target"}), 400

        import metodos.seleccion as sel
        body = request.json or {}

        # ── métodos individuales ──────────────────────────────
        if metodo == "correlacion":
            scores = sel.correlacion(df, tc)
        elif metodo == "chi2":
            scores = sel.chi2(df, tc)
        elif metodo == "random_forest":
            scores = sel.random_forest(df, tc)
        elif metodo == "relief":
            from metodos.seleccion.relief import ejecutar as run_r
            scores = run_r(df, tc, n_neighbors=int(body.get("n_neighbors", 10)))
        elif metodo == "gain_ratio":
            from metodos.seleccion.gain_ratio import ejecutar as run_gr
            scores = run_gr(df, tc, n_bins=int(body.get("n_bins", 10)))

        # ── gower especial ────────────────────────────────────
        elif metodo == "gower":
            from metodos.seleccion.gower import _distancia_gower
            max_i   = int(body.get("max_instancias", 20))
            weights = body.get("weights") or None
            if weights:
                weights = {k: float(v) for k, v in weights.items()}
            scores  = sel.gower(df, tc, weights=weights, max_matriz=max_i)
            n_mat   = min(len(df), max_i)
            df_mat  = (df.sample(n=n_mat, random_state=0).reset_index(drop=True)
                       if len(df) > n_mat else df.reset_index(drop=True))
            feat_m  = df_mat.drop(columns=[tc])
            dist_m  = _distancia_gower(feat_m, weights)
            sim_m   = 1.0 - dist_m
            np.fill_diagonal(sim_m, 1.0)
            vecinos = []
            for i in range(n_mat):
                s = sim_m[i].copy(); s[i] = -1
                vi = int(np.argmax(s))
                vecinos.append({"instancia": i, "vecino": vi,
                                "similitud": round(float(sim_m[i, vi]), 8)})
            elim  = _recomendacion_simple(scores)
            graf  = _heatmap(sim_m)
            _track(metodo)
            return jsonify({"ok": True,
                            "resultados": _scores_list(scores),
                            "grafica": graf,
                            "vecinos": vecinos,
                            "eliminacion": elim})

        # ── comparar todos ────────────────────────────────────
        elif metodo == "todos":
            resumen = sel.comparar_todos(
                df, tc, session.get("nombre", "dataset"))
            session["ultimo_resumen"] = resumen.to_json()
            session["metodos_ejecutados"] = list(set(
                session.get("metodos_ejecutados", []) + ["Comparar Todos"]))
            elim = _recomendacion_todos(resumen)
            filas = (resumen.reset_index()
                     .rename(columns={"index": "variable"})
                     .to_dict("records"))
            filas = [{k: (round(float(v), 8) if isinstance(v, float) else v)
                      for k, v in r.items()} for r in filas]
            graf = _barras(filas, "SCORE_FINAL — 6 métodos")
            return jsonify({"ok": True, "resultados": filas,
                            "grafica": graf, "eliminacion": elim})
        else:
            return jsonify({"error": f"Método '{metodo}' no reconocido"}), 400

        elim  = _recomendacion_simple(scores)
        graf  = _barras(_scores_list(scores), metodo)
        _track(metodo)
        return jsonify({"ok": True,
                        "resultados": _scores_list(scores),
                        "grafica": graf,
                        "eliminacion": elim})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _track(nombre):
    session["metodos_ejecutados"] = list(set(
        session.get("metodos_ejecutados", []) + [nombre]))


def _scores_list(scores):
    max_v = float(scores.max()) if scores.max() > 0 else 1.0
    result = []
    for i, (var, val) in enumerate(scores.items()):
        result.append({
            "variable":  var,
            "score":     round(float(val), 10),
            "pct_merit": round(float(val) / max_v * 100, 2),
            "rank":      i + 1,
        })
    return result


def _recomendacion_simple(scores):
    """Para un selector individual."""
    n     = len(scores)
    top_n = min(5, n)
    bot_n = max(2, int(n * 0.30))
    top5  = list(scores.nlargest(top_n).index)
    cand  = list(scores.nsmallest(bot_n).index)
    max_v = float(scores.max()) if scores.max() > 0 else 1.0
    ranking = []
    for i, (var, val) in enumerate(scores.items()):
        ranking.append({
            "variable":    var,
            "score_final": round(float(val), 6),
            "pct_merit":   round(float(val) / max_v * 100, 1),
            "top5":        var in top5,
            "recomendada": var in cand,
            "rank":        i + 1,
        })
    return {"top5": top5, "candidatas": cand, "ranking": ranking}


def _recomendacion_todos(resumen):
    """Para comparar todos: usa SCORE_FINAL + acuerdo entre métodos."""
    cols_m   = [c for c in resumen.columns if c != "SCORE_FINAL"]
    n_met    = len(cols_m)
    features = list(resumen.index)
    n_feat   = len(features)
    top_n    = min(5, n_feat)
    bot_n    = max(2, int(n_feat * 0.30))
    score_f  = resumen["SCORE_FINAL"].sort_values(ascending=False)
    top5     = list(score_f.head(top_n).index)

    # Top N por cada método
    tops = {m: set(resumen[m].nlargest(top_n).index)
            for m in cols_m if m in resumen.columns}

    # Votos para eliminar = cuántos métodos NO lo tienen en su top5
    acuerdo = {}
    for var in features:
        votos_elim = sum(1 for t in tops.values() if var not in t)
        acuerdo[var] = {"votos_eliminar": votos_elim,
                        "total": n_met,
                        "pct": round(votos_elim / n_met * 100, 1)}

    # Candidatas: bottom por SCORE_FINAL + mayoría de votos de eliminación
    bottom = list(score_f.tail(bot_n).index)
    cand = list(dict.fromkeys(
        [v for v in features
         if acuerdo[v]["votos_eliminar"] >= n_met * 0.5] + bottom
    ))

    ranking = []
    for var in features:
        ranking.append({
            "variable":       var,
            "score_final":    round(float(score_f.get(var, 0)), 6),
            "votos_eliminar": acuerdo[var]["votos_eliminar"],
            "total_metodos":  n_met,
            "pct_acuerdo":    acuerdo[var]["pct"],
            "top5":           var in top5,
            "recomendada":    var in cand,
        })
    return {"top5": top5, "candidatas": cand,
            "ranking": ranking, "n_metodos": n_met}


# ─────────────────────────────────────────────────────────────
# ELIMINAR VARIABLES
# ─────────────────────────────────────────────────────────────
@api_bp.route("/api/eliminar_variables", methods=["POST"])
def eliminar_variables():
    try:
        df = get_df(); tc = get_target()
        if df is None:
            return jsonify({"error": "Sin dataset"}), 400
        cols = (request.json or {}).get("variables", [])
        cols = [c for c in cols if c != tc and c in df.columns]
        if not cols:
            return jsonify({"error": "Sin columnas válidas para eliminar"}), 400
        df_new = df.drop(columns=cols)
        set_df(df_new)
        return jsonify({
            "ok": True,
            "eliminadas": cols,
            "columnas": list(df_new.columns),
            "filas": len(df_new),
            "columnas_count": len(df_new.columns),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# CLASIFICADORES
# ─────────────────────────────────────────────────────────────
@api_bp.route("/api/clasificar", methods=["POST"])
def clasificar():
    try:
        df = get_df(); tc = get_target()
        if df is None or tc is None:
            return jsonify({"error": "Carga dataset y selecciona target"}), 400

        data    = request.json or {}
        tipo    = data.get("tipo", "supervisado")
        nombres = data.get("modelos", [])
        test_sz = float(data.get("test_size", 0.2))
        params  = data.get("params", {})
        es_sup  = tipo == "supervisado"

        if es_sup:
            from metodos.supervisado import get_clasificador, CATALOGO
        else:
            from metodos.no_supervisado import get_clasificador, CATALOGO

        from utils.entrenador       import evaluar
        from utils.preprocesamiento import (preparar_datos,
                                            target_es_categorico,
                                            _target_encoder,
                                            _target_es_string)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics         import confusion_matrix

        X, y     = preparar_datos(df, tc)
        X_arr    = X.values
        n_cl     = int(df[tc].nunique())
        es_cat   = target_es_categorico(df, tc)
        indices = np.arange(len(df))
        idx_tr, idx_te = train_test_split(
            indices, test_size=test_sz, random_state=42,
            stratify=y if es_cat else None)
        X_tr, X_te = X_arr[idx_tr], X_arr[idx_te]
        y_tr, y_te = np.array(y)[idx_tr], np.array(y)[idx_te]
        # Guardar índices de entrenamiento para excluirlos en predicción
        session["train_indices"] = idx_tr.tolist()

        resultados_api    = []
        modelos_guardados = session.get("modelos_entrenados", {}) or {}

        for nombre in nombres:
            clf = get_clasificador(nombre, params.get(nombre, {}), n_cl)
            if clf is None:
                continue
            mod     = CATALOGO.get(nombre)
            display = getattr(mod, "NOMBRE", nombre) if mod else nombre
            skip_cv = (not es_sup and len(df) > 500)

            res       = evaluar(df, tc, clf, display, test_sz, skip_cv)
            clf_fit   = res["clf"]
            y_pred    = clf_fit.predict(X_te)
            clases    = sorted(set(list(y_te) + list(y_pred)))
            cm_arr    = confusion_matrix(y_te, y_pred, labels=clases).tolist()

            if _target_es_string:
                try:
                    clases_str = [str(_target_encoder.classes_[int(c)])
                                  for c in clases]
                except Exception:
                    clases_str = [str(c) for c in clases]
            else:
                clases_str = [str(c) for c in clases]

            # Métricas por clase
            abc = _accuracy_by_class(cm_arr, clases_str)

            entry = {
                "modelo":    nombre,
                "display":   display,
                "accuracy":  round(float(res["accuracy"]), 8),
                "f1":        round(float(res["f1"]), 8),
                "cv_mean":   round(float(res["cv_mean"]), 8),
                "cv_std":    round(float(res["cv_std"]), 8),
                "kappa":     round(float(res.get("kappa", 0)), 6),
                "mae":       round(float(res.get("mae", 0)), 6),
                "cm":        cm_arr,
                "clases":    clases_str,
                "abc":       abc,
            }
            resultados_api.append(entry)
            modelos_guardados[nombre] = {
                "clf":       clf_fit,
                "X_columns": list(X.columns),
                "display":   display,
                "tipo":      tipo,
                "clases":    clases_str,
            }

        session["modelos_entrenados"]  = modelos_guardados
        session["clf_resultados"]      = (
            session.get("clf_resultados", []) + resultados_api)
        session["metodos_ejecutados"]  = list(set(
            session.get("metodos_ejecutados", []) +
            [r["modelo"] for r in resultados_api]))

        graf_comp = _barras_clf(resultados_api)
        grafs_cm  = [_grafica_cm(r["cm"], r["clases"], r["display"])
                     for r in resultados_api]

        return jsonify({"ok": True, "resultados": resultados_api,
                        "grafica": graf_comp, "graficas_cm": grafs_cm})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _accuracy_by_class(cm, clases):
    """Calcula TP Rate, FP Rate, Precision, Recall, F-Measure por clase."""
    n = len(clases)
    rows = []
    for i in range(n):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[ii][i] for ii in range(n)) - tp
        tn = sum(sum(cm[ii]) for ii in range(n)) - tp - fp - fn
        tpr  = tp / (tp + fn + 1e-9)
        fpr  = fp / (fp + tn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec  = tpr
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({
            "clase": clases[i],
            "tp_rate":   round(tpr, 4),
            "fp_rate":   round(fpr, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f_measure": round(f1, 4),
            "support":   sum(cm[i]),
        })
    return rows


# ─────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────
@api_bp.route("/api/predecir", methods=["POST"])
def predecir():
    """
    Tres modos:
      manual  → instancia con valores ingresados a mano
      muestra → N instancias aleatorias del dataset (muestra real vs predicho)
      csv     → archivo CSV externo (con o sin target)
    """
    try:
        df = get_df(); tc = get_target()
        if df is None or tc is None:
            return jsonify({"error": "Carga dataset y selecciona target"}), 400

        mods_guard = session.get("modelos_entrenados", {}) or {}
        if not mods_guard:
            return jsonify({"error": "Entrena al menos un clasificador primero"}), 400

        data    = request.json or {}
        modo    = data.get("modo", "manual")
        nom_mod = data.get("modelo", "")
        entry   = mods_guard.get(nom_mod) or list(mods_guard.values())[0]
        clf     = entry["clf"]
        X_cols  = entry["X_columns"]
        display = entry.get("display", nom_mod)

        from utils.preprocesamiento import (preparar_datos,
                                            _target_encoder, _target_es_string)
        from sklearn.metrics import (accuracy_score, f1_score,
                                     confusion_matrix, cohen_kappa_score)

        tiene_real = False

        if modo == "manual":
            inst = data.get("instancia", {})
            row  = {col: _to_float(inst.get(col, 0)) for col in X_cols}
            X_new = pd.DataFrame([row])[X_cols].values
            y_pred  = clf.predict(X_new)
            y_real  = None
            filas   = [row]

        elif modo == "muestra":
            n_m = int(data.get("n_muestras", 10))
            # Excluir instancias usadas en entrenamiento
            train_idx = set(session.get("train_indices", []))
            all_idx   = list(range(len(df)))
            test_pool = [i for i in all_idx if i not in train_idx]
            if not test_pool:
                # Si no hay info de índices de training, usar todo el dataset
                test_pool = all_idx
            n_take = min(n_m, len(test_pool))
            rng    = np.random.default_rng(99)
            chosen = rng.choice(test_pool, n_take, replace=False).tolist()
            df_m   = df.iloc[chosen].reset_index(drop=True)
            X_m, y_real_enc = preparar_datos(df_m, tc)
            X_new  = X_m.reindex(columns=X_cols, fill_value=0).values
            y_pred = clf.predict(X_new)
            y_real = y_real_enc
            tiene_real = True
            filas  = df_m.drop(columns=[tc]).to_dict("records")

        else:
            return jsonify({"error": "Usa /api/predecir_csv para archivos"}), 400

        y_labels = _decode(y_pred)
        probas, clases_str = _get_probas(clf, X_new)
        metricas = None; cm_data = None; grafs_cm = None

        if tiene_real and y_real is not None:
            y_real_str = _decode(y_real)
            acc  = float(accuracy_score(y_real_str, y_labels))
            f1v  = float(f1_score(y_real_str, y_labels,
                                   average="weighted", zero_division=0))
            try:
                kappa = float(cohen_kappa_score(y_real_str, y_labels))
            except Exception:
                kappa = 0.0
            clases_u = sorted(set(y_real_str + y_labels))
            cm_data  = confusion_matrix(
                y_real_str, y_labels, labels=clases_u).tolist()
            abc      = _accuracy_by_class(cm_data, clases_u)
            metricas = {
                "accuracy":  round(acc, 6),
                "f1":        round(f1v, 6),
                "kappa":     round(kappa, 6),
                "clases":    clases_u,
                "abc":       abc,
                "n_total":   len(y_real_str),
                "n_correctas": int(sum(r == p for r, p in
                                       zip(y_real_str, y_labels))),
            }
            grafs_cm = [_grafica_cm(cm_data, clases_u, display)]
            # Tabla real vs predicho
            filas_result = []
            for i, (real, pred) in enumerate(zip(y_real_str, y_labels)):
                r = {"#": i+1, "real": real, "prediccion": pred,
                     "correcto": "✓" if real == pred else "✗"}
                if probas:
                    for j, cls in enumerate(clases_str):
                        r[f"P({cls})"] = (round(float(probas[i][j]), 4)
                                          if j < len(probas[i]) else 0)
                filas_result.append(r)
        else:
            filas_result = []
            for i, pred in enumerate(y_labels):
                r = {"#": i+1, "prediccion": pred}
                if probas and i < len(probas):
                    for j, cls in enumerate(clases_str):
                        r[f"P({cls})"] = (round(float(probas[i][j]), 4)
                                          if j < len(probas[i]) else 0)
                filas_result.append(r)

        excluidas = len(session.get("train_indices", []))
        return jsonify({
            "ok": True, "modelo": display, "modo": modo,
            "predicciones": filas_result,
            "metricas": metricas, "cm": cm_data, "graficas_cm": grafs_cm,
            "clases": clases_str,
            "excluidas_training": excluidas if modo == "muestra" else 0,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/predecir_csv", methods=["POST"])
def predecir_csv():
    try:
        df = get_df(); tc = get_target()
        if df is None or tc is None:
            return jsonify({"error": "Carga dataset y selecciona target"}), 400
        mods_guard = session.get("modelos_entrenados", {}) or {}
        if not mods_guard:
            return jsonify({"error": "Entrena un clasificador primero"}), 400

        f      = request.files.get("archivo")
        nom_m  = request.form.get("modelo", "")
        sep    = request.form.get("sep", ",")
        if sep == "tab": sep = "\t"
        if f is None:
            return jsonify({"error": "Sin archivo"}), 400

        df_n  = pd.read_csv(f, sep=sep)
        entry = mods_guard.get(nom_m) or list(mods_guard.values())[0]
        clf   = entry["clf"]
        X_cols = entry["X_columns"]
        display = entry.get("display", nom_m)

        from utils.preprocesamiento import preparar_datos, _target_encoder, _target_es_string
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

        tiene_real = tc in df_n.columns
        if tiene_real:
            X_n, y_real_enc = preparar_datos(df_n, tc)
        else:
            df_tmp = df_n.copy(); df_tmp[tc] = 0
            X_n, y_real_enc = preparar_datos(df_tmp, tc)

        X_new  = X_n.reindex(columns=X_cols, fill_value=0).values
        y_pred = clf.predict(X_new)
        y_labels = _decode(y_pred)
        probas, clases_str = _get_probas(clf, X_new)

        metricas = None; cm_data = None; grafs_cm = None
        if tiene_real:
            y_real_str = _decode(y_real_enc)
            acc   = float(accuracy_score(y_real_str, y_labels))
            f1v   = float(f1_score(y_real_str, y_labels,
                                    average="weighted", zero_division=0))
            try:
                kappa = float(cohen_kappa_score(y_real_str, y_labels))
            except Exception:
                kappa = 0.0
            clases_u = sorted(set(y_real_str + y_labels))
            cm_data  = confusion_matrix(
                y_real_str, y_labels, labels=clases_u).tolist()
            abc = _accuracy_by_class(cm_data, clases_u)
            metricas = {"accuracy": round(acc, 6), "f1": round(f1v, 6),
                        "kappa": round(kappa, 6), "clases": clases_u,
                        "abc": abc,
                        "n_total": len(y_real_str),
                        "n_correctas": int(sum(r == p for r, p
                                               in zip(y_real_str, y_labels)))}
            grafs_cm = [_grafica_cm(cm_data, clases_u, display)]

        filas = []
        for i, pred in enumerate(y_labels):
            r = {"#": i+1, "prediccion": pred}
            if tiene_real:
                r["real"] = y_real_str[i]
                r["correcto"] = "✓" if y_real_str[i] == pred else "✗"
            if probas and i < len(probas):
                for j, cls in enumerate(clases_str):
                    r[f"P({cls})"] = (round(float(probas[i][j]), 4)
                                      if j < len(probas[i]) else 0)
            filas.append(r)

        return jsonify({"ok": True, "predicciones": filas,
                        "metricas": metricas, "cm": cm_data,
                        "graficas_cm": grafs_cm, "modelo": display})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# helpers predicción
def _to_float(v):
    try: return float(v)
    except: return 0.0

def _decode(y):
    from utils.preprocesamiento import _target_encoder, _target_es_string
    if _target_es_string:
        try:
            return [str(_target_encoder.classes_[int(p)]) for p in y]
        except Exception:
            pass
    return [str(p) for p in y]

def _get_probas(clf, X_new):
    try:
        ps = clf.predict_proba(X_new)
        clases = [str(c) for c in clf.classes_]
        return ps.tolist(), clases
    except Exception:
        return None, []


# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
@api_bp.route("/api/resultados", methods=["GET"])
def get_resultados():
    try:
        raw     = session.get("ultimo_resumen")
        resumen = None
        if raw:
            df_r = pd.read_json(io.StringIO(raw))
            resumen = (df_r.reset_index()
                       .rename(columns={"index": "variable"})
                       .to_dict("records"))
            resumen = [{k: (round(float(v), 8) if isinstance(v, float) else v)
                        for k, v in r.items()} for r in resumen]
        # deduplica clf_resultados por modelo
        clf_res = session.get("clf_resultados", [])
        seen = {}
        for r in clf_res:
            seen[r["modelo"]] = r
        clf_dedup = list(seen.values())
        return jsonify({"ok": True, "resultados": {
            "metodos_ejecutados":  session.get("metodos_ejecutados", []),
            "ultimo_resumen":      resumen,
            "clasificadores":      clf_dedup,
            "modelos_disponibles": list(
                (session.get("modelos_entrenados") or {}).keys()),
        }})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# ─────────────────────────────────────────────────────────────
# DESCARGAR DATASET MODIFICADO
# ─────────────────────────────────────────────────────────────
@api_bp.route("/api/descargar_dataset", methods=["GET"])
def descargar_dataset():
    """Devuelve el dataset actual de la sesión como archivo CSV descargable."""
    try:
        from flask import make_response
        df = get_df()
        if df is None:
            return jsonify({"error": "Sin dataset en sesión"}), 400
        nombre = session.get("nombre", "dataset").replace(".csv", "")
        nombre_out = nombre + "_modificado.csv"
        csv_str = df.to_csv(index=False)
        resp = make_response(csv_str)
        resp.headers["Content-Type"]        = "text/csv; charset=utf-8"
        resp.headers["Content-Disposition"] = f"attachment; filename={nombre_out}"
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────────────────────────────
def _barras(resultados, titulo):
    if not resultados: return None
    campo = "SCORE_FINAL" if "SCORE_FINAL" in resultados[0] else "score"
    datos = sorted(resultados, key=lambda x: x.get(campo, 0), reverse=True)[:15]
    vars_ = [d["variable"] for d in datos]
    vals  = [d.get(campo, 0) for d in datos]
    max_v = max(vals) if vals else 1
    fig, ax = _dark_fig(10, max(4, len(vars_)*0.42))
    cols = ["#4a9eff" if v >= max_v*0.7
            else "#2d6cbf" if v >= max_v*0.4
            else "#1a3d6b" for v in vals]
    ax.barh(vars_[::-1], vals[::-1], color=cols[::-1],
            height=0.65, edgecolor="none")
    ax.set_title(titulo, color="#ffffff", fontsize=11, pad=10, fontweight="bold")
    ax.set_xlabel("Score", color="#a0c4ff", fontsize=9)
    ax.grid(axis="x", color="#1e3a5f", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    return fig_to_b64(fig)

def _barras_clf(resultados):
    if not resultados: return None
    nombres = [r["modelo"] for r in resultados]
    accs    = [r["accuracy"] for r in resultados]
    f1s     = [r["f1"] for r in resultados]
    cvs     = [r["cv_mean"] for r in resultados]
    x = np.arange(len(nombres)); w = 0.25
    fig, ax = _dark_fig(max(6, len(nombres)*1.5+2), 4)
    ax.bar(x-w,   accs, w, label="Accuracy",  color="#4a9eff", edgecolor="none")
    ax.bar(x,     f1s,  w, label="F1",        color="#2dd4bf", edgecolor="none")
    ax.bar(x+w,   cvs,  w, label="CV mean",   color="#a78bfa", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, color="#a0c4ff", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Comparación de clasificadores", color="#ffffff",
                 fontsize=11, pad=10, fontweight="bold")
    ax.legend(facecolor="#0d1e35", edgecolor="#1e3a5f",
              labelcolor="#a0c4ff", fontsize=8)
    ax.grid(axis="y", color="#1e3a5f", linewidth=0.5, alpha=0.6)
    # Anotar valores
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom",
                        color="#ffffff", fontsize=7)
    plt.tight_layout()
    return fig_to_b64(fig)

def _grafica_cm(cm, clases, titulo):
    n = len(clases)
    sz = max(4, n * 1.2)
    fig, ax = _dark_fig(sz, max(3.5, sz * 0.8))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(clases, rotation=45, ha="right",
                       color="#a0c4ff", fontsize=9)
    ax.set_yticklabels(clases, color="#a0c4ff", fontsize=9)
    ax.set_xlabel("Predicho", color="#7eb3ff", fontsize=9)
    ax.set_ylabel("Real",     color="#7eb3ff", fontsize=9)
    ax.set_title(f"Confusion Matrix — {titulo}", color="#ffffff",
                 fontsize=10, pad=10, fontweight="bold")
    thresh = cm_arr.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_arr[i, j] > thresh else "#4a9eff",
                    fontweight="bold")
    plt.colorbar(im, ax=ax).ax.tick_params(colors="#a0c4ff")
    plt.tight_layout()
    return fig_to_b64(fig)

def _heatmap(mat):
    n = len(mat)
    sz = max(4, min(12, n * 0.45))
    fig, ax = _dark_fig(sz, sz * 0.85)
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Similitud de Gower", color="#ffffff",
                 fontsize=11, pad=8, fontweight="bold")
    if n <= 25:
        lbls = [f"I{i}" for i in range(n)]
        ax.set_xticks(range(n)); ax.set_xticklabels(lbls, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(lbls, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors="#a0c4ff", labelsize=7)
    plt.tight_layout()
    return fig_to_b64(fig)