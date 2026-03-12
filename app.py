"""
app.py — Servidor Flask para Selector de Variables
Corre con: python app.py
Abre:      http://localhost:5000
"""

import os, json, io, base64, traceback
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

app = Flask(__name__)
app.secret_key = "selector_variables_2025"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = ".flask_sessions"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(".flask_sessions", exist_ok=True)
Session(app)

# ─── ESTADO GLOBAL (por sesión) ──────────────────────────────────────────────
_estado = {}

def get_df():
    import pickle
    data = session.get("df_pickle")
    if data is None:
        return None
    return pickle.loads(data)

def set_df(df):
    import pickle
    session["df_pickle"] = pickle.dumps(df)

def get_target():
    return session.get("target_col")

def get_nombre():
    return session.get("nombre_archivo", "dataset.csv")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    """Convierte figura matplotlib a base64 para HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0a1628", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def df_info(df, target_col):
    """Estadísticas rápidas del dataset para la UI."""
    info = {
        "filas": len(df),
        "columnas": len(df.columns),
        "target": target_col,
        "nulos": int(df.isnull().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "clases": int(df[target_col].nunique()) if target_col else 0,
        "tipos": {},
    }
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype == "object" or dtype == "category":
            info["tipos"][col] = "nominal"
        elif "int" in dtype:
            info["tipos"][col] = "entero"
        else:
            info["tipos"][col] = "flotante"
    return info


# ─── RUTAS PRINCIPALES ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cargar", methods=["POST"])
def cargar_dataset():
    try:
        if "archivo" not in request.files:
            return jsonify({"error": "No se recibió archivo"}), 400

        f    = request.files["archivo"]
        sep  = request.form.get("separador", ",")
        if sep == "tab":
            sep = "\t"

        df = pd.read_csv(f, sep=sep)
        if df.shape[1] <= 1:
            return jsonify({"error": "Solo se detectó 1 columna. ¿Separador correcto?"}), 400

        set_df(df)
        session["target_col"]      = None
        session["nombre_archivo"]  = f.filename
        session["ultimo_resumen"]  = None

        columnas = []
        for i, col in enumerate(df.columns):
            dtype = str(df[col].dtype)
            n_uniq = int(df[col].nunique())
            if dtype == "object" or dtype == "category":
                tipo, icono = "STRING", "✓"
            elif "int" in dtype and n_uniq <= 20:
                tipo, icono = f"INT ({n_uniq} clases)", "~"
            elif "int" in dtype:
                tipo, icono = f"INT continuo", "·"
            else:
                tipo, icono = f"FLOAT continuo", "·"
            columnas.append({
                "idx": i, "nombre": col, "tipo": tipo,
                "icono": icono, "nulos": int(df[col].isnull().sum())
            })

        preview = df.head(5).fillna("").to_dict(orient="records")

        return jsonify({
            "ok": True,
            "nombre": f.filename,
            "filas": len(df),
            "columnas_count": len(df.columns),
            "columnas": columnas,
            "preview": preview,
            "headers": list(df.columns),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/set_target", methods=["POST"])
def set_target():
    try:
        df = get_df()
        if df is None:
            return jsonify({"error": "Carga un dataset primero"}), 400
        col = request.json.get("columna")
        if col not in df.columns:
            return jsonify({"error": f"Columna '{col}' no existe"}), 400
        session["target_col"] = col
        clases = df[col].dropna().unique().tolist()[:10]
        return jsonify({
            "ok": True, "target": col,
            "n_clases": int(df[col].nunique()),
            "muestra": [str(c) for c in clases]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── SELECTORES ───────────────────────────────────────────────────────────────

@app.route("/api/selector/<metodo>", methods=["POST"])
def ejecutar_selector(metodo):
    try:
        df         = get_df()
        target_col = get_target()
        if df is None or target_col is None:
            return jsonify({"error": "Carga el dataset y selecciona el target primero"}), 400

        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))

        resultados = []

        if metodo == "correlacion":
            import metodos.correlacion as m
            scores = m.ejecutar(df, target_col)
            resultados = [{"variable": k, "score": round(float(v), 10)}
                          for k, v in scores.items()]

        elif metodo == "chi2":
            import metodos.chi2 as m
            scores = m.ejecutar(df, target_col)
            resultados = [{"variable": k, "score": round(float(v), 10)}
                          for k, v in scores.items()]

        elif metodo == "random_forest":
            import metodos.random_forest as m
            scores = m.ejecutar(df, target_col)
            resultados = [{"variable": k, "score": round(float(v), 10)}
                          for k, v in scores.items()]

        elif metodo == "gower":
            import metodos.gower as m
            scores = m.ejecutar(df, target_col)
            resultados = [{"variable": k, "score": round(float(v), 10)}
                          for k, v in scores.items()]

        elif metodo == "weka":
            import metodos.weka_attsel as m
            res = m.ejecutar(df, target_col)
            if not res.get("disponible"):
                return jsonify({"error": "Weka no disponible. Verifica JAVA_HOME."}), 400
            scores = res.get("InfoGain", pd.Series(dtype=float))
            resultados = [{"variable": k, "score": round(float(v), 10)}
                          for k, v in scores.items()]
            session["weka_result"] = json.dumps({
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in res.items() if k != "disponible"
            })

        elif metodo == "todos":
            from utils.preprocesamiento import normalizar_serie
            import metodos.correlacion   as mc
            import metodos.chi2          as mq
            import metodos.random_forest as mr
            import metodos.gower         as mg

            corr = mc.ejecutar(df, target_col)
            chi  = mq.ejecutar(df, target_col)
            rf   = mr.ejecutar(df, target_col)
            gwr  = mg.ejecutar(df, target_col)

            idx = corr.index.union(chi.index).union(rf.index).union(gwr.index)
            resumen = pd.DataFrame(index=idx)
            resumen["Correlacion"]  = normalizar_serie(corr.reindex(idx).fillna(0))
            resumen["Chi2_F"]       = normalizar_serie(chi.reindex(idx).fillna(0))
            resumen["RandomForest"] = normalizar_serie(rf.reindex(idx).fillna(0))
            resumen["Gower"]        = normalizar_serie(gwr.reindex(idx).fillna(0))
            resumen["SCORE_FINAL"]  = resumen.mean(axis=1)
            resumen = resumen.sort_values("SCORE_FINAL", ascending=False)

            session["ultimo_resumen"] = resumen.to_json()
            resultados = resumen.reset_index().rename(
                columns={"index": "variable"}).to_dict(orient="records")
            resultados = [{k: (round(float(v), 10) if isinstance(v, float) else v)
                          for k, v in r.items()} for r in resultados]

        # Gráfica de barras
        grafica = _grafica_barras(resultados, metodo)

        return jsonify({"ok": True, "resultados": resultados, "grafica": grafica})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _grafica_barras(resultados, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not resultados:
        return None

    campo = "SCORE_FINAL" if "SCORE_FINAL" in resultados[0] else "score"
    datos = sorted(resultados, key=lambda x: x.get(campo, 0), reverse=True)[:15]
    vars_ = [d["variable"] for d in datos]
    vals  = [d.get(campo, 0) for d in datos]
    max_v = max(vals) if vals else 1

    fig, ax = plt.subplots(figsize=(10, max(4, len(vars_) * 0.45)))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0d1e35")

    colores = ["#4a9eff" if v >= max_v * 0.7 else
               "#2d6cbf" if v >= max_v * 0.4 else "#1a3d6b" for v in vals]
    bars = ax.barh(vars_[::-1], vals[::-1], color=colores[::-1],
                   height=0.65, edgecolor="none")

    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + max_v * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.6f}", va="center", ha="left",
                color="#a0c4ff", fontsize=8, fontfamily="monospace")

    ax.set_xlabel("Score", color="#7eb3ff", fontsize=9)
    ax.set_title(titulo.replace("_", " ").title(),
                 color="#ffffff", fontsize=11, pad=12, fontweight="bold")
    ax.tick_params(colors="#a0c4ff", labelsize=8)
    ax.spines[:].set_color("#1e3a5f")
    ax.xaxis.label.set_color("#7eb3ff")
    ax.set_xlim(0, max_v * 1.18)
    ax.grid(axis="x", color="#1e3a5f", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    return fig_to_b64(fig)


# ─── CLASIFICADORES ───────────────────────────────────────────────────────────

@app.route("/api/clasificar", methods=["POST"])
def clasificar():
    try:
        df         = get_df()
        target_col = get_target()
        if df is None or target_col is None:
            return jsonify({"error": "Carga el dataset y selecciona el target primero"}), 400

        data      = request.json
        metodos_s = data.get("metodos", ["j48"])
        test_size = float(data.get("test_size", 0.2))
        params    = data.get("params", {})

        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from utils.preprocesamiento import preparar_datos, target_es_categorico
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import (accuracy_score, f1_score,
                                     confusion_matrix, classification_report)

        X, y, _ = _preparar_local(df, target_col)
        es_clf  = target_es_categorico(df, target_col)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if es_clf else None)

        todos_resultados = []
        matrices         = []

        for nombre in metodos_s:
            clf = _get_clasificador(nombre, params.get(nombre, {}), es_clf)
            if clf is None:
                continue
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            if es_clf:
                acc    = float(accuracy_score(y_te, y_pred))
                f1     = float(f1_score(y_te, y_pred,
                                        average="weighted", zero_division=0))
                cv     = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
                cm     = confusion_matrix(y_te, y_pred).tolist()
                cr     = classification_report(y_te, y_pred,
                                               zero_division=0, output_dict=True)
                clases = [str(c) for c in sorted(set(y_te) | set(y_pred))]

                todos_resultados.append({
                    "modelo": nombre, "accuracy": round(acc, 10),
                    "f1": round(f1, 10),
                    "cv_mean": round(float(cv.mean()), 10),
                    "cv_std":  round(float(cv.std()), 10),
                    "tipo": "clasificacion",
                    "reporte": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                                    for kk, vv in v.items()}
                                if isinstance(v, dict) else round(v, 6)
                                for k, v in cr.items()},
                })
                matrices.append({
                    "modelo": nombre, "cm": cm, "clases": clases
                })
            else:
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                r2   = float(r2_score(y_te, y_pred))
                mae  = float(mean_absolute_error(y_te, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                cv   = cross_val_score(clf, X, y, cv=5, scoring="r2")
                todos_resultados.append({
                    "modelo": nombre, "r2": round(r2, 10),
                    "mae": round(mae, 10), "rmse": round(rmse, 10),
                    "cv_mean": round(float(cv.mean()), 10),
                    "cv_std":  round(float(cv.std()), 10),
                    "tipo": "regresion",
                })

        graficas_cm = [_grafica_confusion(m["cm"], m["clases"], m["modelo"])
                       for m in matrices]

        return jsonify({
            "ok": True,
            "resultados": todos_resultados,
            "matrices": matrices,
            "graficas_cm": graficas_cm,
            "es_clasificacion": es_clf,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _preparar_local(df, target_col):
    from utils.preprocesamiento import preparar_datos, target_es_categorico
    X, y   = preparar_datos(df, target_col)
    es_clf = target_es_categorico(df, target_col)
    return X, y, es_clf


def _get_clasificador(nombre, params, es_clf):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if nombre == "j48":
        from sklearn.tree import DecisionTreeClassifier
        return Pipeline([("sc", StandardScaler()),
                         ("m", DecisionTreeClassifier(
                             criterion="entropy",
                             max_depth=params.get("max_depth", None),
                             min_samples_leaf=params.get("min_samples_leaf", 1),
                             random_state=42))])
    elif nombre == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return Pipeline([("sc", StandardScaler()),
                         ("m", GaussianNB())])
    elif nombre == "ibk":
        from sklearn.neighbors import KNeighborsClassifier
        return Pipeline([("sc", StandardScaler()),
                         ("m", KNeighborsClassifier(
                             n_neighbors=params.get("k", 3)))])
    elif nombre == "smo":
        from sklearn.svm import SVC
        return Pipeline([("sc", StandardScaler()),
                         ("m", SVC(kernel=params.get("kernel", "rbf"),
                                   C=params.get("C", 1.0),
                                   probability=True, random_state=42))])
    return None


def _grafica_confusion(cm, clases, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm_arr = np.array(cm)
    n      = len(clases)
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(3.5, n)))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0d1e35")

    im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(clases, rotation=45, ha="right",
                       color="#a0c4ff", fontsize=8)
    ax.set_yticklabels(clases, color="#a0c4ff", fontsize=8)
    ax.set_xlabel("Predicho", color="#7eb3ff", fontsize=9)
    ax.set_ylabel("Real",     color="#7eb3ff", fontsize=9)
    ax.set_title(f"Matriz de Confusión — {titulo}",
                 color="#ffffff", fontsize=10, pad=10)

    thresh = cm_arr.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_arr[i, j] > thresh else "#4a9eff",
                    fontweight="bold")

    ax.spines[:].set_color("#1e3a5f")
    plt.colorbar(im, ax=ax).ax.tick_params(colors="#a0c4ff")
    plt.tight_layout()
    return fig_to_b64(fig)


# ─── GOWER ────────────────────────────────────────────────────────────────────

@app.route("/api/gower_matriz", methods=["POST"])
def gower_matriz():
    try:
        df         = get_df()
        target_col = get_target()
        if df is None:
            return jsonify({"error": "Carga un dataset primero"}), 400

        data     = request.json or {}
        max_inst = int(data.get("max_instancias", 30))
        df_sub   = df.head(max_inst).copy()

        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from utils.preprocesamiento import preparar_datos

        X, _, _ = _preparar_local(df_sub, target_col) if target_col else (df_sub, None, None)

        # Calcular matriz de Gower manualmente
        n    = len(X)
        cols = X.columns.tolist()
        mat  = np.zeros((n, n))

        rangos = {}
        for col in cols:
            try:
                vals = pd.to_numeric(X[col], errors="coerce").dropna()
                r    = float(vals.max() - vals.min())
                rangos[col] = r if r > 0 else 1.0
            except:
                rangos[col] = 1.0

        for i in range(n):
            for j in range(i + 1, n):
                dists = []
                for col in cols:
                    vi, vj = X.iloc[i][col], X.iloc[j][col]
                    try:
                        vi_n = float(vi); vj_n = float(vj)
                        d = abs(vi_n - vj_n) / rangos[col]
                    except:
                        d = 0.0 if str(vi) == str(vj) else 1.0
                    dists.append(d)
                mat[i, j] = mat[j, i] = float(np.mean(dists))

        sim = 1.0 - mat

        # Vecino más cercano para cada instancia
        vecinos = []
        for i in range(n):
            sims_i = [(j, float(sim[i, j])) for j in range(n) if j != i]
            mejor  = max(sims_i, key=lambda x: x[1])
            vecinos.append({
                "instancia": i,
                "vecino": mejor[0],
                "similitud": round(mejor[1], 8),
            })

        # Gráfica heatmap
        grafica = _grafica_heatmap(sim, "Similitud de Gower")

        etiquetas = [f"I{i}" for i in range(n)]
        return jsonify({
            "ok": True,
            "n": n,
            "similitud": sim.tolist(),
            "distancia": mat.tolist(),
            "etiquetas": etiquetas,
            "vecinos": vecinos,
            "grafica": grafica,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _grafica_heatmap(matriz, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n   = len(matriz)
    sz  = max(5, min(14, n * 0.4))
    fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0d1e35")

    im = ax.imshow(matriz, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_title(titulo, color="#ffffff", fontsize=11, pad=10, fontweight="bold")
    ax.tick_params(colors="#a0c4ff", labelsize=7)
    ax.spines[:].set_color("#1e3a5f")

    if n <= 20:
        ticks = list(range(n))
        lbls  = [f"I{i}" for i in range(n)]
        ax.set_xticks(ticks); ax.set_xticklabels(lbls, rotation=90, fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(lbls, fontsize=7)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors="#a0c4ff", labelsize=7)
    plt.tight_layout()
    return fig_to_b64(fig)


# ─── PREDICCIÓN ───────────────────────────────────────────────────────────────

@app.route("/api/predecir", methods=["POST"])
def predecir():
    try:
        df         = get_df()
        target_col = get_target()
        if df is None or target_col is None:
            return jsonify({"error": "Carga el dataset y selecciona el target primero"}), 400

        data      = request.json or {}
        modelo_n  = data.get("modelo", "j48")
        instancia = data.get("instancia", {})
        test_size = float(data.get("test_size", 0.2))

        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from sklearn.model_selection import train_test_split

        X, y, es_clf = _preparar_local(df, target_col)
        X_tr, X_te, y_tr, _ = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if es_clf else None)

        clf = _get_clasificador(modelo_n, {}, es_clf)
        clf.fit(X_tr, y_tr)

        # Construir fila de instancia nueva
        row = {}
        for col in X.columns:
            val = instancia.get(col, 0)
            try:
                row[col] = float(val)
            except:
                row[col] = 0.0

        X_new = pd.DataFrame([row])[X.columns]
        pred  = clf.predict(X_new)[0]

        # Probabilidades si disponibles
        proba = None
        if hasattr(clf, "predict_proba"):
            try:
                p     = clf.predict_proba(X_new)[0]
                clases = clf.classes_
                proba = {str(c): round(float(p_), 6)
                         for c, p_ in zip(clases, p)}
            except:
                pass

        try:
            from utils.preprocesamiento import _target_encoder, _target_es_string
            if _target_es_string:
                pred = str(_target_encoder.inverse_transform([int(pred)])[0])
        except Exception:
            pred = str(pred)

        return jsonify({
            "ok": True,
            "prediccion": str(pred),
            "probabilidades": proba,
            "modelo": modelo_n,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── REPORTE COMPLETO ─────────────────────────────────────────────────────────

@app.route("/api/reporte_completo", methods=["POST"])
def reporte_completo():
    try:
        df         = get_df()
        target_col = get_target()
        if df is None or target_col is None:
            return jsonify({"error": "Carga el dataset y selecciona el target primero"}), 400

        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from utils.preprocesamiento import normalizar_serie, preparar_datos, target_es_categorico
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import (accuracy_score, f1_score,
                                     confusion_matrix, classification_report)
        import metodos.correlacion   as mc
        import metodos.chi2          as mq
        import metodos.random_forest as mr
        import metodos.gower         as mg

        # Selectores
        corr = mc.ejecutar(df, target_col)
        chi  = mq.ejecutar(df, target_col)
        rf   = mr.ejecutar(df, target_col)
        gwr  = mg.ejecutar(df, target_col)
        idx  = corr.index.union(chi.index).union(rf.index).union(gwr.index)

        resumen = pd.DataFrame(index=idx)
        resumen["Correlacion"]  = normalizar_serie(corr.reindex(idx).fillna(0))
        resumen["Chi2_F"]       = normalizar_serie(chi.reindex(idx).fillna(0))
        resumen["RandomForest"] = normalizar_serie(rf.reindex(idx).fillna(0))
        resumen["Gower"]        = normalizar_serie(gwr.reindex(idx).fillna(0))
        resumen["SCORE_FINAL"]  = resumen.mean(axis=1)
        resumen = resumen.sort_values("SCORE_FINAL", ascending=False)

        top_features = resumen["SCORE_FINAL"].nlargest(5).index.tolist()
        grafica_sel  = _grafica_barras(
            resumen.reset_index().rename(columns={"index": "variable"}).to_dict("records"),
            "Score Final — Todos los Selectores")

        # Clasificadores
        X, y, es_clf = _preparar_local(df, target_col)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if es_clf else None)

        clfs    = ["j48", "naive_bayes", "ibk", "smo"]
        clf_res = []
        cms     = []

        for nombre in clfs:
            clf = _get_clasificador(nombre, {}, es_clf)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            if es_clf:
                acc = float(accuracy_score(y_te, y_pred))
                f1  = float(f1_score(y_te, y_pred,
                                     average="weighted", zero_division=0))
                cv  = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
                cm  = confusion_matrix(y_te, y_pred).tolist()
                clases = [str(c) for c in sorted(set(y_te) | set(y_pred))]
                clf_res.append({
                    "modelo": nombre, "accuracy": round(acc, 6),
                    "f1": round(f1, 6),
                    "cv_mean": round(float(cv.mean()), 6),
                    "cv_std":  round(float(cv.std()), 6),
                })
                cms.append({"modelo": nombre, "cm": cm, "clases": clases})

        graficas_cm = [_grafica_confusion(m["cm"], m["clases"], m["modelo"])
                       for m in cms]
        grafica_comp = _grafica_comparacion_clfs(clf_res)

        return jsonify({
            "ok": True,
            "dataset": df_info(df, target_col),
            "selectores": resumen.reset_index().rename(
                columns={"index": "variable"}).to_dict("records"),
            "top_features": top_features,
            "grafica_selectores": grafica_sel,
            "clasificadores": clf_res,
            "graficas_cm": graficas_cm,
            "grafica_comparacion": grafica_comp,
            "matrices": cms,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _grafica_comparacion_clfs(resultados):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nombres = [r["modelo"] for r in resultados]
    accs    = [r["accuracy"] for r in resultados]
    f1s     = [r["f1"] for r in resultados]
    x       = np.arange(len(nombres))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0d1e35")

    b1 = ax.bar(x - w/2, accs, w, label="Accuracy",
                color="#4a9eff", edgecolor="none")
    b2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)",
                color="#2dd4bf", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(nombres, color="#a0c4ff", fontsize=9)
    ax.tick_params(colors="#a0c4ff")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color="#7eb3ff", fontsize=9)
    ax.set_title("Comparación de Clasificadores",
                 color="#ffffff", fontsize=11, pad=10, fontweight="bold")
    ax.legend(facecolor="#0d1e35", edgecolor="#1e3a5f",
              labelcolor="#a0c4ff", fontsize=8)
    ax.spines[:].set_color("#1e3a5f")
    ax.grid(axis="y", color="#1e3a5f", linewidth=0.5, alpha=0.6)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", color="#a0c4ff", fontsize=7)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", color="#a0c4ff", fontsize=7)

    plt.tight_layout()
    return fig_to_b64(fig)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
