"""
web/routes/api.py
Todas las rutas Flask. Importa la misma lógica que usa el CLI.
"""
import io, base64, traceback, json
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, session
import pickle

api_bp = Blueprint("api", __name__)

# ── Helpers de sesión ────────────────────────────────────────────────────
def get_df():
    data = session.get("df_pickle")
    return pickle.loads(data) if data else None

def set_df(df):
    session["df_pickle"] = pickle.dumps(df)

def get_target():
    return session.get("target_col")

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0a1628", dpi=110)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    import matplotlib.pyplot as plt; plt.close(fig)
    return f"data:image/png;base64,{b64}"

# ── Cargar CSV ───────────────────────────────────────────────────────────
@api_bp.route("/api/cargar", methods=["POST"])
def cargar_dataset():
    try:
        f   = request.files.get("archivo")
        sep = request.form.get("separador", ",")
        if sep == "tab": sep = "\t"
        if f is None: return jsonify({"error": "Sin archivo"}), 400
        df = pd.read_csv(f, sep=sep)
        if df.shape[1] <= 1: return jsonify({"error": "Solo 1 columna — separador?"}), 400
        set_df(df)
        session["target_col"] = None
        session["nombre"]     = f.filename
        columnas = []
        for i, col in enumerate(df.columns):
            nu = int(df[col].nunique())
            dt = str(df[col].dtype)
            if dt == "object": tipo, icono = "STRING", "✓"
            elif "int" in dt and nu <= 20: tipo, icono = f"INT {nu} clases", "~"
            elif "int" in dt: tipo, icono = "INT continuo", "·"
            else: tipo, icono = "FLOAT continuo", "·"
            columnas.append({"idx":i,"nombre":col,"tipo":tipo,"icono":icono,
                              "nulos":int(df[col].isnull().sum())})
        return jsonify({"ok":True,"nombre":f.filename,"filas":len(df),
                        "columnas_count":len(df.columns),"columnas":columnas,
                        "preview":df.head(5).fillna("").to_dict("records"),
                        "headers":list(df.columns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/set_target", methods=["POST"])
def set_target():
    try:
        df  = get_df()
        col = request.json.get("columna")
        if df is None: return jsonify({"error": "Sin dataset"}), 400
        if col not in df.columns: return jsonify({"error": f"Columna '{col}' no existe"}), 400
        session["target_col"] = col
        muestra = df[col].dropna().unique().tolist()[:10]
        return jsonify({"ok":True,"target":col,"n_clases":int(df[col].nunique()),
                        "muestra":[str(c) for c in muestra]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Selectores ───────────────────────────────────────────────────────────
@api_bp.route("/api/selector/<metodo>", methods=["POST"])
def ejecutar_selector(metodo):
    try:
        df = get_df(); tc = get_target()
        if df is None or tc is None:
            return jsonify({"error": "Carga dataset y selecciona target"}), 400
        import metodos.seleccion as sel
        from utils.preprocesamiento import normalizar_serie
        if metodo == "correlacion":  scores = sel.correlacion(df, tc)
        elif metodo == "chi2":       scores = sel.chi2(df, tc)
        elif metodo == "random_forest": scores = sel.random_forest(df, tc)
        elif metodo == "gower":      scores = sel.gower(df, tc)
        elif metodo == "todos":
            resumen = sel.comparar_todos(df, tc, session.get("nombre","dataset"))
            session["ultimo_resumen"] = resumen.to_json()
            resultados = resumen.reset_index().rename(columns={"index":"variable"}).to_dict("records")
            resultados = [{k:(round(float(v),8) if isinstance(v,float) else v) for k,v in r.items()} for r in resultados]
            grafica = _barras(resultados, "Score Final")
            return jsonify({"ok":True,"resultados":resultados,"grafica":grafica})
        else:
            return jsonify({"error": f"Metodo '{metodo}' no reconocido"}), 400
        resultados = [{"variable":k,"score":round(float(v),8)} for k,v in scores.items()]
        grafica    = _barras(resultados, metodo)
        return jsonify({"ok":True,"resultados":resultados,"grafica":grafica})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Clasificadores ───────────────────────────────────────────────────────
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

        if tipo == "supervisado":
            from metodos.supervisado import get_clasificador, CATALOGO
        else:
            from metodos.no_supervisado import get_clasificador, CATALOGO

        from utils.entrenador import evaluar, imprimir_comparacion
        resultados_api = []

        for nombre in nombres:
            n_cl = int(df[tc].nunique())
            clf  = get_clasificador(nombre, params.get(nombre, {}), n_cl)
            if clf is None: continue
            mod     = CATALOGO.get(nombre)
            display = getattr(mod,"NOMBRE",nombre) if mod else nombre
            skip_cv = (tipo=="no_supervisado" and len(df)>500)
            res     = evaluar(df, tc, clf, display, test_sz, skip_cv)
            resultados_api.append({
                "modelo":   nombre,
                "accuracy": round(res["accuracy"],8),
                "f1":       round(res["f1"],8),
                "cv_mean":  round(res["cv_mean"],8),
                "cv_std":   round(res["cv_std"],8),
            })

        grafica = _barras_clf(resultados_api)
        return jsonify({"ok":True,"resultados":resultados_api,"grafica":grafica})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Helpers gráficas ─────────────────────────────────────────────────────
def _barras(resultados, titulo):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not resultados: return None
    campo = "SCORE_FINAL" if "SCORE_FINAL" in resultados[0] else "score"
    datos = sorted(resultados, key=lambda x: x.get(campo,0), reverse=True)[:15]
    vars_ = [d["variable"] for d in datos]
    vals  = [d.get(campo,0) for d in datos]
    max_v = max(vals) if vals else 1
    fig, ax = plt.subplots(figsize=(10, max(4, len(vars_)*0.45)))
    fig.patch.set_facecolor("#0a1628"); ax.set_facecolor("#0d1e35")
    colores = ["#4a9eff" if v>=max_v*0.7 else "#2d6cbf" if v>=max_v*0.4 else "#1a3d6b" for v in vals]
    ax.barh(vars_[::-1], vals[::-1], color=colores[::-1], height=0.65, edgecolor="none")
    ax.set_title(titulo, color="#ffffff", fontsize=11, pad=10, fontweight="bold")
    ax.tick_params(colors="#a0c4ff", labelsize=8)
    ax.spines[:].set_color("#1e3a5f")
    ax.set_facecolor("#0d1e35"); ax.grid(axis="x",color="#1e3a5f",linewidth=0.5,alpha=0.7)
    plt.tight_layout()
    return fig_to_b64(fig)

def _barras_clf(resultados):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not resultados: return None
    nombres = [r["modelo"] for r in resultados]
    accs    = [r["accuracy"] for r in resultados]
    f1s     = [r["f1"] for r in resultados]
    x       = np.arange(len(nombres)); w = 0.35
    fig, ax = plt.subplots(figsize=(9,4))
    fig.patch.set_facecolor("#0a1628"); ax.set_facecolor("#0d1e35")
    ax.bar(x-w/2, accs, w, label="Accuracy", color="#4a9eff", edgecolor="none")
    ax.bar(x+w/2, f1s,  w, label="F1",       color="#2dd4bf", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(nombres, color="#a0c4ff", fontsize=9)
    ax.set_ylim(0,1.1); ax.tick_params(colors="#a0c4ff")
    ax.set_title("Comparación clasificadores", color="#ffffff", fontsize=11, pad=10, fontweight="bold")
    ax.legend(facecolor="#0d1e35",edgecolor="#1e3a5f",labelcolor="#a0c4ff",fontsize=8)
    ax.spines[:].set_color("#1e3a5f")
    ax.grid(axis="y",color="#1e3a5f",linewidth=0.5,alpha=0.6)
    plt.tight_layout()
    return fig_to_b64(fig)
