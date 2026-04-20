"""
web/routes/api_patch.py
────────────────────────────────────────────────────────────
Helpers que enriquecen los dicts de respuesta de cada endpoint
con los campos que el frontend (index.html) ya espera pero que
el api.py original no devolvía:

  meta          → cabecera estilo Weka (evaluador, instancias, tiempo…)
  seleccionadas → atributos con merit ≥ 50%
  acuerdo       → 3 grupos de acuerdo entre métodos (para /todos)
  por_metodo    → ranking individual por método con tabs (para /todos)
  vecinos       → tabla vecinos Gower con clase y clase_vecino
  matrix_info   → info de la matriz de similitud de Gower
"""

import datetime
import numpy as np

# ── Nombres de evaluadores (espejo de la consola) ──────────────
EVALUADOR_NAMES = {
    "correlacion":   "CorrelationAttributeEval",
    "chi2":          "Chi2/F-score SelectKBest",
    "random_forest": "RandomForestClassifier Gini",
    "gower":         "GowerDistanceEval  (inter/intra-class separation ratio)",
    "relief":        "ReliefF  (instancia-based, sin Java)",
    "gain_ratio":    "GainRatioAttributeEval  (nativo Python)",
    "todos":         "Attribute Selection Summary — 6 métodos",
}


# ─────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────

def _build_meta(metodo_key: str, df, nombre_archivo: str,
                eval_time: float, score_values) -> dict:
    """Cabecera tipo Weka: evaluador, relación, instancias, tiempo, merits."""
    vals = [float(v) for v in score_values if v is not None]
    return {
        "evaluador":  EVALUADOR_NAMES.get(metodo_key, metodo_key),
        "relacion":   nombre_archivo,
        "instancias": int(len(df)),
        "atributos":  int(len(df.columns)),
        "timestamp":  datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
        "eval_time":  round(float(eval_time), 4),
        "mean_merit": round(float(np.mean(vals)), 8) if vals else 0.0,
        "max_merit":  round(float(np.max(vals)),  8) if vals else 0.0,
    }


def _build_seleccionadas(resultados: list, campo: str = "score",
                         umbral_pct: float = 50.0) -> list:
    """Lista de variables con merit% ≥ umbral_pct (espejo 'Selected attributes')."""
    if not resultados:
        return []
    max_val = max(float(r.get(campo, 0) or 0) for r in resultados) or 1.0
    return [
        r["variable"]
        for r in resultados
        if (float(r.get(campo, 0) or 0) / max_val) * 100 >= umbral_pct
    ]


def _build_acuerdo(resumen_df, metodos_cols: list) -> dict:
    """
    Construye los 3 grupos de acuerdo entre métodos.
    resumen_df : DataFrame con columnas = métodos + SCORE_FINAL, index = variable.
    """
    n_met = len(metodos_cols)
    acuerdo_total, mayoria, solo_algunos = [], [], []

    for var, row in resumen_df.iterrows():
        score = float(row.get("SCORE_FINAL", 0) or 0)
        votos = 0
        for met in metodos_cols:
            if met not in resumen_df.columns:
                continue
            col_max = float(resumen_df[met].max() or 1.0)
            col_val = float(row.get(met, 0) or 0)
            if col_max > 0 and (col_val / col_max) * 100 >= 50:
                votos += 1

        entry = {"variable": var, "votos": votos,
                 "score": round(score, 4)}
        if votos == n_met:
            acuerdo_total.append(entry)
        elif votos > n_met / 2:
            mayoria.append(entry)
        else:
            solo_algunos.append(entry)

    for lst in (acuerdo_total, mayoria, solo_algunos):
        lst.sort(key=lambda x: x["score"], reverse=True)

    return {
        "metodos_usados": " | ".join(metodos_cols),
        "acuerdo_total":  acuerdo_total,
        "mayoria":        mayoria,
        "solo_algunos":   solo_algunos,
    }


# ─────────────────────────────────────────────────────────────
# API PÚBLICA  (importada desde api.py)
# ─────────────────────────────────────────────────────────────

def enrich_selector(resp: dict, metodo_key: str, df,
                    nombre_archivo: str, eval_time: float,
                    scores_series) -> dict:
    """
    Enriquece el dict de respuesta de cualquier selector simple
    (correlacion, chi2, random_forest, relief, gain_ratio).
    Añade: meta, seleccionadas.
    """
    resultados = resp.get("resultados", [])
    resp["meta"] = _build_meta(
        metodo_key, df, nombre_archivo, eval_time,
        [float(r.get("score", 0) or 0) for r in resultados],
    )
    resp["seleccionadas"] = _build_seleccionadas(resultados, "score")
    return resp


def enrich_gower(resp: dict, df, nombre_archivo: str,
                 eval_time: float, scores_series,
                 max_instancias: int, build_time: float,
                 vecinos_raw: list, target_arr) -> dict:
    """
    Enriquece el dict de respuesta de /api/selector/gower.
    Añade: meta, seleccionadas, matrix_info, vecinos (con clases).
    """
    resultados = resp.get("resultados", [])
    resp["meta"] = _build_meta(
        "gower", df, nombre_archivo, eval_time,
        [float(r.get("score", 0) or 0) for r in resultados],
    )
    resp["seleccionadas"] = _build_seleccionadas(resultados, "score")
    resp["matrix_info"] = {
        "n_instancias":  int(max_instancias),
        "dataset_total": int(len(df)),
        "build_time":    round(float(build_time), 4),
    }
    # Añadir clase y clase_vecino a cada vecino
    enriched = []
    for v in (vecinos_raw or []):
        i  = int(v["instancia"])
        j  = int(v["vecino"])
        cl_i = str(target_arr[i]) if target_arr is not None and i < len(target_arr) else "?"
        cl_j = str(target_arr[j]) if target_arr is not None and j < len(target_arr) else "?"
        enriched.append({
            "instancia":    i,
            "clase":        cl_i,
            "vecino":       j,
            "clase_vecino": cl_j,
            "similitud":    round(float(v["similitud"]), 6),
        })
    resp["vecinos"] = enriched
    return resp


def enrich_todos(resp: dict, df, nombre_archivo: str,
                 eval_time: float, resumen_df,
                 scores_dict: dict) -> dict:
    """
    Enriquece el dict de respuesta de /api/selector/todos.
    Añade: meta, seleccionadas, acuerdo, por_metodo.

    scores_dict : { "correlacion": pd.Series, "chi2": pd.Series, … }
                  con los scores originales (no normalizados) de cada método.
    """
    resultados  = resp.get("resultados", [])
    metodos_cols = [c for c in resumen_df.columns if c != "SCORE_FINAL"]

    # Score finals para meta
    sf_vals = [float(r.get("SCORE_FINAL", r.get("score", 0)) or 0)
               for r in resultados]
    resp["meta"] = _build_meta(
        "todos", df, nombre_archivo, eval_time, sf_vals
    )

    # Seleccionadas
    campo_sf = "SCORE_FINAL" if resultados and "SCORE_FINAL" in resultados[0] else "score"
    resp["seleccionadas"] = _build_seleccionadas(resultados, campo_sf)

    # Acuerdo entre métodos
    resp["acuerdo"] = _build_acuerdo(resumen_df, metodos_cols)

    # Rankings individuales por método → tabs en el frontend
    por_metodo = {}
    for key, series in scores_dict.items():
        if series is None or len(series) == 0:
            continue
        items  = series.sort_values(ascending=False)
        mx_val = float(items.max()) if len(items) > 0 else 1.0
        por_metodo[key] = [
            {
                "variable":  var,
                "score":     round(float(val), 8),
                "merit_pct": round((float(val) / max(mx_val, 1e-9)) * 100, 2),
            }
            for var, val in items.items()
        ]
    resp["por_metodo"] = por_metodo
    return resp
