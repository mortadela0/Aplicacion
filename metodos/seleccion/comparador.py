"""
metodos/seleccion/comparador.py
Ejecuta todos los selectores, normaliza y calcula SCORE_FINAL.
"""
import pandas as pd
from utils.preprocesamiento import normalizar_serie
from utils.reporte_weka     import imprimir_reporte_comparativo
from . import correlacion, chi2, random_forest, gower

def comparar_todos(df, target_col, nombre_archivo="dataset.csv",
                   resultados_weka=None, weka_mod=None):
    corr = correlacion(df, target_col)
    chi  = chi2(df, target_col)
    rf   = random_forest(df, target_col)
    gwr  = gower(df, target_col)

    idx = corr.index.union(chi.index).union(rf.index).union(gwr.index)
    resumen = pd.DataFrame(index=idx)
    resumen["Correlacion"]  = normalizar_serie(corr.reindex(idx).fillna(0))
    resumen["Chi2_F"]       = normalizar_serie(chi.reindex(idx).fillna(0))
    resumen["RandomForest"] = normalizar_serie(rf.reindex(idx).fillna(0))
    resumen["Gower"]        = normalizar_serie(gwr.reindex(idx).fillna(0))

    cols = ["Correlacion","Chi2_F","RandomForest","Gower"]
    if resultados_weka and resultados_weka.get("disponible") and weka_mod:
        resumen["Weka_Avg"] = weka_mod.consolidar_scores_weka(resultados_weka, idx)
        cols.append("Weka_Avg")

    resumen["SCORE_FINAL"] = resumen[cols].mean(axis=1)
    resumen = resumen.sort_values("SCORE_FINAL", ascending=False)

    imprimir_reporte_comparativo(
        resumen,
        {"nombre": nombre_archivo, "filas": len(df), "columnas": len(df.columns)},
        target_col,
    )
    return resumen
