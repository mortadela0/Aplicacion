"""
metodos/seleccion/comparador.py
────────────────────────────────────────────────────────────
Ejecuta todos los selectores nativos, normaliza y calcula SCORE_FINAL.

Columnas del resumen (6 métodos nativos + Weka_Avg opcional):
  Correlacion | Chi2_F | RandomForest | Gower | ReliefF | GainRatio | [Weka_Avg] | SCORE_FINAL

ReliefF y GainRatio son nativos Python — no requieren Java.
Weka_Avg se añade solo si el usuario lo ejecutó previamente (opción 6).
"""

import pandas as pd
from colorama import Fore
from utils.preprocesamiento import normalizar_serie
from utils.reporte_weka     import imprimir_reporte_comparativo
from . import correlacion, chi2, random_forest, gower, relief, gain_ratio


def comparar_todos(df, target_col, nombre_archivo="dataset.csv",
                   resultados_weka=None, weka_mod=None):
    """
    Ejecuta los 6 selectores nativos y genera tabla SCORE_FINAL.

    Weka_Avg se incluye solo si resultados_weka está disponible
    (el usuario ejecutó la opción 6 previamente).

    Retorna pd.DataFrame con columnas normalizadas + SCORE_FINAL.
    """
    print(f"\n  {Fore.YELLOW}[1/6] Correlación de Pearson...")
    corr = correlacion(df, target_col)

    print(f"\n  {Fore.YELLOW}[2/6] Chi² / F-score...")
    chi  = chi2(df, target_col)

    print(f"\n  {Fore.YELLOW}[3/6] Random Forest Gini...")
    rf   = random_forest(df, target_col)

    print(f"\n  {Fore.YELLOW}[4/6] Gower Distance...")
    gwr  = gower(df, target_col)

    print(f"\n  {Fore.YELLOW}[5/6] ReliefF  (nativo, sin Java)...")
    rel  = relief(df, target_col)

    print(f"\n  {Fore.YELLOW}[6/6] Gain Ratio  (nativo, sin Java)...")
    gr   = gain_ratio(df, target_col)

    # Índice unificado de features
    idx = (corr.index.union(chi.index).union(rf.index)
                .union(gwr.index).union(rel.index).union(gr.index))

    resumen = pd.DataFrame(index=idx)
    resumen["Correlacion"]  = normalizar_serie(corr.reindex(idx).fillna(0))
    resumen["Chi2_F"]       = normalizar_serie(chi.reindex(idx).fillna(0))
    resumen["RandomForest"] = normalizar_serie(rf.reindex(idx).fillna(0))
    resumen["Gower"]        = normalizar_serie(gwr.reindex(idx).fillna(0))
    resumen["ReliefF"]      = normalizar_serie(rel.reindex(idx).fillna(0))
    resumen["GainRatio"]    = normalizar_serie(gr.reindex(idx).fillna(0))

    cols_score = ["Correlacion", "Chi2_F", "RandomForest", "Gower", "ReliefF", "GainRatio"]

    # Weka_Avg opcional — solo si ya se ejecutó la opción 6
    if resultados_weka and resultados_weka.get("disponible") and weka_mod:
        weka_avg = weka_mod.consolidar_scores_weka(resultados_weka, idx)
        resumen["Weka_Avg"] = normalizar_serie(weka_avg)
        cols_score.append("Weka_Avg")
        print(f"\n  {Fore.GREEN}[INFO] Weka_Avg incluido en SCORE_FINAL ({len(cols_score)} métodos).")
    else:
        print(f"\n  {Fore.CYAN}[INFO] SCORE_FINAL con 6 métodos nativos "
              f"(Weka no activo — ejecuta opción 6 para incluirlo).")

    resumen["SCORE_FINAL"] = resumen[cols_score].mean(axis=1)
    resumen = resumen.sort_values("SCORE_FINAL", ascending=False)

    dataset_info = {
        "nombre":   nombre_archivo,
        "filas":    len(df),
        "columnas": len(df.columns),
    }
    imprimir_reporte_comparativo(resumen, dataset_info, target_col)


    _imprimir_coincidencias(resumen, cols_score)
    return resumen


def _imprimir_coincidencias(resumen, cols_metodos):
    from colorama import Fore
    n_met  = len(cols_metodos)
    n_feat = len(resumen)
    top_n  = min(5, n_feat)
    tops = {}
    for met in cols_metodos:
        if met in resumen.columns:
            tops[met] = set(resumen[met].nlargest(top_n).index)
    conteo = {var: sum(1 for t in tops.values() if var in t) for var in resumen.index}
    acuerdo_total   = [v for v,c in conteo.items() if c == n_met]
    acuerdo_mayoria = [v for v,c in conteo.items() if n_met//2 < c < n_met]
    solo_algunos    = [v for v,c in conteo.items() if 1 <= c <= n_met//2]
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.WHITE}  Variables en conjunto — acuerdo entre {n_met} metodos")
    print(f"{Fore.CYAN}{'-'*70}")
    print(f"{Fore.WHITE}  Metodos: {Fore.CYAN}{' | '.join(cols_metodos)}")
    print(f"{Fore.CYAN}{'-'*70}")
    print(f"\n  {Fore.GREEN}ACUERDO TOTAL ({len(acuerdo_total)}) — top 5 en TODOS los metodos:")
    for v in acuerdo_total:
        print(f"  {Fore.GREEN}    * {v:<30} SCORE={resumen.loc[v,'SCORE_FINAL']:.4f}")
    print(f"\n  {Fore.YELLOW}MAYORIA ({len(acuerdo_mayoria)}) — top 5 en mas de la mitad:")
    for v in acuerdo_mayoria:
        print(f"  {Fore.YELLOW}    ~ {v:<30} ({conteo[v]}/{n_met} metodos)  SCORE={resumen.loc[v,'SCORE_FINAL']:.4f}")
    print(f"\n  {Fore.WHITE}SOLO ALGUNOS ({len(solo_algunos)}) — top 5 en mitad o menos:")
    for v in solo_algunos[:5]:
        print(f"  {Fore.WHITE}    . {v:<30} ({conteo[v]}/{n_met} metodos)  SCORE={resumen.loc[v,'SCORE_FINAL']:.4f}")
    print(f"{Fore.CYAN}{'='*70}\n")
