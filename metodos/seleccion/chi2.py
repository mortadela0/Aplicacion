import time
import pandas as pd
from colorama import Fore
from utils.preprocesamiento import preparar_datos, escalar_minmax, target_es_categorico
from utils.reporte_weka     import imprimir_reporte

def ejecutar(df, target_col):
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    t0       = time.time()
    X, y     = preparar_datos(df, target_col)
    Xs       = escalar_minmax(X)
    fn       = f_classif if target_es_categorico(df, target_col) else f_regression
    sel      = SelectKBest(fn, k="all").fit(Xs, y)
    scores   = pd.Series(sel.scores_, index=X.columns).dropna().sort_values(ascending=False)
    imprimir_reporte(scores, "Chi2/F-score SelectKBest", "F-Score",
                     {"nombre": df.attrs.get("nombre","dataset"), "filas": len(df), "columnas": len(df.columns)},
                     target_col, time.time()-t0)
    return scores
