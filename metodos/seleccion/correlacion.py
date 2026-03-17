import time
import pandas as pd
from utils.preprocesamiento import preparar_datos
from utils.reporte_weka     import imprimir_reporte

def ejecutar(df, target_col):
    t0       = time.time()
    X, y     = preparar_datos(df, target_col)
    y_series = pd.Series(y, index=X.index, name=target_col)
    scores   = X.corrwith(y_series).abs().dropna().sort_values(ascending=False)
    imprimir_reporte(scores, "CorrelationAttributeEval", "Pearson |r|",
                     {"nombre": df.attrs.get("nombre","dataset"), "filas": len(df), "columnas": len(df.columns)},
                     target_col, time.time()-t0)
    return scores
