import time
import pandas as pd
from colorama import Fore
from utils.preprocesamiento import preparar_datos, target_es_categorico
from utils.reporte_weka     import imprimir_reporte

def ejecutar(df, target_col):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    t0     = time.time()
    X, y   = preparar_datos(df, target_col)
    es_clf = target_es_categorico(df, target_col)
    modelo = (RandomForestClassifier if es_clf else RandomForestRegressor)(
        n_estimators=150, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    scores = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    imprimir_reporte(scores, f"RandomForest{'Classifier' if es_clf else 'Regressor'} Gini",
                     "Gini Imp.",
                     {"nombre": df.attrs.get("nombre","dataset"), "filas": len(df), "columnas": len(df.columns)},
                     target_col, time.time()-t0)
    return scores
