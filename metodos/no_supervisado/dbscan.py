"""
DBSCAN Clustering Classifier.
Prediccion: los puntos nuevos se asignan al cluster cuyo punto de
entrenamiento mas cercano les corresponde (KNN 1-vecino sobre X_train).
Puntos outlier en entrenamiento reciben la clase mayoritaria global.
"""
import numpy as np
from .base import ClusterClassifier
from sklearn.cluster import DBSCAN

NOMBRE = "DBSCAN Clustering Classifier"
TIPO   = "no_supervisado"
PARAMS_INFO = {"eps": "float (default 0.5)", "min_samples": "int (default 5)"}


class DBSCANClassifier(ClusterClassifier):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps         = eps
        self.min_samples = min_samples
        self._y_train    = None   # etiquetas por punto de entrenamiento

    def _ejecutar_cluster(self, Xs):
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return db.fit_predict(Xs)

    def fit(self, X, y):
        super().fit(X, y)
        # Guardar etiquetas para KNN en predict
        self._y_train = np.asarray(y)
        return self

    def predict(self, X):
        """KNN-1 sobre X_train escalado."""
        X  = np.asarray(X, dtype=float)
        Xs = self._scaler.transform(X)
        dists   = np.linalg.norm(Xs[:, np.newaxis] - self._X_train[np.newaxis], axis=2)
        nearest = np.argmin(dists, axis=1)
        return self._y_train[nearest]

    def get_params(self, deep=True):
        return {"eps": self.eps, "min_samples": self.min_samples}


def build(params=None, n_clases=2):
    p = params or {}
    return DBSCANClassifier(
        eps=p.get("eps", 0.5),
        min_samples=p.get("min_samples", 5),
    )
