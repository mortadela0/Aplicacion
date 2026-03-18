"""
metodos/no_supervisado/base.py
Clase base para clasificadores no supervisados adaptados.
Patron comun: fit → cluster → mapa cluster/clase → centroide
              predict → centroide mas cercano
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class ClusterClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptador generico: algoritmo de clustering → clasificador sklearn.

    Subclases deben implementar _ejecutar_cluster(Xs) → array de etiquetas
    y opcionalmente _centroides(Xs, labels) si el algoritmo no los provee.
    """

    def __init__(self):
        self._scaler     = None
        self._mapa       = None
        self._centroides = None
        self.classes_    = None
        self._X_train    = None   # fallback KNN para DBSCAN/GMM

    # ── Metodos que las subclases deben sobreescribir ─────────────────────

    def _ejecutar_cluster(self, Xs):
        raise NotImplementedError

    # ── Fit comun ─────────────────────────────────────────────────────────

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._X_train = Xs.copy()

        labels = self._ejecutar_cluster(Xs)
        self._construir_mapa(y, labels, Xs)
        return self

    def _construir_mapa(self, y, labels, Xs):
        unicos = np.unique(labels)
        self._mapa      = {}
        self._centroides = {}

        # Clase mayoritaria por cluster
        for lab in unicos:
            mask = labels == lab
            if not mask.any():
                continue
            clases, counts = np.unique(y[mask], return_counts=True)
            self._mapa[int(lab)] = clases[np.argmax(counts)]
            self._centroides[int(lab)] = Xs[mask].mean(axis=0)

        # Ruido DBSCAN (-1): clase mas frecuente global
        if -1 in self._mapa:
            vals = [v for k, v in self._mapa.items() if k != -1]
            if vals:
                from collections import Counter
                self._mapa[-1] = Counter(vals).most_common(1)[0][0]

    # ── Predict comun ─────────────────────────────────────────────────────

    def predict(self, X):
        X  = np.asarray(X, dtype=float)
        Xs = self._scaler.transform(X)
        default = self.classes_[0]

        if not self._centroides:
            return np.full(len(X), default)

        centroid_arr = np.array(list(self._centroides.values()))
        centroid_keys = list(self._centroides.keys())

        dists   = np.linalg.norm(Xs[:, np.newaxis] - centroid_arr[np.newaxis], axis=2)
        nearest = np.argmin(dists, axis=1)
        return np.array([self._mapa.get(centroid_keys[i], default) for i in nearest])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self
