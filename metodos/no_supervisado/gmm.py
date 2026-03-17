"""
Gaussian Mixture Model Classifier.
Usa predict_proba nativo de GMM para asignar la clase de mayor probabilidad.
El mapeo cluster→clase se hace igual que los otros: clase mayoritaria por componente.
"""
import numpy as np
from .base import ClusterClassifier
from sklearn.mixture import GaussianMixture

NOMBRE = "GMM  (Gaussian Mixture Model)"
TIPO   = "no_supervisado"
PARAMS_INFO = {
    "n_components": "int | None (auto)",
    "covariance_type": "full|tied|diag|spherical (default full)",
}


class GMMClassifier(ClusterClassifier):
    def __init__(self, n_components=None, covariance_type="full"):
        super().__init__()
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self._gmm            = None

    def _ejecutar_cluster(self, Xs):
        k = self.n_components or len(self.classes_)
        self._gmm = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            random_state=42,
        )
        return self._gmm.fit_predict(Xs)

    def predict(self, X):
        X  = np.asarray(X, dtype=float)
        Xs = self._scaler.transform(X)
        labels = self._gmm.predict(Xs)
        default = self.classes_[0]
        return np.array([self._mapa.get(int(l), default) for l in labels])

    def predict_proba(self, X):
        X  = np.asarray(X, dtype=float)
        Xs = self._scaler.transform(X)
        return self._gmm.predict_proba(Xs)

    def get_params(self, deep=True):
        return {"n_components": self.n_components,
                "covariance_type": self.covariance_type}


def build(params=None, n_clases=2):
    p = params or {}
    return GMMClassifier(
        n_components=p.get("n_components", n_clases),
        covariance_type=p.get("covariance_type", "full"),
    )
