from .base import ClusterClassifier
from sklearn.cluster import KMeans

NOMBRE = "KMeans Clustering Classifier"
TIPO   = "no_supervisado"
PARAMS_INFO = {"n_clusters": "int | None (auto)", "max_iter": "int (default 300)"}


class KMeansClassifier(ClusterClassifier):
    def __init__(self, n_clusters=None, max_iter=300):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter   = max_iter

    def _ejecutar_cluster(self, Xs):
        k   = self.n_clusters or len(self.classes_)
        km  = KMeans(n_clusters=k, max_iter=self.max_iter,
                     random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        # Sobreescribir centroides con los del modelo (mas precisos)
        import numpy as np
        self._centroides = {i: km.cluster_centers_[i] for i in range(k)}
        return labels

    def get_params(self, deep=True):
        return {"n_clusters": self.n_clusters, "max_iter": self.max_iter}


def build(params=None, n_clases=2):
    p = params or {}
    return KMeansClassifier(
        n_clusters=p.get("n_clusters", n_clases),
        max_iter=p.get("max_iter", 300),
    )
