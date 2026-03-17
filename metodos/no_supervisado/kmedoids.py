"""
KMedoids Classifier (PAM).
Usa scikit-learn-extra si esta instalado.
Si no, implementacion propia simplificada (mas lenta pero sin dependencias extra).
"""
import numpy as np
from .base import ClusterClassifier

NOMBRE = "KMedoids  (PAM)"
TIPO   = "no_supervisado"
PARAMS_INFO = {
    "n_clusters": "int | None (auto)",
    "metric": "euclidean|cosine|manhattan (default euclidean)",
}


class KMedoidsClassifier(ClusterClassifier):
    def __init__(self, n_clusters=None, metric="euclidean"):
        super().__init__()
        self.n_clusters = n_clusters
        self.metric     = metric

    def _ejecutar_cluster(self, Xs):
        k = self.n_clusters or len(self.classes_)
        try:
            from sklearn_extra.cluster import KMedoids
            km = KMedoids(n_clusters=k, metric=self.metric, random_state=42)
            labels = km.fit_predict(Xs)
            # Usar medoides reales como centroides
            for i, med in enumerate(km.cluster_centers_):
                self._centroides[i] = med
            return labels
        except ImportError:
            return self._kmedoids_simple(Xs, k)

    def _kmedoids_simple(self, Xs, k):
        """PAM simplificado: inicializacion aleatoria + 1 swap pass."""
        rng     = np.random.default_rng(42)
        medoids = rng.choice(len(Xs), k, replace=False)

        for _ in range(100):
            dists  = np.linalg.norm(Xs[:, np.newaxis] - Xs[medoids][np.newaxis], axis=2)
            labels = np.argmin(dists, axis=1)
            new_medoids = medoids.copy()
            for ci in range(k):
                mask = labels == ci
                if not mask.any():
                    continue
                idxs    = np.where(mask)[0]
                sub     = Xs[idxs]
                intra   = np.sum(np.linalg.norm(sub[:, np.newaxis] - sub[np.newaxis], axis=2), axis=1)
                new_medoids[ci] = idxs[np.argmin(intra)]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        for i, m in enumerate(medoids):
            self._centroides[i] = Xs[m]
        dists  = np.linalg.norm(Xs[:, np.newaxis] - Xs[medoids][np.newaxis], axis=2)
        return np.argmin(dists, axis=1)

    def get_params(self, deep=True):
        return {"n_clusters": self.n_clusters, "metric": self.metric}


def build(params=None, n_clases=2):
    p = params or {}
    return KMedoidsClassifier(
        n_clusters=p.get("n_clusters", n_clases),
        metric=p.get("metric", "euclidean"),
    )
