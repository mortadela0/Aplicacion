from .base import ClusterClassifier
from sklearn.cluster import AgglomerativeClustering

NOMBRE = "Hierarchical Clustering Classifier"
TIPO   = "no_supervisado"
PARAMS_INFO = {"linkage": "ward|complete|average (default ward)", "n_clusters": "int | None (auto)"}


class HierarchicalClassifier(ClusterClassifier):
    def __init__(self, n_clusters=None, linkage="ward"):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage    = linkage

    def _ejecutar_cluster(self, Xs):
        k  = self.n_clusters or len(self.classes_)
        hc = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
        return hc.fit_predict(Xs)

    def get_params(self, deep=True):
        return {"n_clusters": self.n_clusters, "linkage": self.linkage}


def build(params=None, n_clases=2):
    p = params or {}
    return HierarchicalClassifier(
        n_clusters=p.get("n_clusters", n_clases),
        linkage=p.get("linkage", "ward"),
    )
