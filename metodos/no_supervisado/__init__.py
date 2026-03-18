from . import hierarchical, kmeans, dbscan, gmm, kmedoids

CATALOGO = {
    "hierarchical": hierarchical,
    "kmeans":       kmeans,
    "dbscan":       dbscan,
    "gmm":          gmm,
    "kmedoids":     kmedoids,
}

ALIAS = {
    "1": "hierarchical",
    "2": "kmeans",
    "3": "dbscan",
    "4": "gmm",
    "5": "kmedoids",
}

def get_clasificador(nombre, params=None, n_clases=2):
    mod = CATALOGO.get(nombre)
    return mod.build(params, n_clases) if mod else None
