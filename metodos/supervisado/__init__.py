from . import j48, naive_bayes, ibk, svm, sgd

CATALOGO = {
    "j48":         j48,
    "naive_bayes": naive_bayes,
    "ibk":         ibk,
    "svm":         svm,
    "sgd":         sgd,
}

ALIAS = {"1": "j48", "2": "naive_bayes", "3": "ibk", "4": "svm", "5": "sgd"}

def get_clasificador(nombre, params=None, n_clases=2):
    mod = CATALOGO.get(nombre)
    return mod.build(params, n_clases) if mod else None
