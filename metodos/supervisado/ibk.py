from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOMBRE = "IBk  (KNN Classifier)"
TIPO   = "supervisado"
PARAMS_INFO = {"k": "int (default 3)"}

def build(params=None, n_clases=2):
    p = params or {}
    return Pipeline([
        ("sc", StandardScaler()),
        ("m",  KNeighborsClassifier(n_neighbors=p.get("k", 3))),
    ])
