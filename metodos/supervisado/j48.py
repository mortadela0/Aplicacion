from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOMBRE = "J48  (Decision Tree, entropy)"
TIPO   = "supervisado"
PARAMS_INFO = {"max_depth": "int | None", "min_samples_leaf": "int (default 1)"}

def build(params=None, n_clases=2):
    p = params or {}
    return Pipeline([
        ("sc", StandardScaler()),
        ("m",  DecisionTreeClassifier(
            criterion="entropy",
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            random_state=42,
        )),
    ])
