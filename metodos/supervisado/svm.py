from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOMBRE = "SVM  (Support Vector Machine)"
TIPO   = "supervisado"
PARAMS_INFO = {"kernel": "rbf|linear|poly", "C": "float (default 1.0)"}

def build(params=None, n_clases=2):
    p = params or {}
    return Pipeline([
        ("sc", StandardScaler()),
        ("m",  SVC(
            kernel=p.get("kernel", "rbf"),
            C=p.get("C", 1.0),
            gamma=p.get("gamma", "scale"),
            probability=True,
            random_state=42,
        )),
    ])
