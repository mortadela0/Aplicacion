from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOMBRE = "SGD  (Stochastic Gradient Descent)"
TIPO   = "supervisado"
PARAMS_INFO = {"loss": "modified_huber|hinge|log_loss", "max_iter": "int (default 1000)"}

def build(params=None, n_clases=2):
    p = params or {}
    return Pipeline([
        ("sc", StandardScaler()),
        ("m",  SGDClassifier(
            loss=p.get("loss", "modified_huber"),
            max_iter=p.get("max_iter", 1000),
            random_state=42,
        )),
    ])
