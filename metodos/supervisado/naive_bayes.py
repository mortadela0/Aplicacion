from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOMBRE = "Naive Bayes  (GaussianNB)"
TIPO   = "supervisado"
PARAMS_INFO = {}

def build(params=None, n_clases=2):
    return Pipeline([("sc", StandardScaler()), ("m", GaussianNB())])
