# Selector de Variables v3.0

## Estructura
```
selector_variables/
├── selector.py              <- CLI  (python selector.py)
├── validacion.py
├── requirements.txt
├── metodos/
│   ├── supervisado/         <- J48, NB, KNN, SVM, SGD
│   ├── no_supervisado/      <- Hierarchical, KMeans, DBSCAN, GMM, KMedoids
│   └── seleccion/           <- Correlacion, Chi2, RF, Gower, Weka, Comparador
├── utils/
│   ├── preprocesamiento.py
│   ├── consola.py
│   ├── reporte_weka.py
│   ├── exportar.py
│   ├── entrenador.py        <- Train/eval unificado con salida Weka
│   └── predictor.py         <- Guardar/cargar modelos + predecir CSV nuevo
├── web/
│   ├── app.py               <- Web (python web/app.py)
│   ├── routes/api.py
│   ├── templates/index.html
│   └── static/css/ js/
├── modelos/                 <- Modelos .pkl guardados
├── datos/                   <- CSV de entrada
└── resultados/              <- Exportaciones
```

## Uso CLI
```bash
pip install -r requirements.txt
python selector.py
```

## Uso Web
```bash
python web/app.py
# Abrir http://localhost:5000
```

## Clasificadores
| # | Nombre | Tipo |
|---|---|---|
| 1 | J48 Decision Tree | supervisado |
| 2 | Naive Bayes | supervisado |
| 3 | IBk KNN | supervisado |
| 4 | SVM | supervisado |
| 5 | SGD | supervisado |
| 6 | Hierarchical Clustering | no supervisado |
| 7 | KMeans | no supervisado |
| 8 | DBSCAN | no supervisado |
| 9 | GMM Gaussian Mixture | no supervisado |
| 10 | KMedoids PAM | no supervisado |
