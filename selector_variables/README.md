# Selector de Variables v2.0

Herramienta de consola para identificar las variables más importantes
de un dataset CSV. Salida formateada estilo Weka Explorer.

## Estructura del proyecto

```
selector_variables/
│
├── selector.py              <- EJECUTAR ESTO
├── validacion.py            <- Validacion de dataset y target
├── requirements.txt         <- Dependencias pip
├── README.md
│
├── metodos/
│   ├── __init__.py
│   ├── correlacion.py       <- Metodo 1: Pearson |r|
│   ├── chi2.py              <- Metodo 2: F-score / SelectKBest
│   ├── random_forest.py     <- Metodo 3: Gini importance
│   ├── gower.py             <- Metodo 4: Distancia de Gower
│   └── weka_attsel.py       <- Metodo 5: Weka native evaluators
│
├── utils/
│   ├── __init__.py
│   ├── consola.py           <- Banner, menu, helpers UI
│   ├── preprocesamiento.py  <- Encoding, imputacion, escalado
│   └── reporte_weka.py      <- Formateador salida estilo Weka
│
├── datos/                   <- [auto] Coloca aqui tus CSV
└── resultados/              <- [auto] Exportaciones CSV
```

## Instalacion y uso

```bash
# 1. Instalar dependencias base
pip install -r requirements.txt

# 2. (Opcional) Instalar Weka API — requiere Java JDK 8+
pip install python-weka-wrapper3 jpype1

# 3. Ejecutar
python selector.py
```

## Menu

```
[1]  Cargar CSV
[2]  Correlacion de Pearson      (numericas lineales)
[3]  Chi2 / F-score              (SelectKBest ANOVA)
[4]  Random Forest               (Gini importance)
[5]  Distancia de Gower          (datos mixtos, sin encoding)
[6]  Weka evaluators             (InfoGain/GainRatio/ReliefF/CFS)
[7]  Comparar TODOS              (SCORE_FINAL unificado)
[8]  Exportar resultados a CSV
[0]  Salir
```

## Los 5 metodos

| # | Metodo | Archivo | Tipo | Nota |
|---|---|---|---|---|
| 1 | Correlacion Pearson | correlacion.py | Numericas | Relaciones lineales |
| 2 | Chi2 / F-score | chi2.py | Cualquiera | Significancia estadistica |
| 3 | Random Forest | random_forest.py | Mixto | Relaciones no lineales |
| 4 | Gower Distance | gower.py | Mixto | Sin encoding previo |
| 5 | Weka evaluators | weka_attsel.py | Cualquiera | Requiere Java + weka lib |

## Validaciones automaticas

Al cargar el CSV se ejecutan:
- Dimensiones minimas (filas y columnas)
- % de valores nulos por columna
- Columnas sin varianza (constantes)
- Filas duplicadas
- Alta cardinalidad en categoricas
- Desbalance de clases en el target

## Opcion 7 — Comparacion completa

Normaliza todos los metodos a [0,1] y calcula SCORE_FINAL.
Si Weka esta activo, incluye Weka_Avg en el calculo.
Genera reporte de cross-validation: own methods vs Weka.

```
CONFIRMED   -> top-5 en metodos propios Y en Weka  (usar con confianza)
own only    -> solo en propios (revisar)
weka only   -> solo en Weka (considerar)
CFS subset  -> subconjunto optimo de Weka (el mas conservador)
RECOMMENDED -> union de CONFIRMED + CFS subset
```

## Exportacion (opcion 8)

Guarda en resultados/ un CSV con columnas:
Correlacion | Chi2_F | RandomForest | Gower | [Weka_Avg] | SCORE_FINAL
