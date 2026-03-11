"""
metodos/ibk.py
─────────────────────────────────────────────
METODO 6 - IBk — k Nearest Neighbors optimizado

• Busca el mejor K automáticamente
• Usa 10-Fold Cross Validation
• Accuracy en porcentaje
• Matriz de confusión
"""

import time
import pandas as pd
import numpy as np
from colorama import Fore

from utils.preprocesamiento import preparar_datos, target_es_categorico


def ejecutar(df: pd.DataFrame, target_col: str):

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix

    if target_col is None or target_col not in df.columns:
        print(f"{Fore.RED}[ERROR] Columna target inválida")
        return

    t0 = time.time()

    # ───── Preparar datos ─────
    X, y = preparar_datos(df, target_col)

    if not target_es_categorico(df, target_col):
        print(f"{Fore.RED}[ERROR] IBk solo funciona con clasificación")
        return

    # ───── Escalar datos ─────
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\n{Fore.YELLOW}[INFO] Buscando mejor valor de K...")

    mejor_k = 1
    mejor_acc = 0

    # probar muchos valores de K
    for k in range(1, 31):

        modelo = KNeighborsClassifier(n_neighbors=k)

        scores = cross_val_score(
            modelo,
            X,
            y,
            cv=10,
            scoring="accuracy"
        )

        acc = scores.mean()

        if acc > mejor_acc:
            mejor_acc = acc
            mejor_k = k

    print(f"{Fore.GREEN}[OK] Mejor K encontrado: {mejor_k}")

    # ───── Modelo final ─────
    modelo = KNeighborsClassifier(n_neighbors=mejor_k)

    y_pred = cross_val_predict(
        modelo,
        X,
        y,
        cv=10
    )

    matriz = confusion_matrix(y, y_pred)

    elapsed = time.time() - t0

    accuracy_pct = mejor_acc * 100

    # ───── Resultados ─────
    print(f"\n{Fore.CYAN}──────── RESULTADOS IBk ────────")
    print(f"{Fore.WHITE}Accuracy promedio: {Fore.GREEN}{accuracy_pct:.2f}%")
    print(f"{Fore.WHITE}Tiempo ejecución : {elapsed:.2f} s")

    print(f"\n{Fore.CYAN}Matriz de confusión:")

    clases = np.unique(y)

    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real_{c}" for c in clases],
        columns=[f"Pred_{c}" for c in clases]
    )

    print(matriz_df.to_string())

    print(f"\n{Fore.GREEN}[OK] Evaluación completada.\n")