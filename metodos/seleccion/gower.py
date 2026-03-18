"""
metodos/seleccion/gower.py
────────────────────────────────────────────────────────────
Distancia y Similitud de Gower — implementación completa.

Características:
  • Pesos específicos por variable (w_num / w_cat configurables)
  • Matriz de similitud para datasets pequeños (≤ 30 instancias)
  • Señala el vecino más cercano (mayor similitud) por instancia
  • Métrica de la cruz: muestra SOLO similitud (no distancia)
    — la distancia se calcula internamente pero no se expone al usuario
  • Selector de variables: ratio inter/intra clase × contribución media
  • Subsampling automático si n > 2000 (Gower es O(n²))

Distancia por variable:
  Numérica    → |xi − xj| / rango        normalizada a [0,1]
  Categórica  → 0 si iguales, 1 si distintos

Similitud = 1 − Distancia
"""

import time
import numpy as np
import pandas as pd
from colorama import Fore
from utils.reporte_weka import imprimir_reporte


# ─── NÚCLEO GOWER ─────────────────────────────────────────────────────────

def _distancia_gower(df_feat: pd.DataFrame,
                     weights: dict | None = None) -> np.ndarray:
    """
    Construye la matriz de distancia de Gower N×N con pesos opcionales.

    Parameters
    ----------
    df_feat : DataFrame sin la columna target.
    weights : dict {col: peso_float}. Si None todos los pesos = 1.

    Returns
    -------
    np.ndarray (N, N) de distancias en [0, 1].
    """
    n        = len(df_feat)
    num_cols = df_feat.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_feat.select_dtypes(include=["object", "category"]).columns.tolist()
    todas    = num_cols + cat_cols

    if not todas:
        return np.zeros((n, n))

    # Rangos para variables numéricas
    rangos = {}
    for col in num_cols:
        r = df_feat[col].max() - df_feat[col].min()
        rangos[col] = r if r > 0 else 1.0

    # Pesos normalizados
    w_total = sum((weights or {}).get(c, 1.0) for c in todas)
    w = {c: (weights or {}).get(c, 1.0) / w_total for c in todas}

    idx_i, idx_j = np.triu_indices(n, k=1)
    sum_w  = np.zeros(len(idx_i))
    sum_wd = np.zeros(len(idx_i))

    for col in num_cols:
        vals     = df_feat[col].fillna(df_feat[col].median()).values.astype(float)
        d_col    = np.abs(vals[idx_i] - vals[idx_j]) / rangos[col]
        sum_wd  += w[col] * d_col
        sum_w   += w[col]

    for col in cat_cols:
        vals     = df_feat[col].astype(str).values
        d_col    = (vals[idx_i] != vals[idx_j]).astype(float)
        sum_wd  += w[col] * d_col
        sum_w   += w[col]

    dists_flat = np.where(sum_w > 0, sum_wd / sum_w, 0.0)

    mat = np.zeros((n, n))
    mat[idx_i, idx_j] = dists_flat
    mat[idx_j, idx_i] = dists_flat
    return mat


# ─── IMPORTANCIA POR GOWER ────────────────────────────────────────────────

def importancia_por_gower(df: pd.DataFrame, target_col: str,
                          weights: dict | None = None) -> pd.Series:
    """
    Score = ratio(dist_inter_clase / dist_intra_clase) × contribución_media.
    """
    features = df.drop(columns=[target_col])
    target   = df[target_col].astype(str).values
    n        = len(features)
    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    todas    = num_cols + cat_cols

    rangos = {c: max(features[c].max() - features[c].min(), 1e-9) for c in num_cols}
    idx_i, idx_j = np.triu_indices(n, k=1)
    inter_mask   = target[idx_i] != target[idx_j]
    intra_mask   = ~inter_mask
    resultados   = {}

    for col in todas:
        wi = (weights or {}).get(col, 1.0)
        if col in num_cols:
            vals     = features[col].fillna(features[col].median()).values.astype(float)
            dist_col = np.abs(vals[idx_i] - vals[idx_j]) / rangos[col]
        else:
            vals     = features[col].astype(str).values
            dist_col = (vals[idx_i] != vals[idx_j]).astype(float)

        d_inter      = dist_col[inter_mask].mean() if inter_mask.any() else 0.0
        d_intra      = dist_col[intra_mask].mean() if intra_mask.any() else 0.0
        ratio        = d_inter / (d_intra + 1e-9)
        contribucion = dist_col.mean()
        resultados[col] = wi * ratio * contribucion

    return pd.Series(resultados).sort_values(ascending=False)


# ─── MÉTRICA DE LA CRUZ — SOLO SIMILITUD ─────────────────────────────────

def _imprimir_cruz_similitud(sim_mat: np.ndarray, etiquetas: list,
                              target: np.ndarray | None = None):
    """
    Imprime la 'métrica de la cruz' centrada en cada instancia:
    para la instancia i muestra su fila + columna de similitud
    (no la distancia — solo similitud = 1 − distancia).

    Formato visual:
        Instancia i    →   similitud con j1, j2, j3 ...
        Vecino más cercano marcado con ★
    """
    n = len(etiquetas)
    print(f"\n{Fore.CYAN}{'═'*70}")
    print(f"{Fore.WHITE}  Métrica de la Cruz — Similitud de Gower  (1 = idéntico, 0 = opuesto)")
    print(f"{Fore.CYAN}{'─'*70}")

    col_w = max(max(len(e) for e in etiquetas), 6)

    # Encabezado
    header = f"  {Fore.WHITE}{'':>{col_w}}"
    for e in etiquetas:
        header += f"  {e:>{col_w}}"
    print(header)
    print(f"{Fore.CYAN}  {'─'*(col_w + (col_w+2)*n + 2)}")

    for i, e_i in enumerate(etiquetas):
        row = f"  {Fore.WHITE}{e_i:>{col_w}}"
        # Vecino más cercano (excluye diagonal)
        sims  = sim_mat[i].copy()
        sims[i] = -1
        vecino  = int(np.argmax(sims))

        for j, e_j in enumerate(etiquetas):
            v = sim_mat[i, j]
            if i == j:
                color = Fore.WHITE; txt = f"{'—':>{col_w}}"
            elif j == vecino:
                color = Fore.GREEN; txt = f"{v:>{col_w}.4f}"
            elif v >= 0.7:
                color = Fore.YELLOW; txt = f"{v:>{col_w}.4f}"
            else:
                color = Fore.WHITE; txt = f"{v:>{col_w}.4f}"
            row += f"  {color}{txt}"

        # Marca el vecino
        lbl_vecino = etiquetas[vecino]
        clase_i = f"[{target[i]}]" if target is not None else ""
        clase_v = f"[{target[vecino]}]" if target is not None else ""
        row += f"  {Fore.GREEN}★ → {lbl_vecino}{clase_v}  {Fore.CYAN}{clase_i}"
        print(row)

    print(f"{Fore.CYAN}{'═'*70}")
    print(f"{Fore.GREEN}  ★ = vecino más cercano   "
          f"{Fore.YELLOW}Amarillo ≥ 0.70   {Fore.WHITE}— = misma instancia\n")


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────

def ejecutar(df: pd.DataFrame, target_col: str,
             weights: dict | None = None,
             max_matriz: int = 30) -> pd.Series:
    """
    Ejecuta el análisis de Gower completo:
      1. Selector de variables (importancia inter/intra clase)
      2. Matriz de similitud con métrica de la cruz (≤ max_matriz instancias)
      3. Vecino más cercano señalado con ★

    Parameters
    ----------
    weights     : dict {col_name: peso}. None = pesos iguales.
    max_matriz  : int. Número máximo de instancias para mostrar la matriz.
                  Default 30. Si el dataset tiene más, toma una muestra aleatoria.
    """
    n          = len(df)
    df_trabajo = df.copy()

    # Subsampling si el dataset es grande
    if n > 2000:
        print(f"\n  {Fore.YELLOW}[INFO] Dataset grande ({n} filas). "
              f"Subsampling a 2000 para calcular importancias...")
        df_trabajo = df.sample(n=2000, random_state=42).reset_index(drop=True)

    num_cols = df_trabajo.drop(columns=[target_col]).select_dtypes(include=["number"]).columns
    cat_cols = df_trabajo.drop(columns=[target_col]).select_dtypes(include=["object","category"]).columns

    print(f"\n  {Fore.YELLOW}[INFO] GowerDistanceEval — calculando importancias...")
    print(f"  {Fore.CYAN}  Numéricas  : {Fore.WHITE}{len(num_cols)}"
          f"   {Fore.CYAN}Categóricas: {Fore.WHITE}{len(cat_cols)}")
    if weights:
        print(f"  {Fore.CYAN}  Pesos      : {Fore.WHITE}"
              + "  ".join(f"{k}={v:.2f}" for k,v in weights.items()))

    t0           = time.time()
    importancias = importancia_por_gower(df_trabajo, target_col, weights)
    elapsed      = time.time() - t0

    dataset_info = {
        "nombre":   df.attrs.get("nombre", "dataset"),
        "filas":    n,
        "columnas": len(df.columns),
    }
    imprimir_reporte(
        importancias,
        "GowerDistanceEval  (inter/intra-class separation ratio)",
        "Gower Score", dataset_info, target_col,
        elapsed, decimales=10, umbral_pct=50.0,
    )

    # Tipo de variable por atributo
    print(f"{Fore.WHITE}  Tipo de variable:")
    print(f"{Fore.CYAN}  {'─'*50}")
    for attr in importancias.index:
        tipo  = "Numeric" if attr in num_cols else "Categorical"
        color = Fore.CYAN if tipo == "Numeric" else Fore.YELLOW
        w_val = (weights or {}).get(attr, 1.0)
        print(f"  {color}  {attr:<30} {tipo:<12} peso={w_val:.2f}")
    print(f"{Fore.CYAN}  {'─'*50}\n")

    # ── Matriz de similitud (métrica de la cruz) ──────────────────────────
    # Tomar muestra de max_matriz instancias para la matriz visual
    n_mat = min(n, max_matriz)
    if n > max_matriz:
        print(f"  {Fore.YELLOW}[INFO] Mostrando matriz para {n_mat} instancias "
              f"(dataset tiene {n})...")
        df_mat = df.sample(n=n_mat, random_state=0).reset_index(drop=True)
    else:
        df_mat = df.reset_index(drop=True)

    features_mat = df_mat.drop(columns=[target_col])
    target_mat   = df_mat[target_col].astype(str).values
    etiquetas    = [f"I{i}" for i in range(n_mat)]

    print(f"  {Fore.YELLOW}[INFO] Construyendo matriz de similitud "
          f"({n_mat}×{n_mat})...")
    t1      = time.time()
    dist_mat = _distancia_gower(features_mat, weights)
    sim_mat  = 1.0 - dist_mat
    np.fill_diagonal(sim_mat, 1.0)
    t2 = time.time()
    print(f"  {Fore.GREEN}[OK] Matriz construida en {t2-t1:.3f}s")

    # Imprimir métrica de la cruz (solo similitud)
    _imprimir_cruz_similitud(sim_mat, etiquetas, target_mat)

    # Resumen de vecinos más cercanos
    print(f"{Fore.CYAN}{'─'*70}")
    print(f"{Fore.WHITE}  Vecinos más cercanos por instancia:")
    print(f"{Fore.CYAN}  {'─'*70}")
    print(f"{Fore.WHITE}  {'Instancia':<12} {'Clase':<10} {'Vecino más cercano':<16} "
          f"{'Clase vecino':<14} {'Similitud':>10}")
    print(f"{Fore.CYAN}  {'─'*70}")

    for i in range(n_mat):
        sims_i        = sim_mat[i].copy()
        sims_i[i]     = -1
        vecino_idx    = int(np.argmax(sims_i))
        sim_val       = sim_mat[i, vecino_idx]
        clase_i       = target_mat[i]
        clase_v       = target_mat[vecino_idx]
        misma_clase   = clase_i == clase_v
        color = Fore.GREEN if misma_clase else Fore.YELLOW
        print(f"  {color}  I{i:<10} {clase_i:<10} I{vecino_idx:<15} "
              f"{clase_v:<14} {sim_val:>10.6f}"
              + (f"  {Fore.CYAN}(misma clase)" if misma_clase else ""))

    print(f"{Fore.CYAN}{'─'*70}\n")

    return importancias


# ─── CONFIGURAR PESOS INTERACTIVO ────────────────────────────────────────

def configurar_pesos(df: pd.DataFrame, target_col: str) -> dict | None:
    """
    Sub-menú interactivo para asignar pesos específicos a cada variable.
    El usuario puede dejar en blanco para usar peso 1.0 (default).
    Retorna dict de pesos o None si todos son 1.0.
    """
    features = [c for c in df.columns if c != target_col]
    print(f"\n{Fore.CYAN}  Configurar pesos de Gower")
    print(f"  {Fore.WHITE}  (Enter = peso 1.0 para todas las variables)")
    print(f"  {Fore.CYAN}  {'─'*50}")

    usar_pesos = input(
        f"  {Fore.WHITE}  ¿Asignar pesos específicos? (s/N): {Fore.GREEN}"
    ).strip().lower()

    if usar_pesos != "s":
        return None

    pesos = {}
    print(f"  {Fore.CYAN}  Escribe el peso para cada variable (Enter = 1.0):")
    for col in features:
        val = input(f"  {Fore.WHITE}    {col:<30}: {Fore.GREEN}").strip()
        try:
            pesos[col] = float(val) if val else 1.0
        except ValueError:
            pesos[col] = 1.0

    # Mostrar resumen
    print(f"\n  {Fore.CYAN}  Pesos configurados:")
    for col, w in pesos.items():
        color = Fore.GREEN if w > 1.0 else Fore.YELLOW if w < 1.0 else Fore.WHITE
        print(f"    {color}{col:<30} = {w:.2f}")

    return pesos if any(v != 1.0 for v in pesos.values()) else None
