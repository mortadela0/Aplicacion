"""
utils/dataset_editor.py
────────────────────────────────────────────────────────────
Edición interactiva del dataset: eliminar variables y guardar
un nuevo CSV con las columnas restantes.

Flujos disponibles:
  menu_eliminar_por_scores(df, target_col, scores, nombre_metodo)
      → Llamado después de un selector individual (opciones 2-5).
        Muestra ranking, recomienda las últimas N variables
        para eliminar y permite selección una a una.

  menu_eliminar_por_score_final(df, target_col, resumen)
      → Llamado después de Comparar Todos (opción 7).
        Usa SCORE_FINAL para la recomendación.

  guardar_dataset(df, nombre_sugerido)
      → Guarda el df en datos/ con el nombre elegido por el usuario.

Ambos menus retornan el df modificado (o el original si el usuario
cancela), para que selector.py actualice su estado.
"""

import os
import pandas as pd
import numpy as np
from colorama import Fore, init
init(autoreset=True)

ANCHO = 70

def _linea(c="-"):
    return c * ANCHO

# Carpeta fija en el home del usuario del sistema operativo.
# Siempre será C:\Users\<usuario>\datasets_modificados  (Windows)
# o  /home/<usuario>/datasets_modificados               (Linux/Mac)
# No cambia sin importar desde dónde se ejecute el programa.
_DATOS_DIR = os.path.join(os.path.expanduser("~"), "datasets_modificados")

# ── Guardar dataset ───────────────────────────────────────────────────────

def guardar_dataset(df: pd.DataFrame, nombre_sugerido: str = "dataset_filtrado") -> str | None:
    """
    Ofrece dos acciones al usuario:
      [1] Guardar CSV en datasets_modificados/ (carpeta fija en home del usuario)
      [2] Abrir como tabla en el navegador (HTML temporal)
      [3] Ambas
    """
    os.makedirs(_DATOS_DIR, exist_ok=True)

    print(f"\n  {Fore.WHITE}  ¿Qué deseas hacer con el dataset modificado?")
    print(f"  {Fore.CYAN}  [1]  Guardar CSV en: {Fore.YELLOW}{_DATOS_DIR}")
    print(f"  {Fore.CYAN}  [2]  Abrir en el navegador como tabla HTML")
    print(f"  {Fore.CYAN}  [3]  Ambas opciones")
    print(f"  {Fore.CYAN}  [0]  Cancelar")
    accion = input(f"  {Fore.GREEN}> ").strip()

    if accion == "0" or not accion:
        print(f"  {Fore.YELLOW}  Cancelado.")
        return None

    ruta_csv = None

    if accion in ("1", "3"):
        print(f"  {Fore.WHITE}  Nombre del archivo (sin .csv, Enter={nombre_sugerido}): ",
              end="")
        nombre = input(f"{Fore.GREEN}").strip() or nombre_sugerido
        ruta_csv = os.path.join(_DATOS_DIR, f"{nombre}.csv")
        df.to_csv(ruta_csv, index=False)
        print(f"  {Fore.GREEN}  [OK] Guardado en: {ruta_csv}")
        print(f"  {Fore.CYAN}  Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

    if accion in ("2", "3"):
        _abrir_en_navegador(df, nombre_sugerido)

    return ruta_csv


def _abrir_en_navegador(df: pd.DataFrame, titulo: str = "Dataset modificado"):
    """Genera un HTML con la tabla del dataset y lo abre en el navegador por defecto."""
    import tempfile, webbrowser

    n_rows = len(df)
    n_cols = len(df.columns)

    # Construir HTML
    ths = "".join(f"<th>{c}</th>" for c in df.columns)
    trs = ""
    for i, (_, row) in enumerate(df.iterrows()):
        cls = "even" if i % 2 == 0 else "odd"
        trs += f"<tr class='{cls}'>"
        for val in row:
            trs += f"<td>{val}</td>"
        trs += "</tr>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>{titulo}</title>
<style>
  body  {{ font-family: Arial, sans-serif; background: #f4f6f9; margin: 24px; }}
  h2    {{ color: #1F4E79; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 16px; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th    {{ background: #1F4E79; color: #fff; padding: 10px 12px; text-align: left;
           position: sticky; top: 0; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #dde3ec; }}
  tr.even td {{ background: #ffffff; }}
  tr.odd  td {{ background: #eef3f8; }}
  tr:hover td {{ background: #d0e4f5; }}
</style>
</head>
<body>
  <h2>Dataset modificado — {titulo}</h2>
  <div class="meta">{n_rows} filas &nbsp;×&nbsp; {n_cols} columnas &nbsp;|&nbsp;
  Carpeta: {_DATOS_DIR}</div>
  <div class="wrap">
    <table>
      <thead><tr>{ths}</tr></thead>
      <tbody>{trs}</tbody>
    </table>
  </div>
</body>
</html>"""

    # Guardar en archivo temporal y abrir
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8")
    tmp.write(html)
    tmp.close()
    webbrowser.open(f"file:///{tmp.name.replace(chr(92), '/')}")
    print(f"  {Fore.GREEN}  [OK] Abierto en el navegador.")
    print(f"  {Fore.CYAN}  Archivo temporal: {tmp.name}")


# ── Mostrar tabla de ranking ──────────────────────────────────────────────

def _mostrar_ranking(scores: pd.Series, target_col: str, df_cols: list,
                     nombre_metodo: str = "Score"):
    """
    Imprime tabla ranked con marcas visuales:
      🟢 Top 5  → conservar
      🟡 Medio  → opcional
      🔴 Últimas → candidatas a eliminar
    """
    features = [c for c in scores.index if c in df_cols and c != target_col]
    scores_f  = scores.reindex(features).dropna().sort_values(ascending=False)
    n         = len(scores_f)
    top5      = max(5, int(n * 0.35))
    bot5      = max(3, int(n * 0.25))

    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Ranking de importancia — {nombre_metodo}")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  {'#':<4}  {'Variable':<28}  {'Score':>12}  {'Merit%':>7}  Recom.")
    print(f"{Fore.CYAN}  {'─'*60}")

    max_v = scores_f.max() if scores_f.max() > 0 else 1.0

    for rank, (var, val) in enumerate(scores_f.items(), 1):
        merit = val / max_v * 100
        if rank <= top5:
            marca = f"{Fore.GREEN}  CONSERVAR"
            color = Fore.GREEN
        elif rank <= n - bot5:
            marca = f"{Fore.YELLOW}  opcional"
            color = Fore.YELLOW
        else:
            marca = f"{Fore.RED}  ELIMINAR?"
            color = Fore.RED

        print(f"  {color}{rank:<4}  {var:<28}  {val:>12.6f}  {merit:>6.1f}%{marca}")

    print(f"{Fore.CYAN}  {_linea('─')}")
    print(f"{Fore.WHITE}  {Fore.GREEN}CONSERVAR{Fore.WHITE} = top {top5} variables  |  "
          f"{Fore.RED}ELIMINAR?{Fore.WHITE} = últimas {bot5} variables\n")

    return scores_f, top5, bot5


# ── Menú principal de edición ──────────────────────────────────────────────



def _interpretar_variable(col: str, serie, score: float = None):
    """Muestra estadísticas básicas de la variable para ayudar al usuario a decidir."""
    from colorama import Fore
    n_unique = serie.nunique()
    nulos    = serie.isnull().sum()
    dtype    = str(serie.dtype)

    tipo = "categórica" if dtype in ("object","category") or n_unique <= 10 else "continua"

    print(f"    {Fore.CYAN}{col}")
    print(f"      Tipo    : {Fore.WHITE}{tipo}  ({dtype})")
    print(f"      Únicos  : {Fore.WHITE}{n_unique}   Nulos: {nulos}")
    if tipo == "continua":
        try:
            print(f"      Rango   : {Fore.WHITE}[{serie.min():.2f} … {serie.max():.2f}]"
                  f"  Media={serie.mean():.2f}")
        except Exception:
            pass
    else:
        muestra = serie.dropna().unique()[:4]
        print(f"      Valores : {Fore.WHITE}{', '.join(str(v) for v in muestra)}"
              + ("..." if n_unique > 4 else ""))
    if score is not None:
        color = Fore.GREEN if score > 0.5 else Fore.YELLOW if score > 0.2 else Fore.RED
        print(f"      Score   : {color}{score:.4f}"
              + (f"  {Fore.RED}← baja importancia" if score <= 0.2 else ""))

def _menu_opciones(scores_f: pd.Series, df: pd.DataFrame, target_col: str,
                   bot5: int) -> pd.DataFrame:
    """
    Presenta las opciones de eliminación y retorna el df actualizado.
    """
    features  = [c for c in scores_f.index if c != target_col]
    n         = len(features)
    candidatas = list(scores_f.tail(bot5).index)

    print(f"  {Fore.WHITE}¿Qué deseas hacer?")
    print(f"  {Fore.CYAN}  [r]  Eliminar las {bot5} variables recomendadas ({Fore.RED}{', '.join(candidatas)}{Fore.CYAN})")
    print(f"  {Fore.CYAN}  [m]  Eliminar manualmente (una a una)")
    print(f"  {Fore.CYAN}  [l]  Listar y elegir por número")
    print(f"  {Fore.CYAN}  [n]  No eliminar nada")
    opcion = input(f"  {Fore.GREEN}> ").strip().lower()

    if opcion == "r":
        df = _eliminar_columnas(df, candidatas, target_col)

    elif opcion == "m":
        df = _eliminar_una_a_una(df, scores_f, target_col)

    elif opcion == "l":
        df = _eliminar_por_lista(df, scores_f, target_col)

    else:
        print(f"  {Fore.YELLOW}  Sin cambios.")
        return df

    # Preguntar si guardar
    if _preguntar_guardar():
        guardar_dataset(df)

    return df


def _eliminar_columnas(df: pd.DataFrame, columnas: list, target_col: str) -> pd.DataFrame:
    """Elimina columnas del df, protege el target."""
    a_eliminar = [c for c in columnas if c != target_col and c in df.columns]
    if not a_eliminar:
        print(f"  {Fore.YELLOW}  Nada que eliminar.")
        return df

    print(f"\n  {Fore.YELLOW}  Eliminando: {', '.join(a_eliminar)}")
    df_nuevo = df.drop(columns=a_eliminar)
    print(f"  {Fore.GREEN}  Dataset actualizado: {df_nuevo.shape[1]} columnas")
    return df_nuevo


def _eliminar_una_a_una(df: pd.DataFrame, scores_f: pd.Series, target_col: str) -> pd.DataFrame:
    """Loop interactivo: el usuario escribe el nombre de cada variable a eliminar."""
    features = [c for c in df.columns if c != target_col]

    while True:
        cols_act = [c for c in features if c in df.columns]
        print(f"\n  {Fore.WHITE}  Columnas actuales ({len(cols_act)}):")
        for i, c in enumerate(cols_act, 1):
            val = scores_f.get(c, 0)
            print(f"  {Fore.CYAN}    [{i:2d}] {c:<28}  score={val:.6f}")
        print(f"  {Fore.WHITE}  Nombre o número a eliminar (Enter=terminar): ", end="")
        entrada = input(f"{Fore.GREEN}").strip()
        if not entrada:
            break

        # Resolver nombre
        col = None
        if entrada.isdigit():
            idx = int(entrada) - 1
            if 0 <= idx < len(cols_act):
                col = cols_act[idx]
        else:
            if entrada in df.columns:
                col = entrada
            else:
                matches = [c for c in df.columns if entrada.lower() in c.lower()]
                if matches:
                    col = matches[0]

        if col is None:
            print(f"  {Fore.RED}  Columna no encontrada: {entrada}")
            continue
        if col == target_col:
            print(f"  {Fore.RED}  No se puede eliminar el target ({target_col}).")
            continue

        df = df.drop(columns=[col])
        print(f"  {Fore.GREEN}  Eliminada: {col}  →  quedan {df.shape[1]} columnas")

    return df


def _eliminar_por_lista(df: pd.DataFrame, scores_f: pd.Series, target_col: str) -> pd.DataFrame:
    """El usuario escribe los números separados por coma o espacio."""
    features = [c for c in scores_f.index if c in df.columns and c != target_col]
    print(f"\n  {Fore.WHITE}  Variables disponibles:")
    for i, c in enumerate(features, 1):
        val = scores_f.get(c, 0)
        print(f"  {Fore.CYAN}    [{i:2d}] {c:<28}  score={val:.6f}")

    print(f"  {Fore.WHITE}  Números a eliminar (ej: 3 5 8  o  3,5,8): ", end="")
    entrada = input(f"{Fore.GREEN}").strip()
    if not entrada:
        print(f"  {Fore.YELLOW}  Sin cambios.")
        return df

    tokens = entrada.replace(",", " ").split()
    a_eliminar = []
    for t in tokens:
        if t.isdigit():
            idx = int(t) - 1
            if 0 <= idx < len(features):
                a_eliminar.append(features[idx])

    if not a_eliminar:
        print(f"  {Fore.RED}  Ningún número válido.")
        return df

    print(f"  {Fore.YELLOW}  Eliminando: {', '.join(a_eliminar)}")
    df = df.drop(columns=a_eliminar)
    print(f"  {Fore.GREEN}  Dataset actualizado: {df.shape[1]} columnas")
    return df


def _preguntar_guardar() -> bool:
    resp = input(f"\n  {Fore.WHITE}  ¿Guardar nuevo dataset? (s/N): {Fore.GREEN}").strip().lower()
    return resp == "s"


# ── API PÚBLICA ────────────────────────────────────────────────────────────

def menu_eliminar_por_scores(df: pd.DataFrame, target_col: str,
                              scores: pd.Series, nombre_metodo: str = "Score") -> pd.DataFrame:
    """
    Llamar después de cada selector individual (opciones 2-5).
    Muestra ranking y ofrece eliminar variables.
    Retorna df (modificado o no).
    """
    print(f"\n  {Fore.WHITE}¿Deseas gestionar variables basándote en este ranking? (s/N): ", end="")
    if input(f"{Fore.GREEN}").strip().lower() != "s":
        return df

    scores_f, top5, bot5 = _mostrar_ranking(scores, target_col, list(df.columns), nombre_metodo)
    return _menu_opciones(scores_f, df, target_col, bot5)


def menu_eliminar_por_score_final(df: pd.DataFrame, target_col: str,
                                   resumen: pd.DataFrame) -> pd.DataFrame:
    """
    Llamar después de Comparar Todos (opción 7).
    Usa SCORE_FINAL del resumen. Incluye lógica de acuerdo entre métodos.
    """
    print(f"\n  {Fore.WHITE}¿Deseas gestionar variables basándote en el SCORE_FINAL? (s/N): ", end="")
    if input(f"{Fore.GREEN}").strip().lower() != "s":
        return df

    scores_final = resumen["SCORE_FINAL"].sort_values(ascending=False)
    cols_metodos = [c for c in resumen.columns if c != "SCORE_FINAL"]
    features     = [c for c in scores_final.index if c in df.columns and c != target_col]
    n            = len(features)
    top5         = max(5, int(n * 0.35))
    bot5         = max(3, int(n * 0.25))
    candidatas   = list(scores_final.reindex(features).tail(bot5).index)

    # Acuerdo entre métodos: ¿cuántos métodos ponen la variable en el tercio inferior?
    print(f"\n{Fore.CYAN}{_linea('=')}")
    print(f"{Fore.WHITE}  Ranking unificado — SCORE_FINAL + acuerdo entre métodos")
    print(f"{Fore.CYAN}{_linea('-')}")
    print(f"{Fore.WHITE}  {'#':<4}  {'Variable':<28}  {'SCORE':>8}  {'Acuerdo':>8}  Recom.")
    print(f"{Fore.CYAN}  {'─'*68}")

    max_v = scores_final.max() if scores_final.max() > 0 else 1.0

    acuerdos = {}
    for var in features:
        votos_bajos = 0
        for met in cols_metodos:
            if met not in resumen.columns:
                continue
            col_scores = resumen[met].sort_values(ascending=False)
            rank_var   = list(col_scores.index).index(var) + 1 if var in col_scores.index else n
            if rank_var > n * 0.65:          # está en el tercio inferior de este método
                votos_bajos += 1
        acuerdos[var] = votos_bajos

    for rank, var in enumerate(features, 1):
        val   = scores_final.get(var, 0)
        merit = val / max_v * 100
        vb    = acuerdos.get(var, 0)
        total = len(cols_metodos)
        acuerdo_str = f"{vb}/{total}"

        if rank <= top5:
            color = Fore.GREEN;  rec = "CONSERVAR"
        elif vb >= total * 0.6:   # mayoría de métodos la quieren eliminar
            color = Fore.RED;    rec = f"ELIMINAR (acuerdo {acuerdo_str})"
        elif rank > n - bot5:
            color = Fore.RED;    rec = "ELIMINAR?"
        else:
            color = Fore.YELLOW; rec = "opcional"

        print(f"  {color}{rank:<4}  {var:<28}  {val:>8.4f}  {acuerdo_str:>8}  {rec}")

    print(f"{Fore.CYAN}  {_linea('─')}")

    # Candidatas por acuerdo
    cand_acuerdo = [v for v in features if acuerdos.get(v, 0) >= len(cols_metodos) * 0.6]
    cand_rank    = list(scores_final.reindex(features).tail(bot5).index)
    candidatas   = list(dict.fromkeys(cand_acuerdo + cand_rank))  # union sin dups

    print(f"\n  {Fore.WHITE}  Variables recomendadas para eliminar ({len(candidatas)}):")
    for c in candidatas:
        vb = acuerdos.get(c, 0)
        print(f"  {Fore.RED}    • {c:<28}  SCORE={scores_final.get(c,0):.4f}  acuerdo={vb}/{len(cols_metodos)}")

    print(f"\n  {Fore.WHITE}¿Qué deseas hacer?")
    print(f"  {Fore.CYAN}  [r]  Eliminar las {len(candidatas)} variables recomendadas")
    print(f"  {Fore.CYAN}  [m]  Eliminar manualmente (una a una)")
    print(f"  {Fore.CYAN}  [l]  Listar y elegir por número")
    print(f"  {Fore.CYAN}  [n]  No eliminar nada")
    opcion = input(f"  {Fore.GREEN}> ").strip().lower()

    scores_f = scores_final.reindex(features).dropna()

    if opcion == "r":
        df = _eliminar_columnas(df, candidatas, target_col)
    elif opcion == "m":
        df = _eliminar_una_a_una(df, scores_f, target_col)
    elif opcion == "l":
        df = _eliminar_por_lista(df, scores_f, target_col)
    else:
        print(f"  {Fore.YELLOW}  Sin cambios.")
        return df

    if _preguntar_guardar():
        guardar_dataset(df)

    return df
