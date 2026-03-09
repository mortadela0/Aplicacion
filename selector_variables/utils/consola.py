"""
utils/consola.py
─────────────────────────────────────────────
Utilidades de interfaz de consola.
elegir_target() marca visualmente strings vs numéricos
y acepta índice O nombre de columna.
"""

import os
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)


def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║   {Fore.WHITE}SELECTOR DE VARIABLES  {Fore.YELLOW}v2.1{Fore.CYAN}                          ║
║   {Fore.WHITE}Correlation | Chi2 | RF | Gower | {Fore.YELLOW}Weka API{Fore.CYAN}           ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")


def separador(titulo: str = ""):
    print(f"\n{Fore.CYAN}{'─'*70}")
    if titulo:
        print(f"{Fore.WHITE}  {titulo}")
        print(f"{Fore.CYAN}{'─'*70}")


def menu_principal(weka_ok: bool = False) -> str:
    estado_weka = f"{Fore.GREEN}[ON] " if weka_ok else f"{Fore.RED}[OFF]"
    print(f"\n{Fore.CYAN}{'─'*70}")
    print(f"  {Fore.WHITE}[1]{Fore.YELLOW}  Cargar CSV")
    print(f"  {Fore.CYAN}  ── Metodos individuales ──────────────────────────────────")
    print(f"  {Fore.WHITE}[2]{Fore.YELLOW}  Correlacion de Pearson       (numericas lineales)")
    print(f"  {Fore.WHITE}[3]{Fore.YELLOW}  Chi2 / F-score                (SelectKBest ANOVA/F)")
    print(f"  {Fore.WHITE}[4]{Fore.YELLOW}  Random Forest                 (Gini importance)")
    print(f"  {Fore.WHITE}[5]{Fore.YELLOW}  Distancia de Gower            (datos mixtos)")
    print(f"  {Fore.WHITE}[6]{Fore.YELLOW}  Weka evaluators               (InfoGain/GainRatio/ReliefF/CFS)  {estado_weka}")
    print(f"  {Fore.CYAN}  ── Comparacion y exportacion ─────────────────────────────")
    print(f"  {Fore.WHITE}[7]{Fore.YELLOW}  Comparar TODOS los metodos    (incluye Weka si activo)")
    print(f"  {Fore.WHITE}[8]{Fore.YELLOW}  Exportar resultados a CSV")
    print(f"  {Fore.WHITE}[0]{Fore.RED}  Salir")
    print(f"{Fore.CYAN}{'─'*70}")
    return input(f"{Fore.WHITE}  Selecciona opcion: {Fore.GREEN}").strip()


def pausar():
    input(f"\n{Fore.CYAN}  [Enter para continuar...]")


# ─── CLASIFICADOR DE TIPO DE COLUMNA ─────────────────────────────────────────

def _tipo_target(serie: pd.Series) -> tuple:
    """
    Clasifica una columna y devuelve (etiqueta, color, icono).

    Categorías:
      STRING / nominal   → verde  ✓  (ideal para clasificación)
      INT    / N clases  → amarillo ~  (clasificación posible)
      FLOAT  / continuo  → blanco  ·  (regresión)
    """
    dtype  = serie.dtype
    n_uniq = serie.nunique()

    if dtype == object or str(dtype) == "category":
        muestra = serie.dropna().astype(str).unique()[:3]
        ej      = " / ".join(f'"{v}"' for v in muestra)
        return f"STRING  nominal  ({ej}...)", Fore.GREEN, "✓"

    elif pd.api.types.is_integer_dtype(dtype) and n_uniq <= 20:
        muestra = sorted(serie.dropna().unique())[:4]
        ej      = " / ".join(str(v) for v in muestra)
        return f"INT     {n_uniq} clases  ({ej}...)", Fore.YELLOW, "~"

    elif pd.api.types.is_integer_dtype(dtype):
        return f"INT     continuo  ({n_uniq} únicos)", Fore.WHITE, "·"

    else:
        try:
            mn = round(float(serie.min()), 3)
            mx = round(float(serie.max()), 3)
            return f"FLOAT   continuo  [{mn} … {mx}]", Fore.WHITE, "·"
        except (TypeError, ValueError):
            # Columna mixta o con strings — tratar como nominal
            muestra = serie.dropna().astype(str).unique()[:3]
            ej      = " / ".join(f'"{v}"' for v in muestra)
            return f"MIXED   nominal   ({ej}...)", Fore.GREEN, "✓"


# ─── ELEGIR TARGET ────────────────────────────────────────────────────────────

def elegir_target(df: pd.DataFrame) -> str | None:
    """
    Lista todas las columnas marcando visualmente:
      ✓ verde   → STRING/nominal     → ideal como clase
      ~ amarillo → INT con pocas clases → puede usarse como clase
      · blanco  → continuo           → para regresión

    Acepta ÍNDICE numérico o NOMBRE de columna (parcial, case-insensitive).
    Pide confirmación si la columna elegida parece continua.
    """
    separador("SELECT TARGET COLUMN  (class / label)")

    print(f"\n  {Fore.CYAN}  {'Idx':>4}   {'Tipo':<8} {'Columna':<28}  {'Detalle'}")
    print(f"  {Fore.CYAN}  {'─'*75}")

    for i, col in enumerate(df.columns):
        tipo_str, color, icono = _tipo_target(df[col])
        n_null = df[col].isnull().sum()
        null_s = f"  {Fore.RED}[{n_null} null]{color}" if n_null > 0 else ""
        print(f"  {color}  [{i:2d}]  {icono}  {col:<28}  {tipo_str}{null_s}")

    print(f"\n  {Fore.CYAN}  Leyenda: "
          f"{Fore.GREEN}✓ STRING/nominal  "
          f"{Fore.YELLOW}~ INT/clases  "
          f"{Fore.WHITE}· continuo/regresión")

    print(f"\n  {Fore.WHITE}  Escribe el ÍNDICE o el NOMBRE de la columna target:")
    entrada = input(f"  {Fore.GREEN}> ").strip()

    # ── Resolver por índice ───────────────────────────
    try:
        idx        = int(entrada)
        target_col = df.columns[idx]
    except (ValueError, IndexError):
        # ── Resolver por nombre (exact → partial → error) ─
        entrada_l = entrada.lower()
        matches   = [c for c in df.columns if c.lower() == entrada_l]
        if not matches:
            matches = [c for c in df.columns if entrada_l in c.lower()]
        if not matches:
            print(f"  {Fore.RED}[ERROR] Columna '{entrada}' no encontrada.")
            return None
        if len(matches) > 1:
            print(f"  {Fore.YELLOW}[WARN] Varias coincidencias: {matches}")
            print(f"  {Fore.YELLOW}       Escribe el nombre exacto.")
            return None
        target_col = matches[0]

    # ── Mostrar resumen del target elegido ────────────
    serie              = df[target_col]
    tipo_str, color, icono = _tipo_target(serie)
    n_uniq             = serie.nunique()
    muestra            = serie.dropna().unique()[:6]
    muestra_str        = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in muestra)
    if len(serie.dropna().unique()) > 6:
        muestra_str += ", ..."

    print(f"\n  {Fore.CYAN}  {'─'*60}")
    print(f"  {color}  {icono}  Columna  : {target_col}")
    print(f"  {color}     Tipo     : {tipo_str}")
    print(f"  {color}     Únicos   : {n_uniq}")
    print(f"  {color}     Muestra  : {muestra_str}")
    print(f"  {Fore.CYAN}  {'─'*60}")

    # ── Advertencia si parece continuo ───────────────
    if "continuo" in tipo_str and n_uniq > 20:
        print(f"\n  {Fore.YELLOW}  [AVISO] Esta columna parece CONTINUA ({n_uniq} únicos).")
        print(f"  {Fore.YELLOW}          Para clasificación se recomienda una columna STRING")
        print(f"  {Fore.YELLOW}          o con pocas clases enteras.")
        resp = input(f"  {Fore.WHITE}  ¿Continuar de todas formas? (s/N): {Fore.GREEN}").strip().lower()
        if resp != "s":
            print(f"  {Fore.YELLOW}  Selección cancelada. Elige otra columna.")
            return None

    print(f"  {Fore.GREEN}  [OK] Target: '{target_col}'")
    return target_col