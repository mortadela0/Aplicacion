import os
import pandas as pd
from colorama import Fore, Style, init
init(autoreset=True)

def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║   {Fore.WHITE}SELECTOR DE VARIABLES  {Fore.YELLOW}v3.0{Fore.CYAN}                          ║
║   {Fore.WHITE}Selectores | Supervisados | No Supervisados{Fore.CYAN}              ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

def separador(titulo=""):
    print(f"\n{Fore.CYAN}{'─'*70}")
    if titulo:
        print(f"{Fore.WHITE}  {titulo}")
        print(f"{Fore.CYAN}{'─'*70}")

def menu_principal(weka_ok=False):
    weka_st = f"{Fore.GREEN}[ON]" if weka_ok else f"{Fore.RED}[OFF]"
    print(f"\n{Fore.CYAN}{'─'*70}")
    print(f"  {Fore.WHITE}[1]{Fore.YELLOW}  Cargar CSV")
    print(f"  {Fore.CYAN}  ── Selectores de variables ───────────────────────────────")
    print(f"  {Fore.WHITE}[2]{Fore.YELLOW}  Correlacion de Pearson")
    print(f"  {Fore.WHITE}[3]{Fore.YELLOW}  Chi2 / F-score")
    print(f"  {Fore.WHITE}[4]{Fore.YELLOW}  Random Forest  (Gini)")
    print(f"  {Fore.WHITE}[5]{Fore.YELLOW}  Gower Distance")
    print(f"  {Fore.WHITE}[6]{Fore.YELLOW}  Weka evaluators  {weka_st}")
    print(f"  {Fore.WHITE}[7]{Fore.YELLOW}  Comparar TODOS")
    print(f"  {Fore.WHITE}[8]{Fore.YELLOW}  Exportar resultados a CSV")
    print(f"  {Fore.CYAN}  ── Clasificadores supervisados ───────────────────────────")
    print(f"  {Fore.WHITE}[9]{Fore.YELLOW}  J48 | Naive Bayes | KNN | SVM | SGD")
    print(f"  {Fore.CYAN}  ── Clasificadores no supervisados ────────────────────────")
    print(f"  {Fore.WHITE}[10]{Fore.YELLOW} Hierarchical | KMeans | DBSCAN | GMM | KMedoids")
    print(f"  {Fore.WHITE}[0]{Fore.RED}  Salir")
    print(f"{Fore.CYAN}{'─'*70}")
    return input(f"{Fore.WHITE}  Selecciona opcion: {Fore.GREEN}").strip()

def pausar():
    input(f"\n{Fore.CYAN}  [Enter para continuar...]")

def _tipo_target(serie):
    dtype  = serie.dtype
    n_uniq = serie.nunique()
    if dtype == object or str(dtype) == "category":
        muestra = serie.dropna().astype(str).unique()[:3]
        return f"STRING  ({'  /  '.join(muestra)}...)", Fore.GREEN, "✓"
    elif pd.api.types.is_integer_dtype(dtype) and n_uniq <= 20:
        muestra = sorted(serie.dropna().unique())[:4]
        return f"INT {n_uniq} clases  ({'  /  '.join(map(str,muestra))}...)", Fore.YELLOW, "~"
    elif pd.api.types.is_integer_dtype(dtype):
        return f"INT continuo  ({n_uniq} únicos)", Fore.WHITE, "·"
    else:
        try:
            return f"FLOAT  [{round(float(serie.min()),3)} … {round(float(serie.max()),3)}]", Fore.WHITE, "·"
        except:
            return "MIXED nominal", Fore.GREEN, "✓"

def elegir_target(df):
    separador("SELECT TARGET COLUMN")
    print(f"\n  {Fore.CYAN}  {'Idx':>4}   {'Columna':<28}  Detalle")
    print(f"  {Fore.CYAN}  {'─'*70}")
    for i, col in enumerate(df.columns):
        tipo_str, color, icono = _tipo_target(df[col])
        nulos = df[col].isnull().sum()
        null_s = f"  {Fore.RED}[{nulos} null]{color}" if nulos else ""
        print(f"  {color}  [{i:2d}]  {icono}  {col:<28}  {tipo_str}{null_s}")
    entrada = input(f"\n  {Fore.WHITE}  Índice o nombre: {Fore.GREEN}").strip()
    try:
        target_col = df.columns[int(entrada)]
    except (ValueError, IndexError):
        matches = [c for c in df.columns if entrada.lower() in c.lower()]
        if not matches:
            print(f"  {Fore.RED}[ERROR] Columna no encontrada.")
            return None
        target_col = matches[0]
    print(f"  {Fore.GREEN}  [OK] Target: '{target_col}'")
    return target_col
