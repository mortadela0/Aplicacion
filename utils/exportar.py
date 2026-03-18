import os
import pandas as pd
from colorama import Fore

def exportar_csv(resumen, nombre="resultados_seleccion"):
    os.makedirs("resultados", exist_ok=True)
    ruta = os.path.join("resultados", f"{nombre}.csv")
    resumen.to_csv(ruta)
    print(f"\n  {Fore.GREEN}[OK] Guardado: {ruta}")
    return ruta
