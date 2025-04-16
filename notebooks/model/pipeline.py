"""
Pipeline de Feature Engineering que combina todas las transformaciones
desarrolladas por el equipo.
"""

import os
import pandas as pd
import importlib
import sys

# Agregar la carpeta actual al path para poder importar los módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ruta al archivo original
INPUT_PATH = '../../input/petfinder-adoption-prediction/train/train.csv'
# Ruta donde guardar el dataset procesado
OUTPUT_PATH = '../../datasets_procesados/dataset_final.csv'

def run_pipeline():
    # Cargar el dataset original
    print("Cargando dataset original...")
    dataset = pd.read_csv(INPUT_PATH)
    print(f"Dataset original cargado. Shape: {dataset.shape}")
    
    # Lista de módulos de feature engineering a aplicar
    fe_modules = [
        'fe_jose',
        #'FE_gmp',
        'fe_adrian',  # Tu módulo
        #Agrega aquí los módulos de tus compañeros
        # 'fe_nombre_compañero1',
        # 'fe_nombre_compañero2',
    ]
    
    # Aplicar cada módulo de feature engineering en secuencia
    for module_name in fe_modules:
        try:
            print(f"\nAplicando transformaciones de {module_name}...")
            # Importar dinámicamente el módulo
            module = importlib.import_module(module_name)
            
            # Verificar si el módulo tiene una función apply_features
            if hasattr(module, 'apply_features'):
                dataset = module.apply_features(dataset)
            else:
                print(f"⚠️ Advertencia: {module_name} no tiene función apply_features()")
            
            print(f"Transformaciones de {module_name} aplicadas. Nuevo shape: {dataset.shape}")
        except Exception as e:
            print(f"❌ Error al aplicar {module_name}: {str(e)}")
    
    # Guardar el dataset procesado
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset final guardado en {OUTPUT_PATH}")
    print(f"Shape final: {dataset.shape}")
    
    return dataset

if __name__ == "__main__":
    run_pipeline()