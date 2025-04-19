"""
Pipeline de Feature Engineering que combina todas las transformaciones
desarrolladas por el equipo.
"""

import os
import pandas as pd
import sys
import importlib.util
import importlib

# Agregar las carpetas al path para poder importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Agrega la carpeta notebooks/

# Ruta al archivo original
INPUT_PATH = 'input/petfinder-adoption-prediction/train/train.csv'
# Ruta donde guardar el dataset procesado
OUTPUT_PATH = 'datasets_procesados/dataset_final.csv'

def run_pipeline():
    # Cargar el dataset original
    print("Cargando dataset original...")
    dataset = pd.read_csv(INPUT_PATH)
    print(f"Dataset original cargado. Shape: {dataset.shape}")
    
    # Lista de módulos de feature engineering a aplicar con rutas relativas al proyecto
    fe_files = [
        '../fe/fe_jose.py',
        '../fe/fe_gmp.py',
        '../fe/fe_adrian.py',
    ]
    
    # Aplicar cada archivo de feature engineering en secuencia
    for file_path in fe_files:
        try:
            # Construir la ruta completa al archivo
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
            
            if not os.path.exists(full_path):
                print(f"⚠️ Advertencia: El archivo {full_path} no existe.")
                continue
                
            print(f"\nAplicando transformaciones de {os.path.basename(file_path)}...")
            
            # Importar el módulo desde la ruta de archivo
            module_name = os.path.basename(file_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Verificar si el módulo tiene una función apply_features
            if hasattr(module, 'apply_features'):
                dataset = module.apply_features(dataset)
                print(f"Transformaciones de {module_name} aplicadas. Nuevo shape: {dataset.shape}")
            else:
                print(f"⚠️ Advertencia: {module_name} no tiene función apply_features()")
                
        except Exception as e:
            print(f"❌ Error al aplicar {file_path}: {str(e)}")
    
    # Guardar el dataset procesado
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset final guardado en {OUTPUT_PATH}")
    print(f"Shape final: {dataset.shape}")
    
    return dataset

if __name__ == "__main__":
    run_pipeline()