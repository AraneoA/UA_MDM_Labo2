import os
import sys
import pandas as pd
import numpy as np
import importlib.util
import traceback
import inspect
from sklearn.model_selection import train_test_split

# --- Configuración de Rutas ---
try:
    # Intenta obtener la ruta del directorio del script actual
    # __file__ es la ruta al script que se está ejecutando
    # os.path.abspath convierte una ruta relativa en absoluta
    # os.path.dirname obtiene el directorio padre de una ruta
    # Hacemos dirname tres veces para subir desde notebooks/model/ a UA_MDM_Labo2_ADV/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Si __file__ no está definido (p.ej., en un entorno interactivo como Jupyter sin guardar),
    # usa el directorio de trabajo actual como raíz del proyecto.
    print("Advertencia: __file__ no definido. Usando el directorio de trabajo actual como PROJECT_ROOT.")
    PROJECT_ROOT = os.getcwd()

# Añade el directorio raíz del proyecto al sys.path para poder importar módulos desde allí
sys.path.insert(0, PROJECT_ROOT)

# Define directorios clave basados en PROJECT_ROOT
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input', 'petfinder-adoption-prediction')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'datasets_procesados')
FE_DIR = os.path.join(PROJECT_ROOT, 'notebooks', 'fe') # Directorio de módulos de Feature Engineering
FEATURES_PRECOMPUTED_DIR = os.path.join(PROJECT_ROOT, 'features_precalculadas', 'rescuer_ratios') # Directorio para guardar/cargar features precalculadas

# Rutas específicas de archivos de entrada y salida
TRAIN_INPUT_PATH = os.path.join(INPUT_DIR, 'train', 'train.csv') # Dataset original de Kaggle
TRAIN_SPLIT_PATH = os.path.join(OUTPUT_DIR, 'train_split.csv') # Split de entrenamiento temporal (antes de FE)
TEST_SPLIT_PATH = os.path.join(OUTPUT_DIR, 'test_split.csv')   # Split de test temporal (antes de FE)
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'train_final_thres.csv') # Salida final del pipeline para entrenamiento
TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'test_final_thres.csv')   # Salida final del pipeline para test

# Lista de nombres de los módulos de feature engineering (archivos .py en FE_DIR) a aplicar
FE_MODULE_NAMES = [
    'fe_adrian',
    'fe_jose',
    'fe_gloria',
    'fe_seba'
]

# --- Carga dinámica del módulo de pre-cálculo (con umbral) ---
PRECOMPUTE_MODULE_PATH = os.path.join(FE_DIR, 'precalc_rescuer_features_threshold.py')
precompute_rescuer_features = None # Inicializa como None
if os.path.exists(PRECOMPUTE_MODULE_PATH):
    try:
        # Carga la especificación del módulo desde su ruta
        spec_precompute = importlib.util.spec_from_file_location('precalc_rescuer_features_threshold', PRECOMPUTE_MODULE_PATH)
        if spec_precompute and spec_precompute.loader:
            # Crea el objeto módulo desde la especificación
            precompute_rescuer_features = importlib.util.module_from_spec(spec_precompute)
            # Ejecuta el código del módulo para que sus funciones estén disponibles
            spec_precompute.loader.exec_module(precompute_rescuer_features)
            print("Módulo de pre-cálculo 'precalc_rescuer_features_threshold' cargado.")
        else:
            print(f"Advertencia: No se pudo cargar la especificación para {PRECOMPUTE_MODULE_PATH}")
    except Exception as e:
        # Captura cualquier error durante la importación
        print(f"Advertencia: No se pudo importar 'precalc_rescuer_features_threshold'. Error: {e}")
        print(traceback.format_exc()) # Imprime el traceback completo del error
else:
    # Si el archivo .py no existe
    print(f"Advertencia: Archivo de pre-cálculo no encontrado en {PRECOMPUTE_MODULE_PATH}")


def run_feature_engineering(input_df, output_path, is_train=True):
    """
    Aplica el pipeline de feature engineering a un DataFrame dado.

    Args:
        input_df (pd.DataFrame): DataFrame de entrada (train o test split).
        output_path (str): Ruta donde guardar el DataFrame procesado.
        is_train (bool): Flag para indicar si es el conjunto de entrenamiento.
                         Esto se usa para decidir qué features precalculadas mergear
                         (OOF para train, agregadas para test) y se pasa a los módulos FE.

    Returns:
        pd.DataFrame or None: El DataFrame procesado o None si ocurre un error crítico.
    """
    print(f"\n--- Iniciando Feature Engineering {'para ENTRENAMIENTO' if is_train else 'para TEST'} (con Umbral) ---")
    print(f"Input DF shape: {input_df.shape}")
    print(f"Output Path: {output_path}")

    dataset = input_df.copy() # Trabaja sobre una copia para no modificar el original
    original_columns = set(dataset.columns) # Guarda las columnas originales para reporte

    # --- 1. Merge con Features Precalculadas (Agregadas con Umbral) ---
    # En esta versión (umbral), usamos el mismo archivo agregado para train y test.
    print("\nIntentando merge con features precalculadas (agregadas con umbral)...")
    added_feature_cols = []
    # Define la ruta al único archivo generado por precalc_rescuer_features_threshold.py
    agg_features_path = os.path.join(FEATURES_PRECOMPUTED_DIR, 'rescuer_ratios_full_train_agg_thres.csv')

    if os.path.exists(agg_features_path):
        try:
            print(f"Cargando features agregadas (umbral) desde: {agg_features_path}")
            try:
                # Intenta cargar asumiendo que RescuerID es el índice
                agg_features = pd.read_csv(agg_features_path, index_col='RescuerID')
            except ValueError:
                # Si falla (RescuerID no es índice), carga normal y lo establece como índice
                agg_features = pd.read_csv(agg_features_path)
                if 'RescuerID' not in agg_features.columns:
                    raise ValueError("El archivo agregado (umbral) no contiene 'RescuerID' ni como índice ni como columna.")
                agg_features.set_index('RescuerID', inplace=True)

            print(f"Features agregadas (umbral) cargadas. Shape: {agg_features.shape}")

            # Verifica que la columna para el merge exista en el dataset principal
            if 'RescuerID' not in dataset.columns:
                raise KeyError("'RescuerID' no encontrado en el dataset. No se puede hacer merge.")

            # Asegura que los tipos de 'RescuerID' coincidan para el merge
            if dataset['RescuerID'].dtype != agg_features.index.dtype:
                print(f"Advertencia: Tipo de dato de RescuerID difiere. Dataset: {dataset['RescuerID'].dtype}, Agregado: {agg_features.index.dtype}. Intentando convertir ambos a string.")
                try:
                    dataset['RescuerID'] = dataset['RescuerID'].astype(str)
                    agg_features.index = agg_features.index.astype(str)
                except Exception as cast_e:
                    print(f"❌ Error al convertir tipos de RescuerID: {cast_e}. Saltando merge.")
                    raise # Relanza el error para detener el merge

            # Prepara el mapeo para renombrar columnas (ej: _agg -> _rescuer_ratio)
            agg_cols_original = agg_features.columns.tolist()
            rename_map_agg = {col: col.replace('_agg', '_rescuer_ratio') for col in agg_cols_original if '_agg' in col}
            added_feature_cols = list(rename_map_agg.values()) # Guarda los nombres finales

            # Realiza el merge usando la columna 'RescuerID'
            dataset = pd.merge(
                dataset,
                agg_features,
                on='RescuerID', # Columna clave para el merge
                how='left'      # Mantiene todas las filas del dataset original (izquierdo)
            )
            dataset.rename(columns=rename_map_agg, inplace=True) # Renombra las columnas mergeadas
            print(f"Merge agregado (umbral) completado y columnas renombradas. Shape después del merge: {dataset.shape}")

            # Rellena NaNs introducidos por el merge (rescatistas no presentes en agg_features) con 0
            if added_feature_cols:
                dataset[added_feature_cols] = dataset[added_feature_cols].fillna(0)
                print("NaNs rellenados con 0 para features agregadas (renombradas).")
            else:
                print("No se encontraron columnas agregadas para renombrar y rellenar NaNs.")

        except Exception as e:
            # Captura cualquier error durante el proceso de merge
            print(f"❌ Error durante el merge con los ratios agregados (umbral): {e}")
            print(traceback.format_exc())
            print("⚠️ Advertencia: El pipeline continuará sin los ratios agregados (umbral).")
    else:
        # Si el archivo precalculado no existe
        print(f"⚠️ Advertencia: Archivo de ratios agregados (umbral) no encontrado en {agg_features_path}. Saltando merge agregado.")

    # --- 2. Eliminar columnas 'ratio_AdoptionSpeedX' si existen ---
    # Puede ser útil si algún módulo FE las crea y queremos usar solo las precalculadas.
    cols_to_drop = [col for col in dataset.columns if col.startswith('ratio_AdoptionSpeed')]
    if cols_to_drop:
        print(f"Eliminando columnas pre-existentes 'ratio_AdoptionSpeedX': {cols_to_drop}")
        dataset.drop(columns=cols_to_drop, inplace=True)

    # --- 3. Aplicar Módulos de Feature Engineering (fe_*.py) ---
    print("\nAplicando módulos de Feature Engineering...")

    for module_name in FE_MODULE_NAMES: # Itera sobre la lista definida al inicio
        try:
            module_path = os.path.join(FE_DIR, f"{module_name}.py")
            if not os.path.exists(module_path):
                print(f"⚠️ Advertencia: El archivo {module_path} no existe.")
                continue # Salta al siguiente módulo

            print(f"  -> Aplicando {module_name}...")

            # Carga dinámica del módulo FE
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                print(f"❌ Error: No se pudo crear la especificación o el loader para {module_path}")
                continue
            module = importlib.util.module_from_spec(spec)
            if module is None:
                print(f"❌ Error: No se pudo crear el módulo desde la especificación para {module_path}")
                continue

            spec.loader.exec_module(module) # Ejecuta el código del módulo

            # Busca y ejecuta la función 'apply_features' si existe
            if hasattr(module, 'apply_features'):
                current_shape = dataset.shape # Guarda shape antes de aplicar
                try:
                    # Inspecciona la firma de la función para ver si acepta 'is_train'
                    sig = inspect.signature(module.apply_features)
                    if 'is_train' in sig.parameters:
                        # Llama con el flag is_train
                        dataset = module.apply_features(dataset, is_train=is_train)
                        print(f"     ✅ {module_name} aplicado (con is_train). Shape: {current_shape} -> {dataset.shape}")
                    else:
                        # Llama sin el flag is_train
                        dataset = module.apply_features(dataset)
                        print(f"     ✅ {module_name} aplicado (sin is_train). Shape: {current_shape} -> {dataset.shape}")
                except Exception as inner_e:
                    # Captura errores dentro de la función apply_features del módulo
                    print(f"❌ Error dentro de {module_name}.apply_features:")
                    print(traceback.format_exc())
                    print(f"⚠️ Saltando el resto de las transformaciones debido al error en {module_name}.")
                    return None # Detiene el pipeline si un módulo falla
            else:
                # Si el módulo no tiene la función esperada
                print(f"⚠️ Advertencia: {module_name} no tiene función apply_features().")

        except Exception as e:
            # Captura errores al cargar o ejecutar el módulo FE en sí
            print(f"❌ Error al cargar o ejecutar el módulo {module_name}.py:")
            print(traceback.format_exc())
            print(f"⚠️ Saltando el resto de las transformaciones debido al error en {module_name}.")
            return None # Detiene el pipeline

    # --- 4. Guardar Resultados ---
    new_columns = set(dataset.columns) - original_columns # Columnas añadidas
    print(f"\nNuevas columnas agregadas por el pipeline: {new_columns if new_columns else 'Ninguna'}")

    try:
        # Crea el directorio de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Guarda el DataFrame procesado en CSV
        dataset.to_csv(output_path, index=False) # index=False para no guardar el índice del DF
        print(f"\n✅ Dataset procesado guardado en {output_path}")
        print(f"Shape final: {dataset.shape}")
    except Exception as e:
        # Captura errores al guardar el archivo
        print(f"❌ Error al guardar el archivo en {output_path}: {e}")
        return None

    return dataset # Devuelve el DataFrame procesado


# --- Bloque Principal de Ejecución ---
# Este código solo se ejecuta si el script es llamado directamente (no importado)
if __name__ == "__main__":
    print("=============================================")
    print("=== Iniciando Ejecución del Pipeline Completo (Umbral) ===")
    print("=============================================")
    # Imprime las rutas configuradas para verificación
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Input Dir: {INPUT_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"FE Dir: {FE_DIR}")
    print(f"Precomputed Features Dir: {FEATURES_PRECOMPUTED_DIR}")
    print(f"Módulos FE a aplicar: {FE_MODULE_NAMES}")

    # --- 1. Cargar y dividir el dataset original ---
    print(f"\nCargando dataset original desde: {TRAIN_INPUT_PATH}")
    try:
        original_df = pd.read_csv(TRAIN_INPUT_PATH)
        print(f"Dataset original cargado. Shape: {original_df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo de entrada original en {TRAIN_INPUT_PATH}")
        sys.exit(1) # Termina el script si no encuentra el input
    except Exception as e:
        print(f"❌ Error al cargar {TRAIN_INPUT_PATH}: {e}")
        sys.exit(1)

    TARGET_COLUMN = 'AdoptionSpeed' # Nombre de la columna objetivo
    TEST_SIZE = 0.2 # Proporción del dataset para el split de test
    RANDOM_STATE = 42 # Semilla para reproducibilidad del split

    # Verifica que la columna objetivo exista
    if TARGET_COLUMN not in original_df.columns:
        print(f"❌ Error: La columna objetivo '{TARGET_COLUMN}' no se encuentra en el dataset original.")
        sys.exit(1)

    print(f"\nDividiendo el dataset original en train/test ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
    # Realiza el split estratificado para mantener la proporción de clases en ambos splits
    train_df, test_df = train_test_split(
        original_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=original_df[TARGET_COLUMN] # Estratifica por la columna objetivo
    )
    print(f"División completada:")
    print(f"  Train split shape: {train_df.shape}")
    print(f"  Test split shape:  {test_df.shape}")

    # Guarda los splits temporales (antes de FE) para referencia o uso en pre-cálculo
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Asegura que el directorio de salida exista
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)


    # --- 2. Pre-cálculo SOLO con el split de train (usando el módulo con umbral) ---
    if precompute_rescuer_features: # Verifica si el módulo se cargó correctamente
        print("\n--- Ejecutando Pre-cálculo para Rescuer Features SOLO con el split de train (umbral) ---")
        try:
            # Llama a la función del módulo precompute cargado
            precompute_rescuer_features.generate_rescuer_features(
                train_file=TRAIN_SPLIT_PATH, # Usa el split de train como input
                output_dir=FEATURES_PRECOMPUTED_DIR, # Directorio donde guardar el .csv precalculado
                min_count=10  # Define el umbral mínimo de registros por rescatista         ###################################################
            )
            print("--- Pre-cálculo de Rescuer Features (umbral) completado ---")
        except Exception as e:
            # Captura errores durante el pre-cálculo
            print(f"❌ Error durante el pre-cálculo de Rescuer Features (umbral): {e}")
            print(traceback.format_exc())
            print("⚠️ Advertencia: El pipeline continuará, pero el merge de features precalculadas podría fallar.")

    # --- 3. Ejecutar Feature Engineering para el Split de Entrenamiento ---
    print("\n--- Procesando Split de Entrenamiento ---")
    train_result_df = run_feature_engineering(
        input_df=train_df, # Pasa el split de entrenamiento
        output_path=TRAIN_OUTPUT_PATH, # Ruta de salida final para train
        is_train=True # Indica que es el set de entrenamiento
    )
    if train_result_df is None:
        print("\n❌ Pipeline de Feature Engineering para Entrenamiento falló.")
        # Podrías decidir terminar aquí si el FE de train falla: sys.exit(1)
    else:
        print("\n✅ Pipeline de Feature Engineering para Entrenamiento completado exitosamente.")

    # --- 4. Ejecutar Feature Engineering para el Split de Test ---
    print("\n--- Procesando Split de Test ---")
    test_result_df = run_feature_engineering(
        input_df=test_df, # Pasa el split de test
        output_path=TEST_OUTPUT_PATH, # Ruta de salida final para test
        is_train=False # Indica que es el set de test
    )
    if test_result_df is None:
        print("\n❌ Pipeline de Feature Engineering para Test falló.")
    else:
        print("\n✅ Pipeline de Feature Engineering para Test completado exitosamente.")

    print("\n=============================================")
    print("=== Ejecución del Pipeline Finalizada ===")
    print("=============================================")
