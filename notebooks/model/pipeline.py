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
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    print("Advertencia: __file__ no definido. Usando el directorio de trabajo actual como PROJECT_ROOT.")
    PROJECT_ROOT = os.getcwd()

sys.path.insert(0, PROJECT_ROOT)

INPUT_DIR = os.path.join(PROJECT_ROOT, 'input', 'petfinder-adoption-prediction')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'datasets_procesados')
FE_DIR = os.path.join(PROJECT_ROOT, 'notebooks', 'fe')
FEATURES_PRECOMPUTED_DIR = os.path.join(PROJECT_ROOT, 'features_precalculadas', 'rescuer_ratios')

# Rutas específicas de archivos
TRAIN_INPUT_PATH = os.path.join(INPUT_DIR, 'train', 'train.csv')
TRAIN_SPLIT_PATH = os.path.join(OUTPUT_DIR, 'train_split.csv')
TEST_SPLIT_PATH = os.path.join(OUTPUT_DIR, 'test_split.csv')
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'train_final.csv')
TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'test_final.csv')

FE_MODULE_NAMES = [
    'fe_adrian',
    'fe_jose',
    'fe_gloria',
    'fe_seba'
]

PRECOMPUTE_MODULE_PATH = os.path.join(FE_DIR, 'precalc_rescuer_features_cv.py')
precompute_rescuer_features = None
if os.path.exists(PRECOMPUTE_MODULE_PATH):
    try:
        spec_precompute = importlib.util.spec_from_file_location('precompute_rescuer_features', PRECOMPUTE_MODULE_PATH)
        if spec_precompute and spec_precompute.loader:
            precompute_rescuer_features = importlib.util.module_from_spec(spec_precompute)
            spec_precompute.loader.exec_module(precompute_rescuer_features)
            print("Módulo de pre-cálculo 'precompute_rescuer_features' cargado.")
        else:
            print(f"Advertencia: No se pudo cargar la especificación para {PRECOMPUTE_MODULE_PATH}")
    except Exception as e:
        print(f"Advertencia: No se pudo importar 'precompute_rescuer_features'. Error: {e}")
        print(traceback.format_exc())
else:
    print(f"Advertencia: Archivo de pre-cálculo no encontrado en {PRECOMPUTE_MODULE_PATH}")

def run_feature_engineering(input_df, output_path, is_train=True):
    print(f"\n--- Iniciando Feature Engineering {'para ENTRENAMIENTO' if is_train else 'para TEST'} ---")
    print(f"Input DF shape: {input_df.shape}")
    print(f"Output Path: {output_path}")

    dataset = input_df.copy()
    original_columns = set(dataset.columns)

    # --- 1. Merge con Features Precalculadas ---
    print("\nIntentando merge con features precalculadas...")
    added_feature_cols = []

    if is_train:
        oof_features_path = os.path.join(FEATURES_PRECOMPUTED_DIR, 'rescuer_ratios_oof.csv')
        if os.path.exists(oof_features_path):
            try:
                print(f"Cargando features OOF desde: {oof_features_path}")
                oof_features = pd.read_csv(oof_features_path, index_col=0)
                print(f"Features OOF cargadas. Shape: {oof_features.shape}")

                if not dataset.index.equals(oof_features.index.intersection(dataset.index)):
                    print("⚠️ Advertencia: Los índices del dataset y las features OOF no coinciden completamente.")

                oof_cols_original = oof_features.columns.tolist()
                rename_map_oof = {col: col.replace('_oof', '_rescuer_ratio') for col in oof_cols_original if '_oof' in col}
                added_feature_cols = list(rename_map_oof.values())

                dataset = pd.merge(
                    dataset,
                    oof_features,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                dataset.rename(columns=rename_map_oof, inplace=True)
                print(f"Merge OOF completado y columnas renombradas. Shape después del merge: {dataset.shape}")

                if added_feature_cols:
                    oof_nan_check = dataset[added_feature_cols].isnull().sum()
                    print("NaNs en columnas OOF (renombradas) después del merge:\n", oof_nan_check[oof_nan_check > 0])
                else:
                    print("No se encontraron columnas OOF para renombrar y verificar NaNs.")

            except Exception as e:
                print(f"❌ Error durante el merge con los ratios OOF: {e}")
                print(traceback.format_exc())
                print("⚠️ Advertencia: El pipeline continuará sin los ratios OOF.")
        else:
            print(f"⚠️ Advertencia: Archivo de ratios OOF no encontrado en {oof_features_path}. Saltando merge OOF.")
    else:
        agg_features_path = os.path.join(FEATURES_PRECOMPUTED_DIR, 'rescuer_ratios_full_train_agg.csv')
        if os.path.exists(agg_features_path):
            try:
                print(f"Cargando features agregadas desde: {agg_features_path}")
                try:
                    agg_features = pd.read_csv(agg_features_path, index_col='RescuerID')
                except ValueError:
                    agg_features = pd.read_csv(agg_features_path)
                    if 'RescuerID' not in agg_features.columns:
                        raise ValueError("El archivo agregado no contiene 'RescuerID' ni como índice ni como columna.")
                    agg_features.set_index('RescuerID', inplace=True)

                print(f"Features agregadas cargadas. Shape: {agg_features.shape}")

                if 'RescuerID' not in dataset.columns:
                    raise KeyError("'RescuerID' no encontrado en el dataset de test. No se puede hacer merge.")

                if dataset['RescuerID'].dtype != agg_features.index.dtype:
                    print(f"Advertencia: Tipo de dato de RescuerID difiere. Dataset: {dataset['RescuerID'].dtype}, Agregado: {agg_features.index.dtype}. Intentando convertir ambos a string.")
                    try:
                        dataset['RescuerID'] = dataset['RescuerID'].astype(str)
                        agg_features.index = agg_features.index.astype(str)
                    except Exception as cast_e:
                        print(f"❌ Error al convertir tipos de RescuerID: {cast_e}. Saltando merge.")
                        raise

                agg_cols_original = agg_features.columns.tolist()
                rename_map_agg = {col: col.replace('_agg', '_rescuer_ratio') for col in agg_cols_original if '_agg' in col}
                added_feature_cols = list(rename_map_agg.values())

                dataset = pd.merge(
                    dataset,
                    agg_features,
                    on='RescuerID',
                    how='left'
                )
                dataset.rename(columns=rename_map_agg, inplace=True)
                print(f"Merge agregado completado y columnas renombradas. Shape después del merge: {dataset.shape}")

                if added_feature_cols:
                    dataset[added_feature_cols] = dataset[added_feature_cols].fillna(0)
                    print("NaNs rellenados para features agregadas (renombradas).")
                else:
                    print("No se encontraron columnas agregadas para renombrar y rellenar NaNs.")

            except Exception as e:
                print(f"❌ Error durante el merge con los ratios agregados: {e}")
                print(traceback.format_exc())
                print("⚠️ Advertencia: El pipeline continuará sin los ratios agregados.")
        else:
            print(f"⚠️ Advertencia: Archivo de ratios agregados no encontrado en {agg_features_path}. Saltando merge agregado.")

    # Eliminar columnas ratio_AdoptionSpeedX si existen
    cols_to_drop = [col for col in dataset.columns if col.startswith('ratio_AdoptionSpeed')]
    if cols_to_drop:
        print(f"Eliminando columnas no deseadas: {cols_to_drop}")
        dataset.drop(columns=cols_to_drop, inplace=True)

    print("\nAplicando módulos de Feature Engineering...")

    for module_name in FE_MODULE_NAMES:
        try:
            module_path = os.path.join(FE_DIR, f"{module_name}.py")
            if not os.path.exists(module_path):
                print(f"⚠️ Advertencia: El archivo {module_path} no existe.")
                continue

            print(f"  -> Aplicando {module_name}...")

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                print(f"❌ Error: No se pudo crear la especificación o el loader para {module_path}")
                continue
            module = importlib.util.module_from_spec(spec)
            if module is None:
                print(f"❌ Error: No se pudo crear el módulo desde la especificación para {module_path}")
                continue

            spec.loader.exec_module(module)

            if hasattr(module, 'apply_features'):
                current_shape = dataset.shape
                try:
                    sig = inspect.signature(module.apply_features)
                    if 'is_train' in sig.parameters:
                        dataset = module.apply_features(dataset, is_train=is_train)
                        print(f"     ✅ {module_name} aplicado (con is_train). Shape: {current_shape} -> {dataset.shape}")
                    else:
                        dataset = module.apply_features(dataset)
                        print(f"     ✅ {module_name} aplicado (sin is_train). Shape: {current_shape} -> {dataset.shape}")
                except Exception as inner_e:
                    print(f"❌ Error dentro de {module_name}.apply_features:")
                    print(traceback.format_exc())
                    print(f"⚠️ Saltando el resto de las transformaciones debido al error en {module_name}.")
                    return None
            else:
                print(f"⚠️ Advertencia: {module_name} no tiene función apply_features().")

        except Exception as e:
            print(f"❌ Error al cargar o ejecutar el módulo {module_name}.py:")
            print(traceback.format_exc())
            print(f"⚠️ Saltando el resto de las transformaciones debido al error en {module_name}.")
            return None

    new_columns = set(dataset.columns) - original_columns
    print(f"\nNuevas columnas agregadas por el pipeline: {new_columns if new_columns else 'Ninguna'}")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_csv(output_path, index=False)
        print(f"\n✅ Dataset procesado guardado en {output_path}")
        print(f"Shape final: {dataset.shape}")
    except Exception as e:
        print(f"❌ Error al guardar el archivo en {output_path}: {e}")
        return None

    return dataset

if __name__ == "__main__":
    print("=============================================")
    print("=== Iniciando Ejecución del Pipeline Completo ===")
    print("=============================================")
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
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al cargar {TRAIN_INPUT_PATH}: {e}")
        sys.exit(1)

    TARGET_COLUMN = 'AdoptionSpeed'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    if TARGET_COLUMN not in original_df.columns:
        print(f"❌ Error: La columna objetivo '{TARGET_COLUMN}' no se encuentra en el dataset original.")
        sys.exit(1)

    print(f"\nDividiendo el dataset original en train/test ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
    train_df, test_df = train_test_split(
        original_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=original_df[TARGET_COLUMN]
    )
    print(f"División completada:")
    print(f"  Train split shape: {train_df.shape}")
    print(f"  Test split shape:  {test_df.shape}")

    # Guardar splits para referencia y para el precompute
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    # Guardar archivos vírgenes (sin feature engineering) para benchmark
    TRAIN_BASE_PATH = os.path.join(OUTPUT_DIR, 'train_base.csv')
    TEST_BASE_PATH = os.path.join(OUTPUT_DIR, 'test_base.csv')
    train_df.to_csv(TRAIN_BASE_PATH, index=False)
    test_df.to_csv(TEST_BASE_PATH, index=False)

    # --- 2. Pre-cálculo SOLO con el split de train ---
    if precompute_rescuer_features:
        print("\n--- Ejecutando Pre-cálculo para Rescuer Features SOLO con el split de train ---")
        try:
            precompute_rescuer_features.generate_rescuer_features(
                train_file=TRAIN_SPLIT_PATH,
                output_dir=FEATURES_PRECOMPUTED_DIR
            )
            print("--- Pre-cálculo de Rescuer Features completado ---")
        except Exception as e:
            print(f"❌ Error durante el pre-cálculo de Rescuer Features: {e}")
            print(traceback.format_exc())
            print("⚠️ Advertencia: El pipeline continuará, pero el merge de features precalculadas podría fallar.")

    # --- 3. Ejecutar Feature Engineering para el Split de Entrenamiento ---
    train_result_df = run_feature_engineering(
        input_df=train_df,
        output_path=TRAIN_OUTPUT_PATH,
        is_train=True
    )
    if train_result_df is None:
        print("\n❌ Pipeline de Feature Engineering para Entrenamiento falló.")
    else:
        print("\n✅ Pipeline de Feature Engineering para Entrenamiento completado exitosamente.")

    # --- 4. Ejecutar Feature Engineering para el Split de Test ---
    test_result_df = run_feature_engineering(
        input_df=test_df,
        output_path=TEST_OUTPUT_PATH,
        is_train=False
    )
    if test_result_df is None:
        print("\n❌ Pipeline de Feature Engineering para Test falló.")
    else:
        print("\n✅ Pipeline de Feature Engineering para Test completado exitosamente.")

    print("\n=============================================")
    print("=== Ejecución del Pipeline Finalizada ===")
    print("=============================================")

