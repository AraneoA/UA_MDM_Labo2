import pandas as pd
import numpy as np
import os

# --- Rutas a los archivos pre-calculados ---
# Deberían estar en una ubicación accesible por el pipeline
FEATURES_DIR = '../../features_precalculadas/rescuer_ratios'
OOF_FEATURES_PATH = os.path.join(FEATURES_DIR, 'rescuer_ratios_oof.csv')
FULL_TRAIN_STATS_PATH = os.path.join(FEATURES_DIR, 'rescuer_ratios_full_train_agg.csv')

# --- Variables globales para caché simple (opcional, evita recargar archivos) ---
_oof_data_cache = None
_full_stats_cache = None

def apply_features(df, is_train=True):
    """
    Aplica las características pre-calculadas de ratio de RescuerID al dataframe.

    Args:
        df (pandas.DataFrame): El dataframe de entrada (train o test).
        is_train (bool): Indicador de si df es el conjunto de entrenamiento.
                         True para cargar OOF, False para cargar agregados.

    Returns:
        pandas.DataFrame: El dataframe con las nuevas features unidas y nombres consistentes.
    """
    global _oof_data_cache, _full_stats_cache
    dataset = df.copy()

    if 'RescuerID' not in dataset.columns:
         print("Advertencia: La columna 'RescuerID' no está en el DataFrame. No se pueden unir los features.")
         return dataset # Devolver sin cambios si falta la clave

    new_feature_cols = [] # Para almacenar los nombres finales y consistentes

    if is_train:
        # --- Unir Features OOF (para set de entrenamiento) ---
        if not os.path.exists(OOF_FEATURES_PATH):
            print(f"Error: Archivo OOF no encontrado en {OOF_FEATURES_PATH}. Ejecuta el script de pre-cálculo.")
            return dataset # O lanzar un error

        if _oof_data_cache is None:
             print(f"Cargando features OOF desde: {OOF_FEATURES_PATH}")
             # Asumiendo que el archivo CSV se guardó con el índice original
             _oof_data_cache = pd.read_csv(OOF_FEATURES_PATH, index_col=0) # Asume que la col 0 es el índice

        # Identificar columnas originales OOF y crear mapa de renombrado
        original_oof_cols = [col for col in _oof_data_cache.columns if 'rescuer_ratio_speed_' in col]
        # Crear mapa para renombrar: quitar '_oof' y añadir un sufijo común si se desea, o simplemente quitarlo
        # Aquí usamos '_rescuer_ratio' como sufijo común. Ajusta si prefieres otro.
        rename_map_oof = {col: col.replace('_oof', '_rescuer_ratio') for col in original_oof_cols}
        new_feature_cols = list(rename_map_oof.values()) # Guardar los nombres finales

        # Unir usando el índice del DataFrame (debe coincidir con el índice guardado)
        # Unimos solo las columnas necesarias
        dataset = dataset.merge(_oof_data_cache[original_oof_cols], left_index=True, right_index=True, how='left')
        # Renombrar columnas para consistencia
        dataset.rename(columns=rename_map_oof, inplace=True)
        # Los nombres de columna ahora son consistentes, p.ej., '..._rescuer_ratio'

    else:
        # --- Unir Estadísticas Agregadas (para set de test) ---
        if not os.path.exists(FULL_TRAIN_STATS_PATH):
            print(f"Error: Archivo de estadísticas agregadas no encontrado en {FULL_TRAIN_STATS_PATH}. Ejecuta el script de pre-cálculo.")
            return dataset # O lanzar un error

        if _full_stats_cache is None:
            print(f"Cargando estadísticas agregadas desde: {FULL_TRAIN_STATS_PATH}")
            # Asumiendo que el archivo se guardó con RescuerID como índice
            _full_stats_cache = pd.read_csv(FULL_TRAIN_STATS_PATH, index_col='RescuerID')

        # Identificar columnas originales agregadas y crear mapa de renombrado
        original_agg_cols = [col for col in _full_stats_cache.columns if 'rescuer_ratio_speed_' in col]
        # Crear mapa para renombrar: quitar '_agg' y añadir el mismo sufijo común '_rescuer_ratio'
        rename_map_agg = {col: col.replace('_agg', '_rescuer_ratio') for col in original_agg_cols}
        new_feature_cols = list(rename_map_agg.values()) # Guardar los nombres finales

        # Unir usando la columna 'RescuerID'
        # Unimos solo las columnas necesarias
        dataset = dataset.merge(_full_stats_cache[original_agg_cols], left_on='RescuerID', right_index=True, how='left')
        # Renombrar columnas para consistencia
        dataset.rename(columns=rename_map_agg, inplace=True)
        # Los nombres de columna ahora son consistentes, p.ej., '..._rescuer_ratio'


    # --- Manejo de NaNs Post-Merge ---
    # Puede haber NaNs si un RescuerID en 'dataset' (especialmente en test)
    # no estaba presente en los datos usados para calcular los features,
    # o si hubo algún problema en el merge de índices para OOF.
    # Llenar con 0 es una estrategia común.
    # Usamos los nombres de columna ya renombrados y consistentes
    if new_feature_cols: # Asegurarse de que hay columnas para llenar
        dataset[new_feature_cols] = dataset[new_feature_cols].fillna(0.0)
        print(f"Features de ratio de RescuerID unidas y renombradas: {new_feature_cols}")
    else:
        print("No se encontraron columnas de ratio de RescuerID para unir o renombrar.")


    return dataset
