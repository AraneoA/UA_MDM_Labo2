# precompute_rescuer_features.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def generate_rescuer_features(train_file, output_dir, n_splits=5, random_state=42):
    """
    Calcula características OOF de ratio de AdoptionSpeed por RescuerID para el
    conjunto de entrenamiento y estadísticas agregadas para el conjunto de test.

    Args:
        train_file (str): Ruta al archivo CSV del dataset de entrenamiento procesado.
                          Debe contener 'RescuerID', 'Quantity', y 'AdoptionSpeed'.
        output_dir (str): Directorio donde se guardarán los archivos de features.
        n_splits (int): Número de folds para la validación cruzada.
        random_state (int): Semilla aleatoria para la división de folds.
    """
    print(f"Leyendo datos de entrenamiento desde: {train_file}")
    df_train = pd.read_csv(train_file)

    if 'AdoptionSpeed' not in df_train.columns:
        raise ValueError("La columna 'AdoptionSpeed' es necesaria para calcular los features.")
    if 'RescuerID' not in df_train.columns:
        raise ValueError("La columna 'RescuerID' es necesaria.")
    if 'Quantity' not in df_train.columns:
        print("Advertencia: No se encontró 'Quantity', usando conteo de filas (Quantity=1).")
        df_train['Quantity'] = 1 # Asumir Quantity=1 si no está

    target = df_train['AdoptionSpeed']
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Calcular Features OOF para el set de Entrenamiento ---
    # Determinar las clases presentes en el dataset
    adoption_speed_classes = sorted(df_train['AdoptionSpeed'].unique())
    oof_features = pd.DataFrame(index=df_train.index)
    for i in adoption_speed_classes:
        oof_features[f'rescuer_ratio_speed_{i}_oof'] = np.nan # Inicializar solo para clases presentes

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f"Calculando features OOF con {n_splits} folds...")

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train, target)):
        trn_data = df_train.iloc[trn_idx]
        val_data = df_train.iloc[val_idx]

        # Calcular estadísticas SOLO en el fold de entrenamiento (trn_data)
        grouped = trn_data.groupby(["RescuerID", "AdoptionSpeed"])["Quantity"].sum().unstack(fill_value=0)
        grouped.columns = grouped.columns.astype(str)
        grouped["Quantity_Total"] = grouped.sum(axis=1)

        # Calcular ratios solo para clases presentes en este fold
        ratio_cols_data = {}
        for speed_col in adoption_speed_classes:
            speed_col_str = str(speed_col)
            col_name = f"rescuer_ratio_speed_{speed_col_str}_oof"
            if speed_col_str in grouped.columns and "Quantity_Total" in grouped.columns:
                 denominator = grouped["Quantity_Total"].replace(0, np.inf)
                 ratio_cols_data[col_name] = grouped[speed_col_str] / denominator
            else:
                ratio_cols_data[col_name] = 0.0

        fold_ratios = pd.DataFrame(ratio_cols_data, index=grouped.index)

        val_merged = val_data[['RescuerID']].merge(fold_ratios, on="RescuerID", how='left')
        val_merged.fillna(0.0, inplace=True)

        for col in fold_ratios.columns:
             oof_features.loc[val_idx, col] = val_merged[col].values

    oof_features.fillna(0.0, inplace=True)
    oof_output_path = os.path.join(output_dir, 'rescuer_ratios_oof.csv')
    print(f"Guardando features OOF en: {oof_output_path}")
    oof_features.to_csv(oof_output_path, index=True)

    # --- 2. Calcular Estadísticas Agregadas usando TODO el set de Entrenamiento (para el Test) ---
    print("Calculando estadísticas agregadas en todo el set de entrenamiento...")
    grouped_full = df_train.groupby(["RescuerID", "AdoptionSpeed"])["Quantity"].sum().unstack(fill_value=0)
    grouped_full.columns = grouped_full.columns.astype(str)
    grouped_full["Quantity_Total"] = grouped_full.sum(axis=1)

    full_stats_data = {}
    for speed_col in adoption_speed_classes:
        speed_col_str = str(speed_col)
        col_name = f"rescuer_ratio_speed_{speed_col_str}_agg"
        if speed_col_str in grouped_full.columns and "Quantity_Total" in grouped_full.columns:
            denominator = grouped_full["Quantity_Total"].replace(0, np.inf)
            full_stats_data[col_name] = grouped_full[speed_col_str] / denominator
        else:
            full_stats_data[col_name] = 0.0

    full_stats = pd.DataFrame(full_stats_data, index=grouped_full.index)

    full_stats_output_path = os.path.join(output_dir, 'rescuer_ratios_full_train_agg.csv')
    print(f"Guardando estadísticas agregadas en: {full_stats_output_path}")
    full_stats.to_csv(full_stats_output_path, index=True)

    print("Pre-cálculo completado.")