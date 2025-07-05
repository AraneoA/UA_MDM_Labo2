import pandas as pd
import numpy as np
import os

def generate_rescuer_features(train_file, output_dir, min_count=1):
    """
    Calcula características de ratio de AdoptionSpeed por RescuerID usando TODO el train,
    pero solo para RescuerID con al menos min_count registros. El resto queda en 0.

    Args:
        train_file (str): Ruta al archivo CSV del dataset de entrenamiento procesado.
                          Debe contener 'RescuerID', 'Quantity', y 'AdoptionSpeed'.
        output_dir (str): Directorio donde se guardarán los archivos de features.
        min_count (int): Mínimo de registros requeridos para calcular el ratio.
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

    os.makedirs(output_dir, exist_ok=True)

    # Determinar las clases presentes en el dataset
    adoption_speed_classes = sorted(df_train['AdoptionSpeed'].unique())

    # Calcular cantidad de registros por RescuerID
    rescuer_counts = df_train.groupby('RescuerID')['Quantity'].sum()

    # Calcular ratios solo para RescuerID con suficientes registros
    grouped = df_train.groupby(['RescuerID', 'AdoptionSpeed'])['Quantity'].sum().unstack(fill_value=0)
    grouped['Quantity_Total'] = grouped.sum(axis=1)

    # Solo dejar rescuer con suficientes registros
    rescuer_valid = rescuer_counts[rescuer_counts >= min_count].index
    grouped_valid = grouped.loc[rescuer_valid]

    # Asegura que las columnas sean strings para el acceso correcto
    grouped_valid.columns = grouped_valid.columns.astype(str)

    # Calcular ratios para los válidos
    ratios = {}
    for speed_col in adoption_speed_classes:
        speed_col_str = str(speed_col)
        col_name = f"rescuer_ratio_speed_{speed_col_str}_agg"
        if speed_col_str in grouped_valid.columns:
            ratios[col_name] = grouped_valid[speed_col_str] / grouped_valid["Quantity_Total"].replace(0, np.inf)
        else:
            ratios[col_name] = 0.0
    ratios_df = pd.DataFrame(ratios, index=grouped_valid.index)
    ratios_df['Quantity_Total'] = grouped_valid['Quantity_Total']

    # Para los rescuer con menos de min_count, ponemos 0
    rescuer_all = pd.DataFrame(index=grouped.index)
    for col in ratios_df.columns:
        rescuer_all[col] = 0.0
    rescuer_all.loc[ratios_df.index, ratios_df.columns] = ratios_df

    # Guardar archivo de ratios agregados
    full_stats_output_path = os.path.join(output_dir, 'rescuer_ratios_full_train_agg_thres.csv')
    print(f"Guardando estadísticas agregadas en: {full_stats_output_path}")
    rescuer_all.to_csv(full_stats_output_path, index=True)

    print("Pre-cálculo con umbral de registros completado.")