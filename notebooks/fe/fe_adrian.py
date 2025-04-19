#Formato  base para archivo fe_tunombre.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime

def apply_features(df):
    """
    Aplica las transformaciones de feature engineering al dataframe.
    
    Args:
        df (pandas.DataFrame): El dataframe de entrada
        
    Returns:
        pandas.DataFrame: El dataframe con las nuevas features
    """
    # Copia del dataframe para no modificar el original
    dataset = df.copy()
    
   # Agrupar por RescuerID y sumar Quantity por adoption speed
    grouped = dataset.groupby(["RescuerID", "AdoptionSpeed"])["Quantity"].sum().unstack(fill_value=0)

    # Convert column names to strings (important if some columns are integers)
    grouped.columns = grouped.columns.astype(str)

    grouped["Quantity_Total"] = grouped.sum(axis=1)

    for speed_col in grouped.columns.drop("Quantity_Total"):
        grouped[f"ratio_AdoptionSpeed{speed_col}"] = grouped[speed_col] / grouped["Quantity_Total"]

    # Filter only the new ratio columns by checking for the substring
    new_ratio_cols = [col for col in grouped.columns if "ratio_AdoptionSpeed" in col]

    dataset = dataset.merge(grouped[new_ratio_cols],
                                left_on="RescuerID",
                                right_index=True,
                                how="left")
        
    return dataset








# Este código se ejecuta si el archivo se corre directamente,
# pero no cuando se importa desde otro script
if __name__ == "__main__":
    # Código para pruebas independientes
    dataset_fe = pd.read_csv('../../datasets_procesados/dataset_procesado.csv')
    result = apply_features(dataset_fe)
    print(f"Transformaciones aplicadas. Nuevas columnas: {set(result.columns) - set(dataset_fe.columns)}")