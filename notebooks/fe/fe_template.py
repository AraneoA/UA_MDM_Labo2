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
    
    # ACA LAS TRANSFORMACIONES, respetando que el nombre de la variable "dataset"
    # ---------->>>>>>>>>>>> Aquí tus transformaciones, por ejemplo: <<<<<<<<------------------
    # Agrega las variables ratio_adoption_speedN
    for i in range(5):
        dataset[f'ratio_AdoptionSpeed{i}'] = np.nan  # Tu código para calcular esta feature
    
    # ---->>>>>>>> Más transformaciones...        <<<<<<<<<<---------------------
    
    return dataset








# Este código se ejecuta si el archivo se corre directamente,
# pero no cuando se importa desde otro script
if __name__ == "__main__":
    # Código para pruebas independientes
    dataset_fe = pd.read_csv('../../datasets_procesados/dataset_procesado.csv')
    result = apply_features(dataset_fe)
    print(f"Transformaciones aplicadas. Nuevas columnas: {set(result.columns) - set(dataset_fe.columns)}")