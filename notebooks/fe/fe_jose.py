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
    
    # Concatenar Color y crea nueva variable Color
    dataset['Color'] = (
    dataset['Color1'].astype(str) +
    dataset['Color2'].astype(str) +
    dataset['Color3'].astype(str)
    ).astype('int64')
    
    # Se agrega variable dicotomica para saber si tiene nombre o no
    dataset['Tiene_Nombre'] = np.where(dataset['Name'].notnull() & (dataset['Name'].str.strip() != ''), 1, 0)
    
    # Se agrega variable dicotomica para saber si es gratis o no
    dataset['Es_Gratis'] = ((dataset['Fee'] == 0).astype(int))  
    
    
    # Se agrega variable para saber si a mascotas más grandes y más caras
    dataset['Age_Fee'] = data['Age'] * data['Fee']  
    
    return dataset








# Este código se ejecuta si el archivo se corre directamente,
# pero no cuando se importa desde otro script
if __name__ == "__main__":
    # Código para pruebas independientes
    dataset_fe = pd.read_csv('../../datasets_procesados/dataset_procesado.csv')
    result = apply_features(dataset_fe)
    print(f"Transformaciones aplicadas. Nuevas columnas: {set(result.columns) - set(dataset_fe.columns)}")