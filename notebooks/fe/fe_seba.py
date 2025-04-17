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
    
    def categorizar_edad(dataset):
        """
        Crea una nueva columna 'AgeCategory' en el DataFrame basado en la edad de los animales.
        Categorías:
        - 1: Menos de 12 meses
        - 2: Entre 12 y 48 meses (inclusive)
        - 3: 48 meses o más
        """
        dataset['AgeCategory'] = dataset['Age'].apply(lambda x: 1 if x < 12 else (2 if x < 48 else 3))
        return dataset

    # Aplicar la función al dataset
    train = categorizar_edad(train)

    # Verificar los cambios
    print(train[['Age', 'AgeCategory']].head())
    #################################################
    # Crear la nueva columna 'State_importance' basada en la columna 'State'
    train['State_importance'] = train['State'].apply(lambda x: 1 if x == 41326 else (2 if x == 41401 else 3))

    # Verificar los cambios
    print(train[['State', 'State_importance']].head())
    
    
    
    
    # ---->>>>>>>> Más transformaciones...        <<<<<<<<<<---------------------
    
    return dataset








# Este código se ejecuta si el archivo se corre directamente,
# pero no cuando se importa desde otro script
if __name__ == "__main__":
    # Código para pruebas independientes
    dataset_fe = pd.read_csv('../../datasets_procesados/dataset_procesado.csv')
    result = apply_features(dataset_fe)
    print(f"Transformaciones aplicadas. Nuevas columnas: {set(result.columns) - set(dataset_fe.columns)}")