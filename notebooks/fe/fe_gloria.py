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
    
   def clean_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()  
    text = re.sub(r'\W+', ' ', text)  # Eliminar caracteres especiales
    text = re.sub(r'\b\w\b', ' ', text)  # Eliminar números o letras individuales
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra
    return text

    dataset['Description_limpia'] = dataset['Description'].apply(clean_text)
    
    def get_removed_special_chars(original, cleaned):
    if pd.isna(original):  # Si el texto original es NaN
        return []
    # Convertir ambos textos a conjuntos de caracteres
    original_chars = set(original)  # Conjunto de caracteres del texto original
    cleaned_chars = set(cleaned)    # Conjunto de caracteres del texto limpio
    # Identificar los caracteres que están en el original pero no en el limpio
    removed_chars = original_chars - cleaned_chars
    # Mostrar todo lo eliminado (caracteres especiales, números y letras individuales)
    all_removed_chars = [char for char in removed_chars if not char.isspace()]
    return all_removed_chars

# Aplicar la función para generar la nueva columna
dataset['Removed_Special_Chars'] = dataset.apply(
    lambda row: get_removed_special_chars(row['Description'], row['Description_limpia']), axis=1
)


    return dataset










# Este código se ejecuta si el archivo se corre directamente,
# pero no cuando se importa desde otro script
if __name__ == "__main__":
    # Código para pruebas independientes
    dataset_fe = pd.read_csv('../../datasets_procesados/dataset_procesado.csv')
    result = apply_features(dataset_fe)
    print(f"Transformaciones aplicadas. Nuevas columnas: {set(result.columns) - set(dataset_fe.columns)}")