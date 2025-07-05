import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import unicodedata
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 30)
plt.rcParams['figure.figsize'] = [12.0, 8.0]

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
    
    # Función para limpiar texto
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)  # Eliminar caracteres especiales
        text = re.sub(r'\b\w\b', ' ', text)  # Eliminar números o letras individuales
        text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra
        return text
    
    dataset['Description_limpia'] = dataset['Description'].apply(clean_text)
    
    # Generar nuevas columnas basadas en la descripción
    dataset['longitud_descripcion'] = dataset['Description_limpia'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    dataset['cantidad_palabras_no_stopwords'] = dataset['Description_limpia'].apply(
        lambda x: len([word for word in x.split() if word.lower() not in ENGLISH_STOP_WORDS]) if isinstance(x, str) else 0
    )
    dataset['stopwords_eliminadas'] = dataset['Description_limpia'].apply(
        lambda x: [word for word in x.split() if word.lower() in ENGLISH_STOP_WORDS] if isinstance(x, str) else []
    )
    dataset['descripcion_para_analisis'] = dataset['Description_limpia'].apply(
        lambda x: [word for word in x.split() if word.lower() not in ENGLISH_STOP_WORDS] if isinstance(x, str) else []
    )
    
    # Función para limpiar texto y dejar solo caracteres alfanuméricos y espacios
    def clean_text_unicode(text):
        if pd.isna(text):
            return ''
        text = ''.join(
            char for char in unicodedata.normalize('NFKD', text)
            if char.isalnum() or char.isspace()
        )
        text = re.sub(r'[ð]', '', text)  # Eliminar 'ð'
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Aplicar la función a la columna 'Name'
    dataset['Nombres_limpios'] = dataset['Name'].apply(clean_text_unicode)
    
    # Crear un DataFrame con la cantidad de adopciones por rescatista
    adoptados_por_rescatista = dataset.groupby('RescuerID')['Quantity'].sum().reset_index().sort_values(by='Quantity', ascending=False)
    adopciones_por_rescatista = dataset.groupby('RescuerID')['Quantity'].count().reset_index().sort_values(by='Quantity', ascending=False)

    # Calcular y mostrar los percentiles deseados
    percentiles = [1, 2, 5, 10] + list(range(80, 101, 1))
    for p in percentiles:
        valor = adopciones_por_rescatista['Quantity'].quantile(p / 100)
        print(f"Percentil {p}: {valor}")

    # Función para asignar estratos
    def asignar_estrato(valor):
        if valor <= adopciones_por_rescatista['Quantity'].quantile(0.82):
            return 1
        elif valor <= adopciones_por_rescatista['Quantity'].quantile(0.92):
            return 2
        elif valor <= adopciones_por_rescatista['Quantity'].quantile(0.98):
            return 3
        else:
            return 4

    # Asignar los estratos a cada rescatista
    adopciones_por_rescatista['Estrato'] = adopciones_por_rescatista['Quantity'].apply(asignar_estrato)
    dataset['categoria_rescatista'] = dataset['RescuerID'].map(adopciones_por_rescatista.set_index('RescuerID')['Estrato'])

    # Función para categorizar cantidad de animales
    def cantidad_animales(x):
        if x == 1:
            return 1
        elif x == 2 or x == 3:
            return 2
        else:
            return 3
        
    dataset['cantidad_animales'] = dataset['Quantity'].apply(cantidad_animales)

    # Función para determinar disponibilidad de imagen
    def disponibilidad_imagen(x, y):
        if pd.isna(x) and pd.isna(y):
            return 0
        elif pd.isna(x) or pd.isna(y):
            return 1
        elif x == 0 and y == 0:
            return 2
        elif x >= 1 and y == 0:
            return 3
        elif x == 0 and y >= 1:
            return 4
        elif x >= 1 and y >= 1:
            return 5
        else:
            return 6

    dataset['disponibilidad_imagen'] = dataset.apply(lambda row: disponibilidad_imagen(row['PhotoAmt'], row['VideoAmt']), axis=1)

    # Función para determinar estado sanitario
    def estado_sanitario(x, y):
        if x == 1 and y == 1:
            return 1  # Vacunado y desparasitado
        elif x == 1 and y == 2:
            return 2  # Vacunado pero no desparasitado
        elif x == 1 and y == 3:
            return 3  # Vacunado pero desparasitación no segura
        elif x == 2 and y == 1:
            return 4  # No vacunado pero desparasitado
        elif x == 2 and y == 2:
            return 5  # No vacunado y no desparasitado
        elif x == 2 and y == 3:
            return 6  # No vacunado y desparasitación no segura
        elif x == 3 and y == 1:
            return 7  # Vacunación no segura pero desparasitado
        elif x == 3 and y == 2:
            return 8  # Vacunación no segura y no desparasitado
        elif x == 3 and y == 3:
            return 9  # Vacunación y desparasitación no seguras
        else:
            return None

    dataset['estado_sanitario'] = dataset.apply(lambda row: estado_sanitario(row['Vaccinated'], row['Dewormed']), axis=1)

    return dataset