import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

def detect_playpens(input_csv, images_dir, output_csv=None):
    """
    Analiza imágenes para detectar la presencia de estructuras tipo 'playpen' (corralitos)
    usando ResNet50 pre-entrenado con ImageNet.
    
    Args:
        input_csv (str): Ruta al CSV con columna 'PetID'
        images_dir (str): Directorio donde están las imágenes (formato: {PetID}-1.jpg)
        output_csv (str): Ruta donde guardar el CSV con resultados (opcional)
    
    Returns:
        pandas.DataFrame: DataFrame original con columna adicional 'Prob_TienePlaypen'
    """
    print("Cargando modelo ResNet50 pre-entrenado...")
    
    # Cargar el modelo ResNet50 con pesos pre-entrenados de ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()  # Poner en modo evaluación
    
    # Usar GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Transformaciones para las imágenes (mismas que usa ResNet en ImageNet)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Cargar las 1000 categorías de ImageNet
    # Podemos encontrar el archivo de categorías en la documentación o repositorios de PyTorch
    # También lo obtendremos manualmente para este caso
    
    # Clases de ImageNet más relacionadas con playpen/jaulas/rejas
    # Los índices están basados en la clasificación de ImageNet
    # (Los índices reales pueden variar según la implementación específica)
    playpen_related_classes = {
        'crib': 520,         # Cuna (con barrotes)
        'playpen': 666,      # Corralito (estructura con barrotes para niños)
        'chain_link_fence': 489,  # Cerca/valla de eslabones de cadena
        'window_screen': 904,  # Mosquitero de ventana (estructura de malla)
        'iron_bars': 567,    # Barras de hierro/rejas
        'cage': 412         # Jaula
    }
    
    # Cargar el DataFrame
    print(f"Cargando datos desde {input_csv}...")
    df = pd.read_csv(input_csv)
    if 'PetID' not in df.columns:
        raise ValueError("El CSV debe contener una columna 'PetID'")
    
    # Lista para almacenar las probabilidades
    playpen_probs = []
    
    # Procesar cada imagen
    print("Analizando imágenes para detectar estructuras tipo playpen/jaula...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pet_id = row['PetID']
        img_path = os.path.join(images_dir, f"{pet_id}-1.jpg")
        
        if not os.path.exists(img_path):
            print(f"Advertencia: No se encontró la imagen para PetID {pet_id}")
            playpen_probs.append(0.0)  # Valor por defecto
            continue
        
        try:
            # Cargar y preprocesar la imagen
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)  # Añadir dimensión de batch
            
            # Obtener las predicciones del modelo
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Calcular un "score" combinando las probabilidades de las clases relacionadas con playpen
            playpen_score = 0.0
            for class_name, class_idx in playpen_related_classes.items():
                class_prob = probabilities[class_idx].item()
                playpen_score = max(playpen_score, class_prob)  # Tomamos la probabilidad más alta
                # Alternativa: playpen_score += class_prob  # Suma ponderada
            
            playpen_probs.append(playpen_score)
            
        except Exception as e:
            print(f"Error procesando imagen {img_path}: {e}")
            playpen_probs.append(0.0)  # Valor por defecto
    
    # Añadir la columna de probabilidades al DataFrame
    df['Prob_TienePlaypen'] = playpen_probs
    
    # Guardar resultados si se especificó un archivo de salida
    if output_csv:
        print(f"Guardando resultados en {output_csv}...")
        df.to_csv(output_csv, index=False)
    
    return df

if __name__ == "__main__":
    # Rutas fijas para este proyecto
    input_csv = "datasets_procesados/test_final_thres.csv"
    images_dir = "input/petfinder-adoption-prediction/train_images"
    output_csv = "datasets_procesados/test_final_thres_con_jaula.csv"
    
    # Asegurar que las rutas sean relativas al directorio principal del proyecto
    # Si estás ejecutando desde una ubicación distinta, puedes necesitar ajustar estas rutas
    # o construirlas relativamente desde la ubicación del script
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv = os.path.join(base_dir, input_csv)
    images_dir = os.path.join(base_dir, images_dir)
    output_csv = os.path.join(base_dir, output_csv)
    
    print(f"CSV de entrada: {input_csv}")
    print(f"Directorio de imágenes: {images_dir}")
    print(f"CSV de salida: {output_csv}")
    
    result_df = detect_playpens(input_csv, images_dir, output_csv)
    
    print("Procesamiento completado.")
    print(f"Primeras 5 filas con probabilidades:")
    print(result_df[['PetID', 'Prob_TienePlaypen']].head())