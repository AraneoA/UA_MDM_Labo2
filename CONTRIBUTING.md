# Guía de contribución al proyecto

## Flujo de trabajo

1. **Clona el repositorio** a tu computadora local
2. **Crea una rama personal** basada en la rama principal correspondiente:
   - Si trabajas en preprocesamiento: `git checkout -b preprocessing/tu-nombre/tarea preprocessing`
   - Si trabajas en EDA: `git checkout -b eda/tu-nombre/tarea eda`
3. **Desarrolla tu parte** del trabajo
4. **Sube tus cambios** regularmente
5. Cuando termines, **crea un Pull Request** hacia la rama base correspondiente

## Uso de GitHub Desktop

1. Para cambiar de rama: Haz clic en "Current Branch" y selecciona la rama base
2. Para crear tu rama: Haz clic en "Current Branch" > "New Branch"
3. Para subir cambios:
   - Selecciona los archivos modificados
   - Escribe un mensaje de commit descriptivo
   - Haz clic en "Commit to [tu-rama]"
   - Haz clic en "Push origin"

## Estructura de archivos

- `/data/`: Archivos de datos
- `/notebooks/preprocessing/`: Notebooks de preprocesamiento
- `/notebooks/eda/`: Notebooks de análisis exploratorio
