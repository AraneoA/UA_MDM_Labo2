```markdown
# MCD - LAB. DE IMP. 2 - GRUPO 12 

**Guía para colaboradores**  
Proyecto estructurado para análisis de datos sobre adopción de mascotas, incluyendo preprocesamiento, EDA e integración de resultados.

---

## 🏗️ Estructura del Repositorio

### Ramas Principales
- `main`: Rama principal (solo integraciones validadas).
- `preprocessing`: Rama para limpieza y preparación de datos.
- `eda`: Rama para análisis exploratorio y visualizaciones.
- `integration`: Rama final para consolidar todos los análisis.

### Carpetas Clave

├── data/
│   ├── raw/          # Datos crudos originales
│   └── processed/    # Datos limpios (generados en preprocesamiento)
├── notebooks/
│   ├── preprocessing/  # Notebooks de limpieza de datos
│   └── eda/            # Notebooks de análisis exploratorio
├── src/              # Funciones reutilizables (.py)
├── requirements.txt  # Dependencias del proyecto
└── CONTRIBUTING.md   # Guía de contribución detallada


---

## 🛠️ Configuración Inicial

### Para Nuevos Colaboradores
1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/coordinador/nombre-repo.git
   cd nombre-repo
   ```

2. **Configurar upstream** (sincronizar con el repo del profesor):
   ```bash
   git remote add upstream https://github.com/profesor/nombre-repo.git
   ```

3. **Crear rama personal** (ejemplo para preprocesamiento):
   ```bash
   git checkout preprocessing
   git checkout -b preprocessing/tu-usuario/nulls
   ```

---

## 🔄 Flujo de Trabajo

### Pasos para Contribuir
1. **Sincronizar cambios** antes de empezar:
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

2. **Desarrollar en tu rama**:
   - Usar nombres descriptivos para notebooks (ej: `preprocessing_nulls.ipynb`).
   - Hacer commits frecuentes con mensajes claros:
     ```bash
     git add archivo_modificado.ipynb
     git commit -m "Add imputación de nulos para edad usando media"
     ```

3. **Subir cambios y crear Pull Request (PR)**:
   - Dirigir PRs a la rama base correspondiente (`preprocessing`, `eda`, etc.).
   - Asignar revisores y describir cambios detalladamente.

---

## ✅ Buenas Prácticas

### Documentación
- **Notebooks**: Incluir objetivos, metodología y conclusiones.
- **Funciones compartidas**: Guardar en `.py` dentro de `src/`.
- **Commits**: Usar verbos en imperativo (ej: "Fix encoding error in breed column").

### Dependencias
Mantener actualizado `requirements.txt`:
```
pandas==1.5.3
numpy==1.24.2
matplotlib==3.7.1
seaborn==0.12.2
```

---

## 🚨 Resolución de Conflictos
1. **Prevención**:
   - Comunicar qué archivos se están modificando.
   - Hacer PRs pequeños y frecuentes.

2. **Solución**:
   ```bash
   git status  # Identificar archivos conflictivos
   # Editar manualmente y conservar cambios necesarios
   git add archivo_resuelto.ipynb
   git commit -m "Resuelto conflicto en análisis de regiones"
   ```

---

## 📌 Gestión del Proyecto
- **Issues**: Crear tareas con etiquetas (`preprocessing`, `eda`, `bug`).
- **Milestones**: Asignar fechas límite para cada fase.
- **Reuniones**: 
  - Stand-ups semanales (15 min).
  - Revisión de integración al final de cada fase.

---

## 👥 Roles del Equipo
| Rol | Responsabilidades |
|------|-------------------|
| Coordinador | Gestionar ramas principales y PRs |
| Especialistas en Preprocesamiento (2) | Limpieza de datos, tratamiento de outliers |
| Analistas de Datos (3) | Análisis exploratorio y visualizaciones |

---

📌 **¿Cómo Contribuir?**  
Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para detalles técnicos y convenciones de equipo.
``` 

Este README resume la estructura, flujo de trabajo y normas del proyecto de forma clara y accesible. Incluye comandos prácticos, buenas prácticas y enlaces a documentación adicional.
