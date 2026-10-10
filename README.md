# GPIA
Gestion de proyectos IA TECMILENIO
# Análisis Comparativo de Modelos de IA para Análisis de Sentimientos (PLN)

Este proyecto implementa y evalúa un pipeline de Procesamiento de Lenguaje Natural (PLN) para la clasificación automática de polaridad en opiniones de texto (reseñas), comparando el rendimiento, la eficiencia y la latencia de diferentes modelos preentrenados basados en transformadores.

---

## Descripción del Proyecto

El objetivo principal es medir la capacidad predictiva y el costo computacional de varios modelos punteros de *Hugging Face* aplicados a la clasificación de texto (sentimiento positivo/negativo). El pipeline extrae una muestra equilibrada del dataset estándar de IMDb, ejecuta inferencias controladas mediante pipelines de Hugging Face, y calcula métricas clave de rendimiento como *Accuracy*, *Precision*, *Recall*, *F1-score* y latencia promedio por muestra.

### Modelos Evaluados
1. **`siebert/sentiment-roberta-large-english`** (RoBERTa-large)
2. **`textattack/bert-base-uncased-SST-2`** (BERT-base fine-tuned)
3. **`distilbert/distilbert-base-uncased-finetuned-sst-2-english`** (DistilBERT)

---

## Entorno de Ejecución

El proyecto está diseñado para ejecutarse de forma óptima en entornos de nube interactivos con soporte de aceleración por hardware (GPU).

* **Plataforma recomendada:** Google Colab / Jupyter Notebook.
* **Acelerador de hardware:** GPU (ej. NVIDIA Tesla T4 o superior).
* **Librerías y dependencias principales:**
  * `transformers` (Hugging Face)
  * `datasets`
  * `torch` (PyTorch)
  * `scikit-learn`
  * `pandas` / `numpy`

---

## Pasos de Ejecución

Sigue los siguientes pasos para reproducir el experimento y la evaluación de los modelos:

### 1. Clonar o configurar el entorno
Asegúrate de instalar las dependencias necesarias en tu entorno de Python (si estás en Google Colab, la mayoría vienen preinstaladas, pero puedes actualizar las librerías de Hugging Face):
```bash
pip install --upgrade transformers datasets torch scikit-learn pandas

Ejecutar celda a celda
