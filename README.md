
# Clasificación de Radiografías de Tórax para Diagnóstico de COVID-19

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un sistema que clasifique radiografías de tórax en tres categorías: COVID-19, neumonía no COVID-19 y casos normales. Para lograr esto, se utilizan técnicas de análisis de sistemas complejos y redes neuronales, con un enfoque en el análisis multifractal de fluctuaciones sin tendencia (MF-DFA 2D).

## Resumen del Proyecto

El proyecto utiliza el análisis multifractal para identificar patrones en las radiografías de tórax que puedan diferenciar entre pacientes con COVID-19 y aquellos sin la enfermedad. Se desarrolló una librería en Python para aplicar MF-DFA 2D y obtener características representativas de las imágenes, las cuales son utilizadas por un algoritmo de clasificación entrenado para distinguir entre las categorías mencionadas.

## Estructura del Código

1. **Preprocesamiento de Datos**:
    - Conversión de imágenes a escala de grises y normalización.
    - Segmentación de la región pulmonar.

2. **Cálculo del Espectro Multifractal**:
    - Utilización de la librería `MF-DFA 2D` para analizar las radiografías y obtener el espectro multifractal.
    - Extracción de características clave como el alfa mínimo, máximo y el ancho del espectro.

3. **Entrenamiento del Algoritmo Clasificador**:
    - División de los datos en conjuntos de entrenamiento (80%) y prueba (20%).
    - Entrenamiento de un modelo de clasificación utilizando las características multifractales.

4. **Evaluación y Resultados**:
    - Validación del modelo con un conjunto de prueba.
    - Análisis de la matriz de confusión y curva ROC para medir la precisión del clasificador.

5. **Interfaz Gráfica de Usuario**:
    - Implementación de una GUI utilizando Tkinter para cargar radiografías y mostrar los resultados de clasificación.

## Requisitos

- Python 3.x
- Librerías: Matplotlib, NumPy, TensorFlow, Scikit-learn, Pandas, Tkinter, OpenCV
