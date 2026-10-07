# Proyecto-estad-sticas-de-Youtube-Avances-II

# Predicción de Desempeño de Vídeos de YouTube mediante NLP

Este proyecto implementa un flujo de trabajo reproducible conforme a prácticas de **MLOps** para predecir el desempeño (vistas o me gusta) de vídeos de YouTube utilizando el procesamiento de lenguaje natural (NLP) sobre los comentarios y títulos de los vídeos, combinados con variables estadísticas agregadas.

---

##  Estructura del Proyecto y Flujo de Trabajo

El cuaderno está estructurado de forma secuencial y modular para garantizar la reproducibilidad completa:

1. **Decisiones Metodológicas:** Definición de la unidad de análisis, variables objetivo (`Views` o `Likes` transformadas con `log1p`), ingeniería de características lingüísticas y riesgos documentados de los datos.
2. **Instalación e Importaciones:** Configuración del entorno virtual e importación de librerías esenciales (`kagglehub`, `pandera`, `sklearn`, `scipy`).
3. **Acceso Reproducible:** Descarga automatizada y directa del dataset `advaypatil/youtube-statistics` desde Kaggle utilizando la API oficial (`kagglehub`).
4. **Linaje y Versionado:** Creación de un manifiesto de datos (`data_manifest.json`) registrando hashes SHA-256 de los archivos crudos y marcas temporales.
5. **Validación de Esquema y Calidad:** Verificación de tipos de datos, columnas requeridas y reporte detallado de calidad (valores nulos, registros duplicados, consistencia referencial).
6. **Limpieza Controlada:** Tratamiento de valores ausentes (mapeo de `-1` a nulos), eliminación de registros duplicados e imputación.
7. **Ingeniería de Características:** Generación de variables a nivel de vídeo (métricas de longitud, ratios de sentimiento positivo/neutro/negativo, etc.) y concatenación del texto del título con los comentarios (`combined_text`).
8. **Análisis Estadístico Avanzado:** Correlaciones de Spearman con intervalos de confianza calculados vía Bootstrap al 95%, test no paramétrico de Kruskal-Wallis y normalización Z-score por Keyword.
9. **Modelado y Pipeline Pipeline:**
   - **Baseline:** Modelo de referencia basado en la mediana (`DummyRegressor`).
   - **Modelo NLP:** Pipeline robusto que combina TF-IDF de palabras, TF-IDF de caracteres (ngram de 3 a 5 para capturar morfología y emojis) y estandarización de características numéricas con regresión Ridge.
10. **Evaluación de Suficiencia y Estabilidad:** Evaluación mediante validación cruzada (`KFold`), métricas de regresión (MAE, RMSE, R², RMSLE) y análisis de la curva de aprendizaje/suficiencia.
11. **Registro de Artefactos:** Serialización local del mejor modelo entrenado (`.joblib`) y generación automática de la ficha técnica del modelo (`model_card.json`).

---

##  Estructura del Dataset

El proyecto se alimenta de dos conjuntos de datos principales de Kaggle:
- **`videos-stats.csv`:** Información general del vídeo (ID, Título, Fecha de publicación, Keyword, Vistas, Likes, etc.).
- **`comments.csv`:** Comentarios de los usuarios con métricas de sentimiento precalificadas e interacción.

Ambos archivos se relacionan mediante el identificador único `Video ID`.

---

##  Instrucciones de Ejecución

Para ejecutar este cuaderno de manera reproducible, asegúrate de seguir estos pasos:

1. **Autenticación en Kaggle:** Si el entorno te lo solicita, configura tus credenciales de Kaggle (`KAGGLE_USERNAME` y `KAGGLE_KEY`) o utiliza las opciones de inicio de sesión integradas en Google Colab.
2. **Definición del Target:** En la sección **2 (Instalación e importaciones)** puedes alternar la variable objetivo modificando la variable:
   ```python
   TARGET = 'Views' # O cámbialo a 'Likes'
   ```
3. **Ejecución Completa:** Ejecuta todas las celdas secuencialmente en Google Colab o tu entorno Jupyter local.

---

##  Artefactos Generados

Tras completar el flujo, los siguientes archivos se guardarán automáticamente en la carpeta `./artifacts/`:
- `data_manifest.json` — Registro de procedencia y hashes de los datos fuente.
- `data_quality_columns.csv` & `data_quality_checks.json` — Resultados del reporte de calidad.
- `metrics_{target}.csv` — Tabla comparativa de métricas del baseline vs modelo NLP.
- `model_{target}_tfidf_ridge.joblib` — Pipeline final de Sklearn completamente entrenado y listo para inferencia.
- `model_card_{target}.json` — Ficha técnica documentando limitaciones, hiperparámetros y contexto del modelo.
