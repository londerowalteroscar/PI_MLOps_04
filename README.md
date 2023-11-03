<p align="center">
    <img src="https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png">
</p>

# PROYECTO INDIVIDUAL Nº1 
## LONDERO WALTER OSCAR 
### Machine Learning Operations (MLOps)

<p align="center">
    <img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png" height=300>
</p>
  
Bienvenidos al proyecto individual de **MLOps Engineer**.

---

Dirección del trabajo en Producción con <a href='https://deploy-pi-asi7.onrender.com/docs' > FastAPI </a>

# ETL-EDA

Este proyecto se enfoca en el análisis exploratorio de datos (EDA) y en la extracción, transformación y carga de datos (ETL) de conjuntos de datos proporcionados. A continuación se describe el flujo de trabajo y los procedimientos utilizados para analizar los datos de los juegos de Steam.

## Extracción
En esta sección, se emplean funciones específicas para extraer información de los archivos de datos proporcionados. Se utilizan dos funciones diferentes debido a las diferencias en el formato de los conjuntos de datos.

## Transformación
Aquí se detallan las transformaciones realizadas en los datos importados. Se extraen los diccionarios internos de los dataframes y se cambian los tipos de datos de las columnas según sea necesario. Además, se eliminan valores duplicados, filas nulas y columnas innecesarias, lo que ayuda a mantener la consistencia y reducir el tamaño de los conjuntos de datos.

## EDA
En esta sección, se muestra la información contenida en los conjuntos de datos y se presentan las medidas estadísticas más importantes para cada columna. Se lleva a cabo un análisis de outliers y una normalización de precios en el dataframe "df_games". Se realiza un análisis de sentimiento para comprender la naturaleza de las reseñas de los usuarios. Se presentan gráficos que muestran la evolución del sentimiento a lo largo de los años y la distribución de sentimientos en las reseñas.

---

El análisis EDA-ETL proporciona información valiosa sobre la calidad de los juegos en la plataforma de Steam, así como sobre las tendencias y patrones en las opiniones de los usuarios. Los resultados obtenidos de este análisis pueden ser utilizados para realizar mejoras en los productos y para comprender mejor las preferencias de los consumidores.



# Descripción de Funciones de la API

1. **PlayTimeGenre(genero: str)**:
   - Descripción: Devuelve el año con más horas jugadas para un género específico.
   - Ejemplo de Entrada: "Indie"
   - Respuesta Esperada: { "anio de lanzamiento con más horas jugadas para el género indie": 2006 }

2. **UserForGenre(genero: str)**:
   - Descripción: Devuelve el usuario que acumula más horas jugadas para un género dado y una lista de la acumulación de horas jugadas por año.
   - Ejemplo de Entrada: "Indie"
   - Respuesta Esperada: {
       "Usuario con más horas jugadas para Género indie": 765611983889673,
       "Horas jugadas": [
         {
           "anio": 2006,
           "Horas": 421652
         }
       ]
     }

3. **UsersRecommend(anio: int)**:
   - Descripción: Devuelve el top 3 de juegos MÁS recomendados por usuarios para un año dado.
   - Ejemplo de Entrada: 2015
   - Respuesta Esperada: {
       "Puesto 1": "counter-strike: global offensive",
       "Puesto 2": "garry's mod",
       "Puesto 3": "unturned"
     }

4. **UsersNotRecommend(anio: int)**:
   - Descripción: Devuelve el top 3 de juegos MENOS recomendados por usuarios para un año dado.
   - Ejemplo de Entrada: 2015
   - Respuesta Esperada: [
       { "Puesto 1": "counter-strike: global offensive" },
       { "Puesto 2": "payday 2" },
       { "Puesto 3": "unturned" }
     ]

5. **sentiment_analysis(year: int)**:
   - Descripción: Según el año de lanzamiento, devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
   - Ejemplo de Entrada: 2015
   - Respuesta Esperada: { "Negative": 1100, "Neutral": 891, "Positive": 2470 }

6. **recomendacion_juego_item_item(game_id:int)**:
   - Descripción: En este ejemplo, utiliza un algoritmo de Machine Learning llamado k-Nearest Neighbors (k-NN) para realizar recomendaciones de juegos.
   - Ejemplo de Entrada: 774277
   - Respuesta Esperada: {
       "games": [
         "SNOW - All Access Legend Pass",
         "SNOW - All Access Pro Pass",
         "SNOW - Starter Pack",
         "SNOW - Lifetime Pack",
         "High Profits"
       ]
     }
