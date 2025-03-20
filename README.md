# Analizador de Series Temporales

Un módulo de Python para analizar series temporales y generar pronósticos utilizando Facebook Prophet.

## Características

- Carga de datos de series temporales desde archivos CSV o DataFrames de pandas
- Generación de resúmenes estadísticos de los datos
- Visualización de datos de series temporales
- Generación de pronósticos utilizando Facebook Prophet
- Visualización de pronósticos con intervalos de confianza

## Instalación

1. Clona este repositorio
2. Instala las dependencias requeridas:
```bash
pip install -r requirements.txt
```

## Uso

```python
from time_series_analyzer import TimeSeriesAnalyzer

# Inicializar el analizador con tus datos
analyzer = TimeSeriesAnalyzer("tu_archivo.csv", "nombre_columna_fecha")

# Obtener resumen estadístico
summary = analyzer.get_statistical_summary()
print(summary)

# Graficar datos de series temporales
analyzer.plot_time_series()

# Generar y graficar pronóstico para una columna específica
analyzer.plot_forecast("columna_objetivo", periods=30)
```

## Ejemplo

```python
# Usando el conjunto de datos mock_kaggle.csv
analyzer = TimeSeriesAnalyzer("mock_kaggle.csv", "data")

# Obtener resumen estadístico
print(analyzer.get_statistical_summary())

# Graficar todas las series temporales
analyzer.plot_time_series()

# Pronosticar ventas para los próximos 30 días
analyzer.plot_forecast("venda", periods=30)
```

## Parámetros

- `data`: Ruta a un archivo CSV o un DataFrame de pandas
- `date_column`: Nombre de la columna que contiene las fechas
- `target_column`: Nombre de la columna a pronosticar
- `periods`: Número de períodos a pronosticar (predeterminado: 30)
- `changepoint_prior_scale`: Flexibilidad de la tendencia (predeterminado: 0.05)

## Dependencias

- pandas
- numpy
- prophet
- matplotlib
- seaborn
