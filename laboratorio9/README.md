# Laboratorio 9: visualización estática

Análisis de ingresos de turistas y excursionistas a Guatemala entre enero de 2009 y junio de 2026, con datos de INGUAT/IGM y resultados del Laboratorio 1.

## Estructura

- `scripts/`: análisis, exportación de gráficos y validación de resultados.
- `datos/`: series mensuales, indicadores, métricas, predicciones y registros de validación.
- `figuras/`: siete gráficos en PNG a 300 DPI y SVG.
- `docs/`: documento de explicación de diseño y entregas finales en PDF y PNG.

## Ejecución

Desde la raíz del repositorio, con Python 3.11 o posterior:

```powershell
python -m pip install -r laboratorio9/requirements.txt
python laboratorio9/scripts/generar_recursos.py
python laboratorio9/scripts/validar_recursos.py
```

El análisis lee la hoja Datos de la base original y las métricas guardadas en `notebooks/02_resultados.ipynb`. Conserva los archivos originales y escribe los resultados en `laboratorio9/datos` y `laboratorio9/figuras`. Prophet requiere su backend CmdStan funcional. Las versiones y la configuración se registran en `requirements.txt` y `datos/configuracion_ejecucion.json`.

## Criterios del análisis

Se incluyen únicamente Turista y Excursionista. Los valores representan ingresos registrados, no personas únicas. Se conservan los valores decimales de la base. El año 2026 contiene solo enero a junio.

Las métricas originales del notebook conservan tres decimales. Las métricas recalculadas con precisión completa y sus diferencias se encuentran en `datos/metricas_modelos_reproducidas.csv`. La evaluación utiliza abril de 2021 a junio de 2026 como conjunto de prueba.

Prophet utiliza un intervalo nominal del 80 %, con 1000 simulaciones y semilla 2026. Su cobertura observada fue de 10 de 63 meses (15.87 %). Se trata de una evaluación retrospectiva; el nivel nominal no garantiza cobertura futura. La reproducción numérica puede variar entre plataformas y versiones.

Fuente de uso académico: Base de Migración 2009–junio 2026, INGUAT/IGM.
