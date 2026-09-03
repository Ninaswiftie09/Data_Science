# Laboratorio 6

## Integrantes

- Nina Nájera Marakovits, 231088

## Repositorio

https://github.com/Ninaswiftie09/Data_Science/tree/Laboratorio6

## Contenido del avance

Este avance cubre las actividades 1 a 4 del laboratorio.

- Carga y comprensión de los datos
- Integración por video_id
- Diagnóstico de calidad
- Limpieza y preparación de texto
- Análisis exploratorio
- Construcción de la red bipartita autor video
- Tabla de nodos y tabla de aristas
- Visualización de la red completa

Los resultados están en estos notebooks.

- notebooks/01.Analisis_Exploratorio.ipynb
- notebooks/02.Red_Bipartita.ipynb

El código reutilizable está en la carpeta scripts. Los archivos originales están en data/raw y las tablas procesadas están en data/processed.

## Instalación

Se recomienda usar Python 3.11 o una versión más reciente.

```bash
python -m venv .venv
```

En Windows se activa el entorno con este comando.

```bash
.venv\Scripts\activate
```

Después se instalan las dependencias.

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Para procesar los datos y crear las tablas se usa este comando desde la raíz del proyecto.

```bash
python -m scripts.analisis_avance
```

Para crear de nuevo los notebooks se usa este comando.

```bash
python -m scripts.crear_notebooks
```

Luego se abren los notebooks con Jupyter y se ejecutan todas las celdas.

```bash
jupyter notebook
```

Para comprobar las llaves, la unión, los pesos y la estructura bipartita se usa este comando.

```bash
python -m scripts.verificar_avance
```

## Decisiones principales

Los identificadores se mantienen como llaves y los nombres se usan solo como etiquetas. El texto original se conserva. La copia limpia elimina URL, menciones, puntuación, números, emojis y palabras vacías. Los hashtags conservan la palabra sin el símbolo. No se aplica lematización porque el proyecto todavía no tiene un modelo de español validado.

reply_count no se usa para crear relaciones entre autores porque el archivo no indica quién respondió. Una arista de la red solo representa que un autor comentó en un video.

Los resultados describen la muestra recolectada. No representan a todos los usuarios de YouTube ni a toda la población de Guatemala.
