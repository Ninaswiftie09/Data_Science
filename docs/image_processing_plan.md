# Plan para procesamiento de imágenes

## Objetivo

El objetivo es preparar las imágenes del alfabeto ASL para entrenar y comparar varios modelos de clasificación.

El modelo debe recibir una fotografía de una mano y predecir la clase que representa.

## Datos disponibles

El dataset contiene fotografías reales de manos.

Las imágenes están organizadas en carpetas según su clase.

Se espera encontrar 29 clases.

Las clases incluyen las letras de la A a la Z y las clases del, nothing y space.

El conjunto oficial de prueba tiene pocas imágenes. Por esta razón se creará un conjunto propio de entrenamiento, validación y prueba a partir de las imágenes de entrenamiento.

## Revisión inicial

Antes de entrenar se revisarán los siguientes puntos.

- Cantidad total de imágenes
- Cantidad de clases
- Cantidad de imágenes por clase
- Formato de las imágenes
- Resolución de las imágenes
- Modo de color
- Archivos dañados
- Variabilidad dentro de una misma clase
- Clases que se parecen visualmente

Se mostrarán ejemplos de al menos cinco clases.

También se mostrarán varias imágenes de una misma clase para observar cambios de posición, distancia, iluminación y fondo.

Se compararán las clases M, N y S.

También se compararán las clases U, V y R.

Estas clases pueden ser difíciles de separar porque la posición de los dedos es parecida.

## Submuestra

Se usarán como máximo 600 imágenes por clase.

La selección se hará de forma aleatoria con una semilla de 42.

La semilla permite repetir la selección y obtener los mismos conjuntos en otra computadora.

Si una clase tiene menos de 600 imágenes se usarán todas las imágenes disponibles de esa clase.

## División de datos

La división propuesta es la siguiente.

- 70 por ciento para entrenamiento
- 15 por ciento para validación
- 15 por ciento para prueba

La división se hará dentro de cada clase.

De esta forma cada conjunto tendrá una cantidad similar de imágenes de todas las clases.

El conjunto de entrenamiento se usará para aprender los parámetros de los modelos.

El conjunto de validación se usará para comparar configuraciones y controlar el sobreajuste.

El conjunto de prueba se usará al final para medir el resultado del modelo seleccionado.

## Cambio de resolución

Las imágenes se reducirán a 64 por 64 píxeles.

Esta resolución disminuye el tiempo de entrenamiento y el uso de memoria.

También conserva suficiente información para observar la forma general de la mano y los dedos.

El cambio de tamaño se hará con una interpolación de buena calidad.

## Modo de color

Todas las imágenes se convertirán a RGB.

Esto asegura que cada imagen tenga tres canales y evita errores causados por diferencias en el modo de color.

## Normalización

Los valores de los píxeles se dividirán entre 255.

Después de este cambio los valores estarán entre 0 y 1.

La normalización se hará durante la carga de las imágenes.

Las copias guardadas seguirán siendo archivos de imagen normales.

## Filtros

En la primera versión no se aplicarán filtros de desenfoque.

Tampoco se convertirán las imágenes a escala de grises.

El color, las sombras y los bordes pueden ayudar a reconocer la posición de la mano.

Más adelante se podrá comparar el desempeño con otras opciones si el tiempo lo permite.

## Aumento de datos

El aumento de datos se aplicará solamente al conjunto de entrenamiento.

Se probarán transformaciones leves.

- Rotaciones pequeñas
- Desplazamientos pequeños
- Zoom moderado
- Cambios leves de brillo
- Cambios leves de contraste

No se usará flip horizontal en la primera prueba.

Un flip horizontal cambia el lado y la orientación de la mano.

Este cambio puede producir una imagen que no representa la misma seña.

Tampoco se usarán rotaciones grandes, recortes fuertes o deformaciones que oculten los dedos.

## Modelos seleccionados

### CNN base

La primera CNN tendrá pocas capas.

Su objetivo será crear un resultado inicial fácil de entender.

Tendrá capas convolucionales, capas de reducción, una capa densa y una salida para las 29 clases.

### CNN mejorada

La segunda CNN tendrá más capacidad.

Se probarán más filtros, normalización por lotes, dropout y reducción global.

También se compararán diferentes tasas de aprendizaje y tamaños de lote.

### Red neuronal fully connected

Este modelo recibirá cada imagen como un vector de valores.

Se usará para comparar una red simple contra los modelos convolucionales.

Se espera que pierda parte de la información espacial porque los píxeles se convierten en una sola lista.

### HOG y SVM

HOG se usará para extraer información sobre bordes y direcciones.

SVM se usará para clasificar esas características.

Esta opción tiene sentido porque la forma de la mano y la dirección de los dedos son importantes para separar las clases.

## Parámetros que se compararán

En las CNN se probarán los siguientes parámetros.

- Cantidad de filtros
- Cantidad de capas
- Tamaño de lote
- Tasa de aprendizaje
- Dropout
- Cantidad de épocas

En SVM se probarán diferentes valores de C y diferentes tipos de kernel.

## Métricas

Los modelos se evaluarán con las siguientes métricas.

- Accuracy
- Precision macro
- Recall macro
- F1 macro
- Matriz de confusión
- Tiempo de entrenamiento

Las métricas macro darán el mismo peso a cada clase.

La matriz de confusión permitirá revisar cuáles letras se confunden con mayor frecuencia.

## Plan de trabajo

Primero se ejecutará el análisis exploratorio.

Después se generará la submuestra y la división de datos.

Luego se entrenará la CNN base.

Después se entrenará la CNN mejorada.

También se entrenará la red fully connected y el modelo HOG con SVM.

Los modelos se volverán a entrenar con aumento de datos cuando corresponda.

Al final se compararán los resultados y se elegirá el mejor modelo.

El mejor modelo se probará con fotografías tomadas por los integrantes del grupo.

## Riesgos

El dataset puede tener condiciones de iluminación y fondo que no representen un uso real.

También puede tener poca variedad de tonos de piel, tamaños de mano y ángulos de cámara.