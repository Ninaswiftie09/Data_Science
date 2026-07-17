# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    PROCESSED_DIR
)


# Cargar los datos
df = cargar_datos()

# Crear la carpeta de resultados
crear_carpeta_processed()


# Cantidad de meses para prueba
MESES_PRUEBA = 36


# Separar los datos
train = df.iloc[:-MESES_PRUEBA].copy()
test = df.iloc[-MESES_PRUEBA:].copy()


# Mostrar resultados
print("\n--- DIVISIÓN DE DATOS ---")

print("Total de registros:", len(df))
print("Registros de entrenamiento:", len(train))
print("Registros de prueba:", len(test))


# Fechas de entrenamiento
print("\nEntrenamiento:")
print("Fecha inicial:", train["month"].min())
print("Fecha final:", train["month"].max())


# Fechas de prueba
print("\nPrueba:")
print("Fecha inicial:", test["month"].min())
print("Fecha final:", test["month"].max())


# Verificar los 36 meses
if len(test) == MESES_PRUEBA:
    print("\nEl conjunto de prueba tiene 36 meses.")
else:
    print("\nEl conjunto de prueba no tiene 36 meses.")


# Guardar los archivos
train.to_csv(
    PROCESSED_DIR / "train.csv",
    index=False
)

test.to_csv(
    PROCESSED_DIR / "test.csv",
    index=False
)

print("\nArchivos guardados correctamente.")