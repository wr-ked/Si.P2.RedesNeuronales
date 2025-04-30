from tensorflow import keras
import matplotlib.pyplot as plt
from random import sample

# Cargar los datos
(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

# Ver estructura de los datos
print(X_train.shape, X_train.dtype)
print(Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype)
print(Y_test.shape, Y_test.dtype)

# Lista de nombres de clases
clases = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Función para mostrar imágenes
def show_image(imagen, titulo):
    plt.figure()
    plt.suptitle(titulo)
    plt.imshow(imagen)  # Sin cmap para ver colores reales
    plt.axis('off')
    plt.show()

# Mostrar 3 imágenes aleatorias
for i in sample(list(range(len(X_train))), 3):
    clase = clases[int(Y_train[i])]
    titulo = f"Imagen X_train[{i}] - Clase: {clase} (Etiqueta: {int(Y_train[i])})"
    show_image(X_train[i], titulo)
