import numpy as np
import matplotlib.pyplot as plt

# Crear un array de ejemplo (matriz 6x7 con valores 0 y 1)
array_binario = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, -1, 1, 0, 0, 0],
    [0, 1, -1, 1, 1, 0, 0],
    [0, 1, 1, -1, -1, 1, 0],
    [0, -1, -1, 1, 1, -1, 1]
])

encoded_state = np.stack(
    (array_binario == -1, array_binario == 0, array_binario == 1)
).astype(np.float32)


for i in range(3):
    # Crear la figura y establecer el tamaño para que coincida con las dimensiones de la matriz
    plt.figure(figsize=(7, 6))

    # Mostrar la matriz con la escala de grises
    plt.imshow(encoded_state[i], cmap="gray", interpolation="nearest")

    # Colocar el grid en el borde de los píxeles, pero solo en las marcas menores (bordes)
    plt.grid(which='minor', color='gray', linestyle='-', linewidth=2)

    # Establecer las posiciones de las líneas de la cuadrícula para que coincidan con los bordes de los píxeles
    plt.xticks(np.arange(0, array_binario.shape[1], 1))  # Alineación para las columnas (ticks en el centro de los píxeles)
    plt.yticks(np.arange(0, array_binario.shape[0], 1))  # Alineación para las filas (ticks en el centro de los píxeles)

    # Ajustar los ticks menores (grid) para que se alineen con los bordes de las celdas
    plt.xticks(np.arange(-0.5, array_binario.shape[1], 1), minor=True)  # Alineación de grid entre las celdas en el eje X
    plt.yticks(np.arange(-0.5, array_binario.shape[0], 1), minor=True)  # Alineación de grid entre las celdas en el eje Y

    # Activar el grid para las marcas menores (bordes de las celdas) y desactivar para los ejes principales
    plt.gca().tick_params(which='minor', size=0)  # No mostrar los ticks menores en los ejes

    # Establecer el aspecto de la imagen para que los píxeles sean cuadrados
    plt.gca().set_aspect('equal', adjustable='box')

    plt.colorbar()

    # Mostrar la imagen con la cuadrícula
    plt.show()