import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

# Ruta de la carpeta "datos"
carpeta = 'datos'

# Lista para almacenar los DataFrames de cada archivo CSV
dataframes = []
letters = ['A','B','C','D','E','F','G','H']

for archivo in os.listdir(carpeta):
    # Verifica si el archivo es un archivo CSV
    if archivo.endswith('.csv'):
        # Construye la ruta completa del archivo
        ruta_completa = os.path.join(carpeta, archivo)
        # Lee el archivo CSV y agrega el DataFrame a la lista
        df = pd.read_csv(ruta_completa, skiprows= 1, sep='\t')
        for char in letters:
            if re.search(char, archivo):
                dataframes.append((df, char))
                break
columnas_por_letra = {}

# Itera sobre cada DataFrame y su correspondiente nombre de archivo en la lista de dataframes
for df, char in dataframes:
    if char not in columnas_por_letra:
        columnas_por_letra[char] = {}
    for col in ['x', 'y', 't', 'θ']:
        columnas_por_letra[char][f'columna_{col}_{char}'] = df[col]

# Define la función para transformar los ángulos
def angle_transformer(x, y, H):
    return np.arcsin(x / (H - y))

# Itera sobre cada letra en el diccionario columnas_por_letra
for letra, columnas in columnas_por_letra.items():
    # Verifica si la letra es F, G o H
    if letra in ['F', 'G', 'H']:
        # Itera sobre las columnas para esta letra
        for nombre_columna, columna in columnas.items():
            # Verifica si la columna es theta
            if nombre_columna.endswith('_θ_' + letra):
                # Obtiene el nombre de la columna x y correspondiente a la misma letra
                nombre_columna_x = nombre_columna.replace('_θ_', '_x_')
                nombre_columna_y = nombre_columna.replace('_θ_', '_y_')
                # Calcula los nuevos valores de theta utilizando angle_transformer
                columnas[nombre_columna] = angle_transformer(columnas[nombre_columna_x], columnas[nombre_columna_y], 45)

for letra, columnas in columnas_por_letra.items():
    print(f"Letra {letra}:")
    for nombre_columna, columna in columnas.items():
        # Verifica si la columna es theta
        if nombre_columna.startswith('columna_θ_'):
            print(f"{nombre_columna}:")
            print(columna)
