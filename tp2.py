import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

# Ruta de la carpeta "datos"
carpeta = 'datos_platino'


# Lista para almacenar los DataFrames de cada archivo CSV
dataframes_platino = []
dataframes_bronce = []
dataframes_madera = []
dataframes_largos = []

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
                dataframes_platino.append((df, char))
                break

def open_csv_comma(file, data):
    for archivo in os.listdir(file):
    # Verifica si el archivo es un archivo CSV
        if archivo.endswith('.csv'):
            # Construye la ruta completa del archivo
            ruta_completa = os.path.join(file, archivo)
            # Lee el archivo CSV y agrega el DataFrame a la lista
            df = pd.read_csv(ruta_completa, skiprows= 1, sep=',')
            for char in letters:
                if re.search(char, archivo):
                    data.append((df, char))
                    break

open_csv_comma('datos_bronce', dataframes_bronce)
open_csv_comma('datos_madera', dataframes_madera)
open_csv_comma('datos_cambio_largo', dataframes_largos)

columnas_por_letra_platino = {}
columnas_por_letra_madera = {}

# Define la función para transformar los ángulos
def angle_transformer(x, y, H):
    angle_rad = np.arctan(x / (H - y))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Itera sobre cada DataFrame y su correspondiente nombre de archivo en la lista de dataframes
def generate_dicc(dicc, dataframe):
    for df, char in dataframe:
        if char not in dicc:
            dicc[char] = {}
        for col in ['x', 'y', 't', 'θ']:
            dicc[char][f'columna_{col}_{char}'] = df[col]
    
    # Itera sobre cada letra en el diccionario columnas_por_letra
    for letra, columnas in dicc.items():
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

    for letra, columnas in dicc.items():
        # Verifica si la letra es H, F o G
        if letra in ['H', 'F', 'G']:
            print(f"Letra {letra}:")
            for nombre_columna, columna in columnas.items():
                # Verifica si la columna es theta
                if nombre_columna.startswith('columna_θ_'):
                    print(f"{nombre_columna}:")
                    print(columna)

generate_dicc(columnas_por_letra_platino, dataframes_platino)
generate_dicc(columnas_por_letra_madera, dataframes_madera)
print(columnas_por_letra_madera)
print(columnas_por_letra_platino)

