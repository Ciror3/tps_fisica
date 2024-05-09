import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks
import re

# Ruta de la carpeta "datos"
carpeta = 'datos_platino'


# Lista para almacenar los DataFrames de cada archivo CSV
dataframes_platino = []
dataframes_bronce = []
dataframes_madera = []
dataframes_largos = []

letters = ['A','B','C','D','E','F','G','H','largo35','largo25','largo15','largo5']

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
columnas_por_letra_bronce = {}
columnas_por_letra_largo = {}

# Define la función para transformar los ángulos
def angle_transformer(x, y, H):
    angle_rad = np.arctan(x / (H - y))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Itera sobre cada DataFrame y su correspondiente nombre de archivo en la lista de dataframes
def generate_dicc_and_find_angles(dicc, dataframe):
    for df, char in dataframe:
        if char not in dicc:
            dicc[char] = {}
        for col in ['x', 'y', 't', 'θ']:
            if col == 'y':
                y_values = df[col].values
                minimo = min(y_values)
                if minimo < -1:
                    dicc[char][f'columna_{col}_{char}'] = df[col] - minimo
                else:
                    dicc[char][f'columna_{col}_{char}'] = df[col]
            else:
                dicc[char][f'columna_{col}_{char}'] = df[col]
    
    # Itera sobre cada letra en el diccionario columnas_por_letra

    for letra, columnas in dicc.items():
        # Verifica si la letra es F, G o H
        if letra in ['F', 'G', 'H','largo35','largo25']:
            # Itera sobre las columnas para esta letra
            for nombre_columna, columna in columnas.items():
                # Verifica si la columna es theta
                if nombre_columna.endswith('_θ_' + letra):
                    # Obtiene el nombre de la columna x y correspondiente a la misma letra
                    nombre_columna_x = nombre_columna.replace('_θ_', '_x_')
                    nombre_columna_y = nombre_columna.replace('_θ_', '_y_')
                    # Calcula los nuevos valores de theta utilizando angle_transformer
                    i = 45
                    if letra == 'largo35':
                        i = 35
                    elif letra == 'largo25':
                        i = 25

                    columnas[nombre_columna] = angle_transformer(columnas[nombre_columna_x], columnas[nombre_columna_y], i)

def generate_dicc_for_bronce(dicc, dataframe):
    for df, char in dataframe:
        if char not in dicc:
            dicc[char] = {}
        for col in ['x', 'y', 't', 'θ']:
            if col == 'y':
                y_values = df[col].values
                minimo = min(y_values)
                if minimo < -1:
                    dicc[char][f'columna_{col}_{char}'] = df[col] - minimo
                else:
                    dicc[char][f'columna_{col}_{char}'] = df[col]
            elif col == 'θ':
                dicc[char][f'columna_{col}_{char}'] = df[col] + 90
            else:
                dicc[char][f'columna_{col}_{char}'] = df[col]

generate_dicc_and_find_angles(columnas_por_letra_platino, dataframes_platino)
generate_dicc_and_find_angles(columnas_por_letra_madera, dataframes_madera)
generate_dicc_and_find_angles(columnas_por_letra_largo, dataframes_largos)
generate_dicc_for_bronce(columnas_por_letra_bronce, dataframes_bronce)

def plot_pendulum_trajectories(data):
    for char, columns in data.items():
        if char in ['A', 'D', 'F', 'H']:
            plt.plot(columns['columna_x_' + char], columns['columna_y_' + char], label=f'Amplitud: {char}')
    plt.xlabel('Posición en x')
    plt.ylabel('Posición en y')
    plt.legend()
    plt.show()


# Llama a la función para graficar el recorrido del péndulo para el platino

def plot_time_trajectories(data):
    for char, columns in data.items():
        if char in ['A', 'D', 'F', 'H']:
            plt.plot(columns['columna_t_' + char], columns['columna_θ_' + char], label=f'Amplitud: {char}')
    plt.xlabel('Tiempo')
    plt.ylabel('θ')
    plt.legend()
    plt.show()

# Por ejemplo, para graficar en función del tiempo para la posición en x (fijando amplitud inicial):

def plot_time_trajectories_sin(data):
    for char, columns in data.items():
        if char in [ 'H']:
            plt.plot(columns['columna_t_' + char], np.sin(columns['columna_θ_' + char]), label=f'Amplitud: {char}')
    plt.xlabel('Tiempo')
    plt.ylabel('θ')
    plt.legend()
    plt.show()

def plot_amplitude_vs_time(dataframes):
    for df, char in dataframes:
        amplitude = np.sqrt(df['x']**2 + df['y']**2)
        plt.plot(df['t'], amplitude, label=f'Amplitud: {char}')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud de oscilación')
    plt.title('Amplitud de oscilación en función del tiempo')
    plt.legend()
    plt.show()

def calculate_and_plot_periods(data):
    for char, columns in data.items():
        if char in ['A', 'D', 'F', 'H']:  # Solo para amplitudes específicas
            theta_con_ruido = columns['columna_θ_' + char].values
            t = columns['columna_t_' + char].values
            omega = 2*np.pi*np.sqrt(45/9.81)
            # Encontrar picos positivos
            picos, _ = find_peaks(theta_con_ruido, distance=omega*2, height=0.15)
            # Extraer alturas de los picos
            alturas_picos = theta_con_ruido[picos]
            # Calcular el período de la oscilación
            periodos = np.diff(t[picos])
            mean_period = np.mean(periodos)
            std_period = np.std(periodos)
            print(f"Amplitud: {char}, Período medio: {mean_period:.1f} +/- {std_period:.1f}")
            # Gráfica de Picos
            plt.plot(t, theta_con_ruido)
            plt.plot(t[picos], alturas_picos, "x")
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Ángulo (rad)')
            plt.title(f'Picos Identificados en la Oscilación (Amplitud: {char})')
            plt.show()

# Ejemplo de uso:
calculate_and_plot_periods(columnas_por_letra_platino)

# plot_pendulum_trajectories(columnas_por_letra_platino)
# plot_pendulum_trajectories(columnas_por_letra_madera)
# plot_pendulum_trajectories(columnas_por_letra_bronce)
# plot_time_trajectories(columnas_por_letra_platino)
# plot_time_trajectories(columnas_por_letra_madera)
# plot_time_trajectories(columnas_por_letra_bronce)
# plot_time_trajectories_sin(columnas_por_letra_platino)
# plot_time_trajectories_sin(columnas_por_letra_madera)
# plot_time_trajectories_sin(columnas_por_letra_bronce)
