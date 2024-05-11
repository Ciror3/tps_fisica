import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import re

# Ruta de la carpeta "datos"
carpeta = 'datos_platino'


# Lista para almacenar los DataFrames de cada archivo CSV
dataframes_platino = []
dataframes_bronce = []
dataframes_madera = []
dataframes_largos = []
masas = [72.18, 21.96, 4.83]
errores_masas = [0.02, 0.06, 0.03]

letters = ['A','B','C','D','E','F','G','H','largo45','largo35','largo25','largo15','largo5']

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
        if letra in ['F', 'G', 'H', 'largo5', 'largo15','largo35','largo25']:
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
                    elif letra == 'largo15':
                        i = 15
                    elif letra == 'largo5':
                        i = 5

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
        if char in ['A', 'E', 'H']:
            plt.plot(columns['columna_t_' + char], columns['columna_θ_' + char], label=f'Amplitud: {char}')

    plt.xlabel('Tiempo')
    plt.ylabel('θ')
    plt.legend()
    plt.show()

def plot_time_trajectories_L(data):
    for char, columns in data.items():
        if char in ['largo45', 'largo15', 'largo25', 'largo35']:
            if char == 'largo45':
                i = 45
            elif char == 'largo5':
                i = 5
            elif char == 'largo15':
                i = 15
            elif char == 'largo25':
                i = 25
            elif char == 'largo35':
                i = 35
            if char == 'largo45':
                plt.plot(columns['columna_t_' + char],  -1* (columns['columna_θ_' + char].values + 90), label=f'longitud: {i} cm')
            else:                             
                plt.plot(columns['columna_t_' + char], columns['columna_θ_' + char], label=f'longitud: {i} cm')
    
    plt.xlabel('Tiempo')
    plt.ylabel('θ')
    plt.legend()
    plt.show()

def plot_Taylor_vs_H(data):
    for char, columns in data.items():
        if char == 'H':
            max_theta = max(columns['columna_θ_' + char])
            tiempo = columns['columna_t_' + char]
            periodo = 1.4  # Definir el periodo deseado
            omega = 2 * np.pi / periodo  # Calcular la frecuencia angular
            amplitud_seno = np.sin(omega * tiempo) * max_theta  # Calcular el seno con la amplitud de θ y el periodo deseado
            plt.plot(tiempo, amplitud_seno, linestyle='--', label=f'seno de theta de: {char}')
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
    pers = []
    for char, columns in data.items():
        if char in ['D']:  # Solo para amplitudes específicas
            theta_con_ruido = columns['columna_θ_' + char].values
            t = columns['columna_t_' + char].values
            omega = 2*np.pi*np.sqrt(45/9.81)
            # Encontrar picos positivos
            picos, _ = find_peaks(theta_con_ruido, distance=omega*2)
            # Extraer alturas de los picos
            alturas_picos = theta_con_ruido[picos]
            # Calcular el período de la oscilación
            periodos = np.diff(t[picos])
            mean_period = np.mean(periodos)
            std_period = np.std(periodos)
            pers.append((mean_period,std_period))
            print(f"Amplitud: {char}, Período medio: {mean_period:.2f} +/- {std_period:.2f}")
            # Gráfica de Picos
            plt.plot(t, theta_con_ruido)
            plt.plot(t[picos], alturas_picos, "x")
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Ángulo (rad)')
            plt.title(f'Picos Identificados en la Oscilación (Amplitud: {char})')
            plt.show()
    return pers

def calculate_and_plot_periods_L(data):
    returnable = []
    for char, columns in data.items():
        if char in ['largo45','largo5', 'largo15', 'largo25', 'largo35']:  # Solo para amplitudes específicas
            theta_con_ruido = columns['columna_θ_' + char].values
            t = columns['columna_t_' + char].values
            if char == 'largo5':
                i= 5
            elif char == 'largo15':
                i = 15
            elif char == 'largo25':
                i = 25
            elif char == 'largo35':
                i = 35
            elif char == 'largo45':
                theta_con_ruido = -1* (columns['columna_θ_' + char].values + 90)
                i = 45
            omega = 2*np.pi*np.sqrt(i/9.81)#Periodo
            # Encontrar picos positivos
            picos, _ = find_peaks(theta_con_ruido, distance=omega*2, height=0.15)
            # Extraer alturas de los picos
            alturas_picos = theta_con_ruido[picos]
            # Calcular el período de la oscilación
            periodos = np.diff(t[picos])
            mean_period = np.mean(periodos)
            std_period = np.std(periodos)
            returnable.append((mean_period,std_period, i))
            print(f"largo: {char}, Período medio: {mean_period:.2f} +/- {std_period:.2f}")
            # Gráfica de Picos
            plt.plot(t, theta_con_ruido)
            plt.plot(t[picos], alturas_picos, "x")
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Ángulo (rad)')
            plt.title(f'Picos Identificados en la Oscilación (Amplitud: {char})')
            plt.show()
    return returnable

def plot_omega_vs_long(per):
    # Desempaquetamos los valores de T_per, err_per y longitudes de la lista per
    T_per, err_per, longitudes = zip(*per)

    # Calculamos omega a partir de T_per
    omega = [2 * np.pi / T for T in T_per]

    # Trama de puntos de datos con barras de error
    plt.errorbar(longitudes, omega, xerr= 0.01 ,yerr=err_per, fmt='o', capsize=5)

    # Etiquetas de los ejes y título
    plt.xlabel('Longitud (cm)')
    plt.ylabel('Frecuencia Angular (omega) (Rad/s)')
    
    # Mostramos la cuadrícula
    plt.grid(True)

    # Mostramos la gráfica
    plt.show()

def plot_omega_vs_mass(pers):
    # Extraemos los períodos y errores de la lista de tuplas

    periodos = []
    for per in pers:
        t = per[0][0]
        periodos.append(t)
    errores_periodos = []
    for per in pers:
        err = per[0][1]
        errores_periodos.append(err)

    # Calculamos las frecuencias angulares (omega) a partir de los períodos
    omegas = [2 * np.pi / T for T in periodos]

    # Creamos la figura y los ejes
    plt.figure(figsize=(8, 6))

    # Trazamos los puntos de datos con barras de error
    plt.errorbar(masas, omegas, xerr=errores_masas, yerr=errores_periodos, fmt='o', capsize=5)

    # Etiquetas de los ejes y título
    plt.xlabel('Masa (g)')
    plt.ylabel('Frecuencia Angular (omega) (Rad/s)')

    # Mostramos la cuadrícula
    plt.grid(True)

    # Mostramos la gráfica
    plt.show()

def linear_func(l, m, c):
    return m * l + c

def estimate_gravity_from_periods(periods_longitudes):
    # Desempaquetamos los datos de períodos y longitudes
    periods, errores, longitudes = zip(*periods_longitudes)

    # Calculamos el cuadrado de los períodos
    T_squared = np.square(periods)

    # Realizamos el ajuste lineal en escala log-log
    log_longitudes = np.log(longitudes)
    log_T_squared = np.log(T_squared)
    params, _ = curve_fit(linear_func, log_longitudes, log_T_squared)
    m, c = params

    # Calculamos la pendiente (m), que está relacionada con g
    g_estimado = 4 * np.pi**2 / np.exp(m)

    # Graficamos los datos y la línea ajustada
    plt.figure(figsize=(10, 6))
    plt.errorbar(longitudes, T_squared, yerr=np.square(errores), fmt='o', capsize=5, label='Datos')
    plt.plot(longitudes, np.exp(linear_func(np.log(longitudes), m, c)), label=f'Ajuste Lineal: $4π²/g$ = {np.exp(m):.2f}', color='red')
    plt.xlabel('Longitud (cm)')
    plt.ylabel('$T^2$ (s^2)')
    plt.title(f'Estimación de g: {g_estimado:.2f} m/s^2')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Devolvemos el valor estimado de g
    return g_estimado
# Ejemplo de uso:
#calculate_and_plot_periods(columnas_por_letra_platino)
#calculate_and_plot_periods(columnas_por_letra_bronce)
per = calculate_and_plot_periods_L(columnas_por_letra_largo)
plot_omega_vs_long(per)
# plot_time_trajectories_L(columnas_por_letra_largo)
perp = calculate_and_plot_periods(columnas_por_letra_platino)
perm = calculate_and_plot_periods(columnas_por_letra_madera)
perb = calculate_and_plot_periods(columnas_por_letra_bronce)
periods = [perp,perm,perb]
plot_omega_vs_mass(periods)
g = estimate_gravity_from_periods(per)
#plot_Taylor_vs_H(columnas_por_letra_bronce)