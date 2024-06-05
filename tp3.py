import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Cargar los datos de los CSV
normal = pd.read_csv('mediciones_Normales.csv')
papel = pd.read_csv('mediciones_Papel.csv')

# Definir las constantes
g = 9.81  # Aceleración debida a la gravedad en m/s^2

# Masas específicas para la primera columna de mediciones normales
pp = 43.92
solo = 108.78

# Listas de masas para las demás columnas
colgante = [72.18, 72.18,72.18,72.18,94.14,94.14,94.14,94.14]
masas = [280.79, 352.97, 302.75, 375.57]

m_b = [108.78,280.79, 352.97, 302.75, 375.57]
m_bp = [280.79, 352.97, 302.75, 375.57]

colg = [94.14]

# Función para calcular el coeficiente de rozamiento dinámico
def calcular_mu_d(a, M, m, g):
    return (M * g - (M + m) * a )/ (m * g)

def calc_distance(t, c=343): 
   # t is in microseconds, convert to seconds for the calculation
    t_seconds = t / 1e6
    distance_meters = t_seconds * c / 2
    distance_centimeters = distance_meters * 100
    return distance_centimeters

def cuad(t,a,b,c):
    return a/2*(t**2)+b*t+c

# Función para calcular mu_d y devolver los resultados en un diccionario
def calcular_mu_d_para_dataframe(df, colgante, masas, pp, solo, g):
    resultados = {}
    a_values_bronce = []
    a_values_bronceplat = []
    columns_bronce = []
    columns_bronceplat = []
    for i, column in enumerate(df.columns[1:]):
        # Encontrar el índice del primer cero en la columna, si existe
        zero_index = (df[column] == 0).to_numpy().argmax() if (df[column] == 0).any() else len(df)
        
        # Tomar los valores hasta el primer cero
        t_values = df['t'][:zero_index] / 1000  # Convertir tiempo a segundos
        column_values = df[column][:zero_index]

        # Interpolación lineal manteniendo los puntos originales
        f = interp1d(t_values, column_values, kind='linear', fill_value="extrapolate")
        
        # Crear nuevos puntos para interpolación
        t_new = np.linspace(t_values.min(), t_values.max(),2* len(t_values))  # Más puntos para interpolación
        x_new = f(t_new)

        # Combinar los puntos originales y los nuevos
        t_combined = np.concatenate((t_values, t_new))
        x_combined = np.concatenate((column_values, x_new))

        # Ordenar los valores combinados por tiempo
        sorted_indices = np.argsort(t_combined)
        t_combined = t_combined[sorted_indices]
        x_combined = x_combined[sorted_indices]

        # Convertir x_combined a distancia usando calc_distance
        x_combined = np.array([calc_distance(x) for x in x_combined])

        # Determinar M y m
        if i == 0:
            M, m = pp, solo
        else:
            M, m = colgante[(i-1) % len(colgante)], masas[(i-1) % len(masas)]

        # Ajuste de curva (polinomio de grado 2 en este caso)
        popt, pcov = curve_fit(cuad, t_combined, x_combined)
        t_fit = np.linspace(t_combined.min(), t_combined.max(), 100)
        x_fit = cuad(t_fit, *popt)
        
        # Obtener el valor de 'a'
        a_value = popt[0]
        print(f"Column: {column}, a: {a_value} cm/s²")
        if i < 5:
            a_values_bronce.append(abs(a_value))
            columns_bronce.append(column)
        else:
            a_values_bronceplat.append(abs(a_value))
            columns_bronceplat.append(column)
        
        # Calcular mu_d
        mu_d = calcular_mu_d(a_value, M, m, g)

        # Guardar el resultado promedio en el diccionario
        resultados[column] = abs(mu_d)

        # Crear el scatter plot con ajuste de curva
        # plt.figure(figsize=(10, 6))
        # plt.scatter(t_combined, x_combined, label=f'{column} (Interpolado y Original)')
        # plt.plot(t_fit, x_fit, label='Ajuste de curva (polinomio de grado 2)', color='red')
        # plt.title(f'Interpolación, Datos Originales y Ajuste de Curva: {column}')
        # plt.xlabel('Tiempo (s)')
        # plt.ylabel('Distancia (cm)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    # Crear el gráfico de los valores de 'a' como constantes
    plt.figure(figsize=(12, 6))
    plt.scatter(m_b, a_values_bronce, marker='o', linestyle='-', color='b')
    plt.title('Valores de a en función de las masas')
    plt.xlabel('Masas (kg)')
    plt.ylabel('a (cm/s²)')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.scatter(m_bp, a_values_bronceplat, marker='o', linestyle='-', color='b')
    plt.title('Valores de a en función de las masas')
    plt.xlabel('Masas (kg)')
    plt.ylabel('a (cm/s²)')
    plt.grid(True)
    plt.show()

    return resultados

def calcular_mu_d_para_papel(df, colgante, masas, g):
    resultados = {}
    a_values_bronceplat = []
    columns_bronceplat = []
    for i, column in enumerate(df.columns[1:]):
        # Encontrar el índice del primer cero en la columna, si existe
        zero_index = (df[column] == 0).to_numpy().argmax() if (df[column] == 0).any() else len(df)
        
        # Tomar los valores hasta el primer cero
        t_values = df['t'][:zero_index] / 1000  # Convertir tiempo a segundos
        column_values = df[column][:zero_index]

        # Interpolación lineal manteniendo los puntos originales
        f = interp1d(t_values, column_values, kind='linear', fill_value="extrapolate")
        
        # Crear nuevos puntos para interpolación
        t_new = np.linspace(t_values.min(), t_values.max(),2* len(t_values))  # Más puntos para interpolación
        x_new = f(t_new)

        # Combinar los puntos originales y los nuevos
        t_combined = np.concatenate((t_values, t_new))
        x_combined = np.concatenate((column_values, x_new))

        # Ordenar los valores combinados por tiempo
        sorted_indices = np.argsort(t_combined)
        t_combined = t_combined[sorted_indices]
        x_combined = x_combined[sorted_indices]

        # Convertir x_combined a distancia usando calc_distance
        x_combined = np.array([calc_distance(x) for x in x_combined])

        # Determinar M y m
        M, m = colgante[0], masas[i]

        # Ajuste de curva (polinomio de grado 2 en este caso)
        popt, pcov = curve_fit(cuad, t_combined, x_combined)
        t_fit = np.linspace(t_combined.min(), t_combined.max(), 100)
        x_fit = cuad(t_fit, *popt)
        
        # Obtener el valor de 'a'
        a_value = popt[0]
        print(f"Column: {column}, a: {a_value} cm/s²")    
        a_values_bronceplat.append(abs(a_value))
        columns_bronceplat.append(column)
        
        # Calcular mu_d
        mu_d = calcular_mu_d(a_value, M, m, g)

        # Guardar el resultado promedio en el diccionario
        resultados[column] = abs(mu_d)

        # Crear el scatter plot con ajuste de curva
        plt.figure(figsize=(10, 6))
        plt.scatter(t_combined, x_combined, label=f'{column} (Interpolado y Original)')
        plt.plot(t_fit, x_fit, label='Ajuste de curva (polinomio de grado 2)', color='red')
        plt.title(f'Interpolación, Datos Originales y Ajuste de Curva: {column}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Distancia (cm)')
        plt.legend()
        plt.grid(True)
        plt.show()
    # Crear el gráfico de los valores de 'a' como constantes
    plt.figure(figsize=(12, 6))
    plt.scatter(m_bp, a_values_bronceplat, marker='o', linestyle='-', color='b')
    plt.title('Valores de a en función de las masas')
    plt.xlabel('Masas (kg)')
    plt.ylabel('a (cm/s²)')
    plt.grid(True)
    plt.show()

    return resultados

# Calcular mu_d para mediciones normales
# resultados_normales = calcular_mu_d_para_dataframe(normal, colgante, masas, pp, solo, g)

# Calcular mu_d para mediciones con rozamiento papel
resultados_papel = calcular_mu_d_para_papel(papel, colg, masas, g)

# Ejemplo de cómo acceder a los resultados
print("Resultados para mediciones normales:")
for clave, valor in resultados_papel.items():
    print(f"{clave}: {valor:.5f}")
