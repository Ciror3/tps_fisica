import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

# Función para calcular el coeficiente de rozamiento dinámico
def calcular_mu_d(t, x, x0, M, m, g):
    return (M * g - (M + m) * (2 * (x - x0) / t**2)) / (m * g)

def calc_distance(t, c=343): 
    # t is in microseconds, convert to seconds for the calculation
    t_seconds = t / 1e6
    distance_meters = t_seconds * c / 2
    distance_centimeters = distance_meters * 100
    return distance_centimeters

# Función para calcular mu_d y devolver los resultados en un diccionario
def calcular_mu_d_para_dataframe(df, colgante, masas, pp, solo, g):
    resultados = {}
    for i, column in enumerate(df.columns[1:]):
        # Encontrar el índice del primer cero en la columna, si existe
        zero_index = (df[column] == 0).to_numpy().argmax() if (df[column] == 0).any() else len(df)
        
        # Tomar los valores hasta el primer cero
        t_values = df['t'][:zero_index]
        column_values = df[column][:zero_index]
        x0 = column_values.iloc[0] if len(column_values) > 0 else 0

        # Interpolación lineal manteniendo los puntos originales
        f = interp1d(t_values, column_values, kind='linear', fill_value="extrapolate")
        
        # Crear nuevos puntos para interpolación
        t_new = np.linspace(t_values.min(), t_values.max(), len(t_values) + 2)
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

        # Calcular mu_d para los puntos interpolados y originales combinados
        mu_d_values = [calcular_mu_d(t, x, x0, M, m, g) for t, x in zip(t_combined, x_combined)]

        # Calcular el promedio de mu_d
        mu_d_promedio = np.mean(mu_d_values)

        # Guardar el resultado promedio en el diccionario
        resultados[column] = mu_d_promedio
    
        # Crear el scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(t_combined, x_combined, label=f'{column} (Interpolado y Original)')
        plt.title(f'Interpolación y Datos Originales: {column}')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Distancia (cm)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return resultados
# Calcular mu_d para mediciones normales
resultados_normales = calcular_mu_d_para_dataframe(normal, colgante, masas, pp, solo, g)

# Calcular mu_d para mediciones con rozamiento papel
# resultados_papel = calcular_mu_d_para_dataframe(papel, colgante, masas, pp, solo, g)

# Ejemplo de cómo acceder a los resultados
# print("Resultados para mediciones normales:")
# for clave, valor in resultados_normales.items():
#     print(f"{clave}: mu_d_promedio = {valor['mu_d_promedio']:.5f}")
#     print(f"x_teorico: {valor['x_teorico']}")
#     print()
