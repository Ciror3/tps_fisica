import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Cargar los datos de los CSV
normal = pd.read_csv('mediciones_Normales.csv')
papel = pd.read_csv('mediciones_Papel.csv')

# Función para crear un scatter plot con puntos interpolados agregados
def plot_scatter_with_interpolation(df, title_prefix):
    for column in df.columns[1:]:
        # Encontrar el índice del primer cero en la columna, si existe
        zero_index = (df[column] == 0).to_numpy().argmax() if (df[column] == 0).any() else len(df)
        
        # Tomar los valores hasta el primer cero
        t_values = df['t'][:zero_index]
        column_values = df[column][:zero_index]
        
        # Interpolación lineal
        f = interp1d(t_values, column_values, kind='linear', fill_value="extrapolate")
        
        # Crear nuevos puntos para interpolación
        t_new = np.linspace(t_values.min(), t_values.max(), len(t_values) + 2)
        column_new = f(t_new)
        
        # Crear gráfico
        plt.figure(figsize=(10, 6))
        plt.scatter(t_new, column_new, label=column)
        plt.title(f'{title_prefix}: {column}')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Metros (mm)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Crear scatter plots para mediciones normales con puntos interpolados agregados
plot_scatter_with_interpolation(normal, 'Mediciones Normales')

# Crear scatter plots para mediciones con rozamiento papel con puntos interpolados agregados
plot_scatter_with_interpolation(papel, 'Mediciones Papel')
