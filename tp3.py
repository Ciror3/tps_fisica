import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos de los CSV
normal = pd.read_csv('mediciones_Normales.csv')
papel = pd.read_csv('mediciones_Papel.csv')

# Función para crear un scatter plot para cada columna
def plot_scatter(df, title_prefix):
    for column in df.columns[1:]:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['t'], df[column], label=column)
        plt.title(f'{title_prefix}: {column}')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Metros (mm)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Crear scatter plots para mediciones normales
plot_scatter(normal, 'Mediciones Normales')

# Crear scatter plots para mediciones con rozamiento papel
plot_scatter(papel, 'Mediciones Papel')
