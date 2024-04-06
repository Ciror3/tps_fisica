import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

peso_error = 0.01
error_medicion_diametro = 0.05
error_regla = 0.1

gramajes = np.array([60, 83, 103, 123])
errores = np.array([4.160723615, 0,6631219505, 8.438760905, 2.857876348])
errores_totales_gramaje_peso = np.sqrt(errores**2 + peso_error**2)

# Calcular errores totales para la medición del diámetro y la regla
errores_totales_medicion = np.sqrt(error_medicion_diametro**2 + error_regla**2)
print(errores_totales_gramaje_peso)

# Sumar todos los errores totales
error_total = np.sqrt(np.sum(errores_totales_gramaje_peso**2) + np.sum(errores_totales_medicion**2))

# Datos de peso y diámetro para cada material
afiche_peso = np.array([ 0.17, 0.37, 0.62, 1.28, 2.083333333, 4.4])
cartulina_peso = np.array([ 0.32, 0.8, 1.233333333, 2.733333333, 4.373333333, 8.78])
liso_peso = np.array([ 0.2066666667, 0.5366666667, 0.82, 1.85, 2.97, 6.03])
cuadriculado_peso = np.array([ 0.2233333333, 0.66, 1.12, 2.326666667, 3.866666667])

afiche_diametro = np.array([ 1.03, 1.66, 1.78, 3.00, 3.67, 4.61])
cartulina_diametro = np.array([ 1.00, 1.88, 2.24, 3.07, 4.63, 5.16])
cuadriculado_diametro = np.array([ 1.23, 2.14, 2.11, 3.44, 4.37])
liso_diametro = np.array([ 1.24, 1.84, 1.94, 3.18, 3.71, 5.29])

afiche_gramaje = np.array([24.3902439, 56.33802817, 75.06255213, 158.220293])
# afiche_gramaje = np.array([24.3902439, 38.61236802, 56.33802817, 75.06255213, 98.36512262, 158.220293])
cartulina_gramaje = np.array([24.93765586,  44.59308807, 73.34963325, 141.2106538])
# cartulina_gramaje = np.array([24.93765586, 34.13333333, 44.59308807, 73.34963325, 78.05405405, 141.2106538])
cuadriculado_gramaje = np.array([20.32520325, 47.50593824, 82.70332188, 137.8071834])
# cuadriculado_gramaje = np.array([20.32520325, 29.94152047, 47.50593824, 65.35947712, 82.70332188, 137.8071834])
liso_gramaje = np.array([20.24291498, 51.67958656, 70.81038552, 97.43589744])
# liso_gramaje = np.array([20.24291498, 34.7826087, 51.67958656, 70.81038552, 97.43589744])


afiche_errores_especificos = np.array([0.01, 0.01, 0.02645751311, 0.01, 0.02081665999, 0.02645751311])
afiche_errores_totales_peso = np.sqrt(afiche_errores_especificos**2 + peso_error**2)
cartulina_errores_especificos = np.array([0.01, 0.02, 0.005773502692, 0.005773502692, 0.01527525232, 0.02516611478])
cartulina_errores_totales_peso = np.sqrt(cartulina_errores_especificos**2 + peso_error**2)
liso_errores_especificos = np.array([0.0305505046, 0.005773502692, 0.01732050808, 0, 0.01, 0.01732050808])
liso_errores_totales_peso = np.sqrt(liso_errores_especificos**2 + peso_error**2)
cuadriculado_errores_especificos = np.array([0.04725815626, 0.02645751311, 0.01, 0.005773502692, 0.01527525232])
cuadriculado_errores_totales_peso = np.sqrt(cuadriculado_errores_especificos**2 + peso_error**2)

afiche_diametro_errores_especificos = np.array([0.21, 0.07, 0.07, 0.23, 0.30, 0.22])
cartulina_diametro_errores_especificos = np.array([0.15, 0.15, 0.14, 0.13, 0.50, 0.13])
cuadriculado_diametro_errores_especificos = np.array([0.17, 0.14, 0.18, 0.24, 0.16])
liso_diametro_errores_especificos = np.array([0.05, 0.06, 0.11, 0.29, 0.07, 0.41])

afiche_diametro_errores_totales = np.sqrt(afiche_diametro_errores_especificos**2 + error_medicion_diametro**2)
cartulina_diametro_errores_totales = np.sqrt(cartulina_diametro_errores_especificos**2 + error_medicion_diametro**2)
cuadriculado_diametro_errores_totales = np.sqrt(cuadriculado_diametro_errores_especificos**2 + error_medicion_diametro**2)
liso_diametro_errores_totales = np.sqrt(liso_diametro_errores_especificos**2 + error_medicion_diametro**2)
# Graficar
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.errorbar(afiche_peso, afiche_diametro, xerr=afiche_errores_totales_peso, yerr=afiche_diametro_errores_totales, fmt='o', label="Afiche", color='blue')
plt.errorbar(cartulina_peso, cartulina_diametro, xerr=cartulina_errores_totales_peso, yerr=cartulina_diametro_errores_totales, fmt='o', label="Cartulina", color='green')
plt.errorbar(cuadriculado_peso, cuadriculado_diametro, xerr=cuadriculado_errores_totales_peso, yerr=cuadriculado_diametro_errores_totales, fmt='o', label="Cuadriculado", color='red')
plt.errorbar(liso_peso, liso_diametro, xerr=liso_errores_totales_peso, yerr=liso_diametro_errores_totales, fmt='o', label="Liso", color='purple')

plt.xlabel("Masa (gr)")
plt.ylabel("Diámetro (cm)")
plt.legend()

plt.subplot(1, 2, 2)
plt.errorbar(afiche_peso, afiche_diametro, xerr=afiche_errores_totales_peso, yerr=afiche_diametro_errores_totales, fmt='o', label="Afiche", color='blue')
plt.errorbar(cartulina_peso, cartulina_diametro, xerr=cartulina_errores_totales_peso, yerr=cartulina_diametro_errores_totales, fmt='o', label="Cartulina", color='green')
plt.errorbar(cuadriculado_peso, cuadriculado_diametro, xerr=cuadriculado_errores_totales_peso, yerr=cuadriculado_diametro_errores_totales, fmt='o', label="Cuadriculado", color='red')
plt.errorbar(liso_peso, liso_diametro, xerr=liso_errores_totales_peso, yerr=liso_diametro_errores_totales, fmt='o', label="Liso", color='purple')

plt.xlabel("Masa (gr)")
plt.xlabel("Masa (gr)")
plt.ylabel("Diámetro (cm)")
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.grid(True)
plt.show()


def linear_model(x, m, c):
    return m * x + c

# Hacemos el logaritmo de los datos
log_peso_afiche = np.log10(afiche_peso)
log_peso_cartulina = np.log10(cartulina_peso)
log_peso_cuadriculado = np.log10(cuadriculado_peso)
log_peso_liso = np.log10(liso_peso)

log_diametro_afiche = np.log10(afiche_diametro)
log_diametro_cartulina = np.log10(cartulina_diametro)
log_diametro_cuadriculado = np.log10(cuadriculado_diametro)
log_diametro_liso = np.log10(liso_diametro)

# Ajuste lineal para el logaritmo de los datos
popt_linear_afiche, pcov_linear_afiche = curve_fit(linear_model, log_peso_afiche, log_diametro_afiche)#Pcov nos devuelve la matriz de covarianza
popt_linear_cartulina, pcov_linear_cartulina = curve_fit(linear_model, log_peso_cartulina, log_diametro_cartulina)
popt_linear_cuadriculado, pcov_linear_cuadriculado = curve_fit(linear_model, log_peso_cuadriculado, log_diametro_cuadriculado)
popt_linear_liso, pcov_linear_liso = curve_fit(linear_model, log_peso_liso, log_diametro_liso)

m_afiche, c_afiche = popt_linear_afiche
m_cartulina, c_cartulina = popt_linear_cartulina
m_cuadriculado, c_cuadriculado = popt_linear_cuadriculado
m_liso, c_liso = popt_linear_liso

pendientes = np.array([m_afiche, m_cartulina, m_cuadriculado, m_liso])
errors_pendientes_afiche, errors_c_afiche = np.sqrt(np.diag(pcov_linear_afiche))
errors_pendientes_cartulina, errors_c_cartulina = np.sqrt(np.diag(pcov_linear_cartulina))
errors_pendientes_cuadriculado, errors_c_cuadriculado = np.sqrt(np.diag(pcov_linear_cuadriculado))
errors_pendientes_liso, errors_c_liso = np.sqrt(np.diag(pcov_linear_liso))

errors_pendientes = np.array([errors_pendientes_afiche, errors_pendientes_liso, errors_pendientes_cuadriculado, errors_pendientes_cartulina])
errors_ords = np.array([errors_c_afiche, errors_c_cartulina, errors_c_cuadriculado, errors_c_liso])
plt.figure(figsize=(12, 6))

plt.errorbar(log_peso_afiche, log_diametro_afiche,yerr=errors_pendientes_afiche, fmt='o', label="Afiche", color='blue')
plt.errorbar(log_peso_cartulina, log_diametro_cartulina,yerr=errors_pendientes_cartulina, fmt='o', label="Cartulina", color='green')
plt.errorbar(log_peso_cuadriculado, log_diametro_cuadriculado,yerr=errors_pendientes_cuadriculado, fmt='o', label="Cuadriculado", color='red')
plt.errorbar(log_peso_liso, log_diametro_liso,yerr=errors_pendientes_liso, fmt='o', label="Liso", color='purple')
plt.plot(log_peso_afiche, linear_model(log_peso_afiche, m_afiche, c_afiche), label=f"Ajuste lineal Afiche", color='blue')
plt.plot(log_peso_cartulina, linear_model(log_peso_cartulina, m_cartulina, c_cartulina), label=f"Ajuste lineal Cartulina:", color='green')
plt.plot(log_peso_cuadriculado, linear_model(log_peso_cuadriculado, m_cuadriculado, c_cuadriculado), label=f"Ajuste lineal Cuadriculado", color='red')
plt.plot(log_peso_liso, linear_model(log_peso_liso, m_liso, c_liso), label=f"Ajuste lineal Liso", color='purple')
plt.xlabel("log(Masa(gr))")
plt.ylabel("log(Diámetro(cm))")
plt.legend()
plt.show()
# print(y=({m_afiche:.2f}±{errors_pendientes_afiche:.2f})x + ({c_afiche:.2f}±{c_err_afiche:.2f})")


def power_law(x, a, b):
    return a * x**b

# Ajuste de la ley de potencia a los datos
popt_afiche, pcov_afiche = curve_fit(power_law, afiche_peso, afiche_diametro)
popt_cartulina, pcov_cartulina = curve_fit(power_law, cartulina_peso, cartulina_diametro)
popt_liso, pcov_liso = curve_fit(power_law, liso_peso, liso_diametro)
popt_cuadriculado, pcov_cuadriculado = curve_fit(power_law, cuadriculado_peso, cuadriculado_diametro)

# Coeficientes de la ley de potencia
a_afiche, b_afiche = popt_afiche
a_cartulina, b_cartulina = popt_cartulina
a_liso, b_liso = popt_liso
a_cuadriculado, b_cuadriculado = popt_cuadriculado

error_a_afiche, error_b_afiche = np.sqrt(np.diag(pcov_afiche))
error_a_cartulina, error_b_cartulina = np.sqrt(np.diag(pcov_cartulina))
error_a_liso, error_b_liso = np.sqrt(np.diag(pcov_liso))
error_a_cuadriculado, error_b_cuadriculado = np.sqrt(np.diag(pcov_cuadriculado))

# Generar puntos para la curva ajustada
x_afiche = np.linspace(min(afiche_peso), max(afiche_peso), 100)
x_cartulina = np.linspace(min(cartulina_peso), max(cartulina_peso), 100)
x_liso = np.linspace(min(liso_peso), max(liso_peso), 100)
x_cuadriculado = np.linspace(min(cuadriculado_peso), max(cuadriculado_peso), 100)

y_afiche = power_law(x_afiche, a_afiche, b_afiche)
y_cartulina = power_law(x_cartulina, a_cartulina, b_cartulina)
y_liso = power_law(x_liso, a_liso, b_liso)
y_cuadriculado = power_law(x_cuadriculado, a_cuadriculado, b_cuadriculado)

print("A:\n","Afiche ", a_afiche, "± ", error_a_afiche, "\nCartulina ", a_cartulina, "± ", error_a_cartulina, "\nLiso ", a_liso, "± ", error_a_liso,"\nCuadriculado ", a_cuadriculado, "± ", error_a_cuadriculado)
print("b:\n","Afiche ", b_afiche, "± ", error_b_afiche, "\nCartulina ", b_cartulina, "± ", error_b_cartulina, "\nLiso ", b_liso, "± ", error_b_liso,"\nCuadriculado ", b_cuadriculado, "± ", error_b_cuadriculado)
plt.figure(figsize=(10, 6))

plt.errorbar(afiche_peso, afiche_diametro, xerr=afiche_errores_totales_peso, yerr=afiche_diametro_errores_totales, fmt='o', label="Afiche", color='blue')
plt.errorbar(cartulina_peso, cartulina_diametro, xerr=cartulina_errores_totales_peso, yerr=cartulina_diametro_errores_totales, fmt='o', label="Cartulina", color='green')
plt.errorbar(cuadriculado_peso, cuadriculado_diametro, xerr=cuadriculado_errores_totales_peso, yerr=cuadriculado_diametro_errores_totales, fmt='o', label="Cuadriculado", color='red')
plt.errorbar(liso_peso, liso_diametro, xerr=liso_errores_totales_peso, yerr=liso_diametro_errores_totales, fmt='o', label="Liso", color='purple')

plt.plot(x_afiche, power_law(x_afiche, a_afiche, b_afiche), label=f"Ley de Potencia Afiche", color='blue')
plt.plot(x_cartulina, power_law(x_cartulina, a_cartulina, b_cartulina), label=f"Ley de Potencia Cartulina", color= 'green')
plt.plot(x_cuadriculado, power_law(x_cuadriculado, a_cuadriculado, b_cuadriculado), label=f"Ley de Potencia Cuadriculado", color= 'red')
plt.plot(x_liso, power_law(x_liso, a_liso, b_liso), label=f"Ley de Potencia Liso", color= 'purple')
plt.xlabel("Masa (gr)")
plt.ylabel("Diámetro (cm)")
plt.legend()

plt.tight_layout()
plt.show()
a_values = np.array([a_afiche, a_cartulina, a_liso, a_cuadriculado])
c_values = np.array([c_afiche, c_cartulina, c_liso, c_cuadriculado])
deltas_c = np.array([errors_c_afiche, errors_c_cartulina, errors_c_liso, errors_c_cuadriculado])

# Calcular A
A_values = 10**c_values

# Calcular la incerteza de A
delta_A_values = np.log(10) * A_values * deltas_c

# Mostrar resultados
for material, A, delta_A, a in zip(["Afiche", "Cartulina", "Liso", "Cuadriculado"], A_values, delta_A_values, a_values):
    print(f"Material: {material}")
    print(f"A calculado: {A} ± {delta_A}")
    print(f"A real: {a}")
    print()

# Calculamos los gramajes para cada pendiente usando un rango de valores de pendiente
# gramajes = pendientes * np.arange(len(pendientes)) + 100

# Graficamos
plt.figure(figsize=(8, 6))
plt.errorbar(gramajes, pendientes, xerr= error_regla, yerr= errors_pendientes, fmt='o-', color='blue')
# Etiquetas y título
plt.xlabel('Gramajes (mg/cm^2)')
plt.ylabel('Pendientes (cm/g)')

# Mostrar la rejilla
plt.grid(True)

# Mostrar el gráfico
plt.show()

# xerr = error_regla, yerr = errors_pendientes
#Gramajes

ms = np.array([m_afiche, m_cartulina, m_cuadriculado, m_liso])
cs = np.array([c_afiche, c_cartulina, c_cuadriculado, c_liso])

# Print slopes and intercepts with their errors
for i, tipo_bollo in enumerate(["Afiche", "Cartulina", "Cuadriculado", "Liso"]):
    print(f"Pendiente (m) para {tipo_bollo}: {ms[i]:.2f} ± {errors_pendientes[i]:.2f}")
    print(f"Ordenada al origen (c) para {tipo_bollo}: {cs[i]:.2f} ± {errors_ords[i]:.2f}")
# y=({a:.2f}±$\sigma_a$)x^({b:.2f}±{m_err:.2f})
