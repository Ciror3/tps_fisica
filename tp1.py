import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Datos de peso y diámetro para cada material
afiche_peso = np.array([0.01, 0.17, 0.37, 0.62, 1.28, 2.083333333, 4.4])
cartulina_peso = np.array([0.01, 0.32, 0.8, 1.233333333, 2.733333333, 4.373333333, 8.78])
liso_peso = np.array([0.01, 0.2066666667, 0.5366666667, 0.82, 1.85, 2.97, 6.03])
cuadriculado_peso = np.array([0.01, 0.2233333333, 0.66, 1.12, 2.326666667, 3.866666667])

afiche_diametro = np.array([0.47, 1.03, 1.66, 1.78, 3.00, 3.67, 4.61])
cartulina_diametro = np.array([0.53, 1.00, 1.88, 2.24, 3.07, 4.63, 5.16])
cuadriculado_diametro = np.array([0.58, 1.23, 2.14, 2.11, 3.44, 4.37])
liso_diametro = np.array([0.56, 1.24, 1.84, 1.94, 3.18, 3.71, 5.29])

# Errores de medición
peso_error = 0.01
diametro_error = 0.05

# Graficar
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.errorbar(afiche_peso, afiche_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Afiche", color='blue')
plt.errorbar(cartulina_peso, cartulina_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Cartulina", color='green')
plt.errorbar(cuadriculado_peso, cuadriculado_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Cuadriculado", color='red')
plt.errorbar(liso_peso, liso_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Liso", color='purple')

plt.xlabel("Masa (gr)")
plt.ylabel("Diámetro (cm)")
plt.legend()

plt.subplot(1, 2, 2)
plt.errorbar(afiche_peso, afiche_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Afiche", color='blue')
plt.errorbar(cartulina_peso, cartulina_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Cartulina", color='green')
plt.errorbar(cuadriculado_peso, cuadriculado_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Cuadriculado", color='red')
plt.errorbar(liso_peso, liso_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Liso", color='purple')

plt.xlabel("Masa (gr)")
plt.ylabel("Diámetro (cm)")
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.grid(True)
plt.show()


# Modelo linear con la data en log-log
peso_error = 0.01
diametro_error = 0.05

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
popt_linear_afiche, pcov_linear_afiche = curve_fit(linear_model, log_peso_afiche, log_diametro_afiche)
popt_linear_cartulina, pcov_linear_cartulina = curve_fit(linear_model, log_peso_cartulina, log_diametro_cartulina)
popt_linear_cuadriculado, pcov_linear_cuadriculado = curve_fit(linear_model, log_peso_cuadriculado, log_diametro_cuadriculado)
popt_linear_liso, pcov_linear_liso = curve_fit(linear_model, log_peso_liso, log_diametro_liso)

m_afiche, c_afiche = popt_linear_afiche
m_cartulina, c_cartulina = popt_linear_cartulina
m_cuadriculado, c_cuadriculado = popt_linear_cuadriculado
m_liso, c_liso = popt_linear_liso

m_err_afiche, c_err_afiche = np.sqrt(np.diag(pcov_linear_afiche))
m_err_cartulina, c_err_cartulina = np.sqrt(np.diag(pcov_linear_cartulina))
m_err_cuadriculado, c_err_cuadriculado = np.sqrt(np.diag(pcov_linear_cuadriculado))
m_err_liso, c_err_liso = np.sqrt(np.diag(pcov_linear_liso))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.errorbar(log_peso_afiche, log_diametro_afiche, fmt='o', label="Afiche", color='blue')
plt.errorbar(log_peso_cartulina, log_diametro_cartulina, fmt='o', label="Cartulina", color='green')
plt.errorbar(log_peso_cuadriculado, log_diametro_cuadriculado, fmt='o', label="Cuadriculado", color='red')
plt.errorbar(log_peso_liso, log_diametro_liso, fmt='o', label="Liso", color='purple')
plt.plot(log_peso_afiche, linear_model(log_peso_afiche, m_afiche, c_afiche), label=f"Ajuste linear Afiche", color='blue')
plt.plot(log_peso_cartulina, linear_model(log_peso_cartulina, m_cartulina, c_cartulina), label=f"Ajuste linear Cartulina:", color='green')
plt.plot(log_peso_cuadriculado, linear_model(log_peso_cuadriculado, m_cuadriculado, c_cuadriculado), label=f"Ajuste linear Cuadriculado", color='red')
plt.plot(log_peso_liso, linear_model(log_peso_liso, m_liso, c_liso), label=f"Ajuste linear Liso", color='purple')
plt.xlabel("log(Masa)")
plt.ylabel("log(Diámetro)")
plt.legend()
plt.show()
# print(y=({m_afiche:.2f}±{m_err_afiche:.2f})x + ({c_afiche:.2f}±{c_err_afiche:.2f})")
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

# Generar puntos para la curva ajustada
x_afiche = np.linspace(min(afiche_peso), max(afiche_peso), 100)
x_cartulina = np.linspace(min(cartulina_peso), max(cartulina_peso), 100)
x_liso = np.linspace(min(liso_peso), max(liso_peso), 100)
x_cuadriculado = np.linspace(min(cuadriculado_peso), max(cuadriculado_peso), 100)

y_afiche = power_law(x_afiche, a_afiche, b_afiche)
y_cartulina = power_law(x_cartulina, a_cartulina, b_cartulina)
y_liso = power_law(x_liso, a_liso, b_liso)
y_cuadriculado = power_law(x_cuadriculado, a_cuadriculado, b_cuadriculado)



plt.subplot(1, 2, 2)
plt.errorbar(afiche_peso, afiche_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Data", color='blue')
plt.errorbar(cartulina_peso, cartulina_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Data", color='green')
plt.errorbar(cuadriculado_peso, cuadriculado_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Data", color='red')
plt.errorbar(liso_peso, liso_diametro, xerr=peso_error, yerr=diametro_error, fmt='o', label="Data", color='purple')


plt.plot(x_afiche, power_law(x_afiche, a_afiche, b_afiche), label=f"Power law Afiche", color='blue')
plt.plot(x_cartulina, power_law(x_cartulina, a_cartulina, b_cartulina), label=f"Power law Cartulina", color= 'green')
plt.plot(x_cuadriculado, power_law(x_cuadriculado, a_cuadriculado, b_cuadriculado), label=f"Power law Cuadriculado", color= 'red')
plt.plot(x_liso, power_law(x_liso, a_liso, b_liso), label=f"Power law Liso", color= 'purple')
plt.xlabel("Masa (gr)")
plt.ylabel("Diámetro (cm)")
plt.legend()

plt.tight_layout()
plt.show()
# y=({a:.2f}±$\sigma_a$)x^({b:.2f}±{m_err:.2f})
# # Es importante analizar los coeficientes junto a sus incertezas asociadas
# print(f"Pendiente (m): {m:.2f} ± {m_err:.2f}")
# print(f"Ordenada al origen (c): {c:.2f} ± {c_err:.2f}")