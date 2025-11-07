from materials import *

wavelength = 1030 # nm
tau_p = 150 # fs
energy_pulse = 10e-6 # J
peak_power = energy_pulse / (tau_p * 1e-15)  # W, valid for square pulse, approximate for Gaussian

F_th_Al2O3 = F_th(U_g_Al2O3, tau_p)  # J/cm^2
F_th_YAG = F_th(6.5, tau_p)  # J/cm^2
print(f"Fluence threshold for Al2O3 at {wavelength} nm and {tau_p} fs: {F_th_Al2O3:.2f} J/cm^2")
print(f"Fluence threshold for YAG at {wavelength} nm and {tau_p} fs: {F_th_YAG:.2f} J/cm^2")

threshold_beam_area = energy_pulse / F_th_Al2O3  * 1e-2 # m^2
threshold_w0 = (threshold_beam_area / 3.14159)**0.5  # m, assuming circular beam
threshold_NA = wavelength * 1e-9 / (3.14159 * threshold_w0)  # dimensionless
print(f"Threshold beam area for E_p={energy_pulse:.2e} J: {threshold_beam_area:.2e} m^2")
print(f"Threshold beam waist for E_p={energy_pulse:.2e} J: {threshold_w0:.2e} m")
print(f"Threshold numerical aperture for E_p={energy_pulse:.2e} J: {threshold_NA:.2e}")

P_cr_Al2O3_value = P_cr_Al2O3(wavelength)
print(f"Critical power for self-focusing in Al2O3 at {wavelength} nm: {P_cr_Al2O3_value*1e-6:.2f} MW")

power_ratio = peak_power / P_cr_Al2O3_value
print(f"Ratio of peak power to critical power: {power_ratio:.2f}")

print(f"Peak power of the pulse: {peak_power*1e-6:.2f} MW")

