import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tools import pulse_front_tilt_angle
from materials import n_CaF2, n_FS

# script for plotting pulse front tilt angle and required incidence angle on prism

# Parameters obtained from phase matching
alpha_degree = 3.7  # Pump-signal angle in degrees
theta = np.radians(31.2) # critical phase matching angle in radians

# Prism parameters
lmd_p_prism = 800 # pump wavelength in nm, do PFM before SHG
lmd_p_BBO = 800/2 # pump wavelength in nm in BBO
f1_telescope = 200e-3 # focal length of first telescope lens in m
f2_telescope = 50e-3 # focal length of second telescope lens in m
material = 'FS'  # 'CaF2' or 'FS'
pol_vector = np.array([0, 1])  # example polarization vector, pol[0]: s-polarization, pol[1]: p-polarization

theta_apex_fixed = np.radians(45)  # fixed apex angle for plotting in radians
phi_i1_fixed = np.radians(2.71)  # fixed incidence angle for plotting in radians
phi_i2_fixed = np.radians(43.11)  # fixed incidence angle at second surface for plotting in radians
optimize_phi = True  # whether to optimize incidence angle for given apex angle or use fixed value


# n_funcs
materials_n_funcs = {
    'CaF2': n_CaF2,
    'FS': n_FS
}


def fresnel_reflectance(n1, n2, theta_i, pol_vector: np.ndarray = None):
    """
    Calculate the Fresnel reflectance for s and p polarizations at an interface.
    """
    R_s = ((n1 * np.cos(theta_i) - n2 * np.sqrt(1 - (n1 / n2 * np.sin(theta_i))**2)) /
            (n1 * np.cos(theta_i) + n2 * np.sqrt(1 - (n1 / n2 * np.sin(theta_i))**2)))**2
    
    R_p = ((n1 * np.sqrt(1 - (n1 / n2 * np.sin(theta_i))**2) - n2 * np.cos(theta_i)) /
            (n1 * np.sqrt(1 - (n1 / n2 * np.sin(theta_i))**2) + n2 * np.cos(theta_i)))**2
    
    if pol_vector is None:
        return R_s, R_p
    else:
        pol_vector = pol_vector / np.linalg.norm(pol_vector)
        R = (pol_vector[0]**2) * R_s + (pol_vector[1]**2) * R_p
        return R

def get_phi_opt(theta_apex):
    def objective(phi):
        alpha_tilt = pulse_front_tilt_angle(phi, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=False)
        return np.abs(alpha_tilt - np.radians(alpha_degree))
    
    # dirty way to find minimum incidence angle
    phi_test_array = np.linspace(0, np.pi/2, 1000)
    alpha_test_array = pulse_front_tilt_angle(phi_test_array, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=False)

    # phi_min: minimum incidence angle where alpha is valid (not NaN or infinite)
    phi_valid = phi_test_array[np.isfinite(alpha_test_array)]

    # phi_min = phi_test_array[np.where(np.isfinite(alpha_test_array))[0][0]]
    phi_min = phi_valid.min()
    phi_max = phi_valid.max()
    result_opt = minimize_scalar(objective, bounds=(phi_min, phi_max), method='bounded')
    return result_opt.x

if __name__ == "__main__":

    n_func = materials_n_funcs[material]

    # Optimize apex angle for maximum transmission
    def objective(theta_apex):
    
        phi_opt = get_phi_opt(theta_apex)

        result = pulse_front_tilt_angle(phi_opt, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=True)

        theta_i_1 = phi_opt
        theta_i_2 = result['prism refraction angle 2']
        n1 = 1.0  # air
        n2 = n_func(lmd_p_prism)
        R1 = fresnel_reflectance(n1, n2, theta_i_1, pol_vector)
        R2 = fresnel_reflectance(n2, n1, theta_i_2, pol_vector) # assume no change in polarization (approximately fine if transmission is high)
        T_total = (1 - R1) * (1 - R2)
        return 1 - T_total # minimize loss

    result_apex = minimize_scalar(objective, bounds=(np.radians(35), np.radians(88)), method='bounded')
    T_max = 1 - result_apex.fun 
    theta_apex_opt = result_apex.x
    phi_opt = get_phi_opt(theta_apex_opt)
    result_tilt = pulse_front_tilt_angle(phi_opt, theta, n_func, theta_apex_opt, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=True)

    print("\n\n --- Refractive Index --- \n")
    print(f'Refractive index of {material} at {lmd_p_prism} nm: {n_func(lmd_p_prism):.4f}\n')

    print("\n--- Optimal Parameters ---\n")
    print(f'Optimal apex angle for maximum transmission: {np.degrees(theta_apex_opt):.2f} degrees')
    print(f'Transmission: {T_max:.4f}')
    print(f'Prism Incidence angle φ: {np.degrees(phi_opt):.2f} degrees')
    print(f'Incidence at second surface: {np.degrees(result_tilt["prism refraction angle 2"]):.2f} degrees\n')

    if optimize_phi:
        phi_opt = get_phi_opt(theta_apex_fixed)

        result = pulse_front_tilt_angle(phi_opt, theta, n_func, theta_apex_fixed, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=True)

        phi_i1_fixed = phi_opt
        phi_i2_fixed = result['prism refraction angle 2']
    
    R1 = fresnel_reflectance(1.0, n_func(lmd_p_prism), phi_i1_fixed, pol_vector)
    R2 = fresnel_reflectance(n_func(lmd_p_prism), 1.0, phi_i2_fixed, pol_vector) # assume no change in polarization (approximately fine if transmission is high)
    
    T_fixed = (1 - R1) * (1 - R2)
    print("\n\n--- Comparison to Fixed Parameters ---\n")
    print(f'Fixed apex angle: {np.degrees(theta_apex_fixed):.2f} degrees')
    print(f'Transmission: {T_fixed:.4f}')
    print(f'Prism Incidence angle φ: {np.degrees(phi_i1_fixed):.2f} degrees')
    print(f'Incidence at second surface: {np.degrees(phi_i2_fixed):.2f} degrees\n')