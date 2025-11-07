import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tools import pulse_front_tilt_angle
from materials import n_CaF2, n_fused_silica

# script for plotting pulse front tilt angle and required incidence angle on prism

# Parameters obtained from phase matching
alpha_degree = 3.7  # Pump-signal angle in degrees
theta = np.radians(31.2) # critical phase matching angle in radians

# Prism parameters
lmd_p_prism = 800 # pump wavelength in nm, do PFM before SHG
lmd_p_BBO = 800/2 # pump wavelength in nm in BBO
f1_telescope = 200e-3 # focal length of first telescope lens in m
f2_telescope = 50e-3 # focal length of second telescope lens in m
material = 'CaF2'  # 'CaF2' or 'FS'
pol_vector = np.array([0, 1])  # example polarization vector, pol[0]: s-polarization, pol[1]: p-polarization
apex_range = (np.radians(45), np.radians(83))  # apex angle range to consider in radians



# n_funcs
materials_n_funcs = {
    'CaF2': n_CaF2,
    'FS': n_fused_silica
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
    phi_min = phi_test_array[np.where(np.isfinite(alpha_test_array))[0][0]]
    result_opt = minimize_scalar(objective, bounds=(phi_min, np.pi/2), method='bounded')
    return result_opt.x

if __name__ == "__main__":

    n_func = materials_n_funcs[material]

    apex_array = np.linspace(apex_range[0], apex_range[1], 500)
    transmission_array = np.zeros_like(apex_array)
    theta_i_1_array = np.zeros_like(apex_array)
    theta_i_2_array = np.zeros_like(apex_array)

    # Optimize apex angle for maximum transmission
    for i, apex in enumerate(apex_array):

        phi_opt = get_phi_opt(apex)
        result = pulse_front_tilt_angle(phi_opt, theta, n_func, apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=True)

        theta_i_1 = phi_opt
        theta_i_2 = result['prism refraction angle 2']
        n1 = 1.0  # air
        n2 = n_func(lmd_p_prism)
        R1 = fresnel_reflectance(n1, n2, theta_i_1, pol_vector)
        R2 = fresnel_reflectance(n2, n1, theta_i_2, pol_vector) # assume no change in polarization (approximately fine if transmission is high)
        T_total = (1 - R1) * (1 - R2)
        transmission_array[i] = T_total
        theta_i_1_array[i] = theta_i_1
        theta_i_2_array[i] = theta_i_2

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    brewster_1 = np.arctan(n_func(lmd_p_prism))  # Brewster angle for air to prism
    brewster_2 = np.arctan(1 / n_func(lmd_p_prism))  # Brewster angle for prism to air

    plt.sca(ax)
    plt.plot(np.degrees(apex_array), transmission_array)
    plt.xlabel("Apex Angle (degrees)")
    plt.ylabel("Transmission", color='blue')
    plt.title(f"Prism Transmission {material}")
    plt.grid()

    plt.sca(ax2)
    plt.plot(np.degrees(apex_array), np.degrees(theta_i_1_array), label=r'$\theta_{i,1}$', color='orange')
    plt.plot(np.degrees(apex_array), np.degrees(theta_i_2_array), label=r'$\theta_{i,2}$', color='green')
    plt.axhline(np.degrees(brewster_1), color='orange', linestyle='--', label=r'$\theta_{B,1}$')
    plt.axhline(np.degrees(brewster_2), color='green', linestyle='--', label=r'$\theta_{B,2}$')
    plt.ylabel("Incidence Angles (degrees)")
    plt.legend()
    plt.show()