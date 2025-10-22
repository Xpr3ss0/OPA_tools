from scipy.optimize import minimize_scalar
import numpy as np
from scipy import constants as const
from materials import n_BBO
from tools import get_k_vectors



def idler_angle_method_1(k_s, k_i, k_p, alpha):
    """
    Compute the idler angle using method 1
    """
    def objective(Omega):
        delta_k_par = k_p * np.cos(alpha) - k_s - k_i * np.cos(Omega)
        delta_k_perp = k_p * np.sin(alpha) - k_i * np.sin(Omega)
        delta_k = np.sqrt(delta_k_par**2 + delta_k_perp**2)
        return delta_k

    result = minimize_scalar(objective, bounds=(0, np.pi / 2), method='bounded')
    omega = result.x
    beta = result.x - alpha
    delta_k = objective(result.x)
    return omega, beta, delta_k

def idler_angle_method_2(k_s, k_i, k_p, alpha):
    """
    Compute the idler angle using method 2
    """
    def objective(beta):
        delta_k_par = k_p - k_s * np.cos(alpha) - k_i * np.cos(beta)
        delta_k_perp = k_s * np.sin(alpha) - k_i * np.sin(beta)
        delta_k = np.sqrt(delta_k_par**2 + delta_k_perp**2)
        return delta_k
    
    result = minimize_scalar(objective, bounds=(0, np.pi / 2), method='bounded')
    beta = result.x
    omega = beta + alpha
    delta_k = objective(result.x)
    return omega, beta, delta_k

def idler_angle_method_3(k_s, k_i, k_p, alpha):
    """
    Compute the idler angle using method 3 (analytical), by computing the angle between k_p and (k_p - k_s).
    """
    # compute vector (k_p - k_s) components (pump is along x-axis)
    p_minus_s_x = k_p - k_s * np.cos(alpha)
    p_minus_s_y = - k_s * np.sin(alpha)

    # angle between pump (x-axis) and (k_p - k_s): use arctan2(|y|, x)
    # take absolute value of y because beta is defined as a positive angle magnitude
    beta = np.arctan2(np.abs(p_minus_s_y), p_minus_s_x)
    omega = beta + alpha

    delta_k_par = k_p * np.cos(alpha) - k_s - k_i * np.cos(omega)
    delta_k_perp = k_p * np.sin(alpha) - k_i * np.sin(omega)
    delta_k = np.sqrt(delta_k_par**2 + delta_k_perp**2)

    return omega, beta, delta_k

def test_angle_methods(lmd_s, theta=0, alpha=0, lmd_p=400, type='ooe'):
    """
    Test the two idler angle calculation methods.
    1. minimize delta_k by splitting into parallel and perpendicular components wrt signal, find optimal idler-signal angle Omega
    2. minimize delta_k by splitting into parallel and perpendicular components wrt pump, find optimal idler-pump angle beta
    Returns:
    """
    k_s, k_i, k_p = get_k_vectors(lmd_s, theta, alpha, lmd_p=lmd_p)

    omega_list = []
    beta_list = []
    mismatch_list = []
    for fun in [idler_angle_method_1, idler_angle_method_2, idler_angle_method_3]:
        omega, beta, delta_k = fun(k_s, k_i, k_p, alpha)
        omega_list.append(omega)
        beta_list.append(beta)
        mismatch_list.append(delta_k)

    return omega_list, beta_list, mismatch_list

if __name__ == "__main__":
    # Parameters
    alpha_deg = 3.69  # degrees
    theta_deg = 31.21  # degrees
    lmd_p = 400  # nm
    signal_range = (500, 700)  # nm

    # Convert degrees to radians
    alpha = np.radians(alpha_deg)
    theta = np.radians(theta_deg)

    # Wavelengths
    lmd_s = np.linspace(signal_range[0], signal_range[1], 300)  # nm

    omega_arrays = [[], [], []]
    beta_arrays = [[], [], []]
    mismatch_arrays = [[], [], []]
    for lmd in lmd_s:
        omega_list, beta_list, mismatch_list = test_angle_methods(lmd, theta, alpha, lmd_p=lmd_p)
        for i in range(3):
            omega_arrays[i].append(omega_list[i])
            beta_arrays[i].append(beta_list[i])
            mismatch_arrays[i].append(mismatch_list[i])

    import matplotlib.pyplot as plt

    fig, (ax1, ax3) = plt.subplots(figsize=(12, 9), ncols=2)
    ax2 = ax1.twinx()

    # plot angles from all methods
    plt.sca(ax1)
    for i in range(3):
        plt.sca(ax1)
        plt.plot(lmd_s, np.degrees(omega_arrays[i]), label=f'Method {i+1} $\\omega$')
        plt.plot(lmd_s, np.degrees(beta_arrays[i]), label=f'Method {i+1} $\\beta$')
        plt.sca(ax2)
        plt.plot(lmd_s, np.array(mismatch_arrays[i]) * 1e-3, label=f'Method {i+1} $\\Delta k$', linestyle='--')  # convert to mm^-1

    # plot k-vectors for illustration
    lmd_s_illustrate = 650  # nm
    omega_list, beta_list, mismatch_list = test_angle_methods(lmd_s_illustrate, theta, alpha, lmd_p=lmd_p)
    k_s, k_i, k_p = get_k_vectors(lmd_s_illustrate, theta, alpha, lmd_p=lmd_p)

    # plot k-vectors as lines, with signal and pump starting from the origin, and idler from the signal tip
    origin = np.array([0, 0])
    k_s_vec = np.array([k_s * np.cos(alpha), k_s * np.sin(alpha)])
    k_p_vec = np.array([k_p, 0])
    
    plt.sca(ax3)
    plt.quiver(*origin, *k_p_vec, angles='xy', scale_units='xy', scale=1, color='r', label='k_p (pump)')
    plt.quiver(*origin, *k_s_vec, angles='xy', scale_units='xy', scale=1, color='g', label='k_s (signal)')

    idler_colors = ['b', 'm', 'c']
    for i in range(3):
        k_i_vec = np.array([k_i * np.cos(beta_list[i]), -k_i * np.sin(beta_list[i])])
        plt.quiver(*(k_s_vec), *k_i_vec, angles='xy', scale_units='xy', scale=1, color=idler_colors[i], label=f'k_i (idler) {i+1}')
 
    plt.legend()

    plt.sca(ax1)
    plt.xlabel('Signal Wavelength (nm)')
    plt.ylabel('Angle (degrees)')
    plt.legend(loc='upper left')

    plt.sca(ax2)
    plt.ylabel('Wavevector Mismatch (mm$^{-1}$)')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()