import numpy as np
from scipy import constants as const
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


# functions used for calculating phase matching

def n_BBO(wavelength, extraordinary=False, theta=0):
    """
    Calculates the refractive index of BBO for a given wavelength in nm.
    If extraordinary is True, calculates the extraordinary refractive index, at angle theta (in radians) to the optical axis.
    If extraordinary is False, calculates the ordinary refractive index.
    
    Args:
        wavelength (float): Wavelength in nm.
        extraordinary (bool): Whether to calculate the extraordinary refractive index. Defaults to False.
        theta (float): Angle in radians to the optical axis for extraordinary index. Defaults to 0.
        
    Returns:
        float: Refractive index of BBO at the given wavelength.
        Source of coefficients: https://www.newlightphotonics.com/v1/alpha-BBO-properties.html
    """
    wl = wavelength / 1000  # Convert nm to micrometers

    if not extraordinary:
        n_o = np.sqrt(2.67579 + 0.02099 / (wl**2 - 0.00470) - 0.00528 * wl**2)
        return n_o
    elif theta == 0:
        n_e = np.sqrt(2.31197 + 0.01184 / (wl**2 - 0.01607) - 0.00400 * wl**2)
        return n_e
    else:
        n_e = n_BBO(wavelength, extraordinary=True) # extraordinary index at theta=0
        n_o = n_BBO(wavelength, extraordinary=False)
        n_theta = np.sqrt(1 / (np.sin(theta)**2 / n_e**2 + np.cos(theta)**2 / n_o**2))
        return n_theta

def v_g_BBO(wavelength, extraordinary=False, theta=0):
    """Calculates the group velocity of BBO for a given wavelength in nm.
    
    Args:
        wavelength (float): Wavelength in nm.
        
    Returns:
        float: Group velocity of BBO at the given wavelength in m/s.
    """
    wl = wavelength / 1000  # Convert nm to micrometers
    n = n_BBO(wavelength, extraordinary, theta)

    # Calculate dn/dλ using numerical differentiation
    delta_wl = 1e-5  # Small change in wavelength in micrometers
    n_plus = n_BBO((wl + delta_wl) * 1000, extraordinary, theta)
    n_minus = n_BBO((wl - delta_wl) * 1000, extraordinary, theta)
    dn_dwl = (n_plus - n_minus) / (2 * delta_wl)

    # Group index calculation
    n_g = n - wl * dn_dwl

    # Speed of light in vacuum (m/s)
    c = const.c

    # Group velocity
    v_g = c / n_g

    return v_g

def compute_k_mismatch(theta, lmd_s, alpha, lmd_p=400, type='ooe'):

    """
    Computes the wavevector mismatch for NOPA phase matching, given propagation angle, signal wavelength and pump-signal angle.
    Signal-idler angle is computed from perpendicular phase matching condition.
    The parallel mismatch is then computed from the parallel phase matching condition, and returned (to be minimized).
    Pump wavelength can be specified, default is 400 nm (2nd harmonic of 800 nm).

    Args:
        theta (float): Propagation angle in radians.
        lmd_s (float): Signal wavelength in nm.
        alpha (float): Pump-signal angle in radians.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
    """

    # frequencies in rad/s
    w_s = 2 * np.pi * const.c / (lmd_s * 1e-9)
    w_p = 2 * np.pi * const.c / (lmd_p * 1e-9)
    w_i = w_p - w_s

    # wavelengths in nm
    lmd_i = 2 * np.pi * const.c / w_i * 1e9

    # refractive indices, wavevectors (abs. magnitude, in 1/m) and group velocities (in m/s)
    type_boolean = [type[i] == 'e' for i in range(3)]
    n_s = n_BBO(lmd_s, type_boolean[0], theta)
    n_i = n_BBO(lmd_i, type_boolean[1], theta)
    n_p = n_BBO(lmd_p, type_boolean[2], theta)

    k_s = 2 * np.pi * n_s / (lmd_s * 1e-9)
    k_i = 2 * np.pi * n_i / (lmd_i * 1e-9)
    k_p = 2 * np.pi * n_p / (lmd_p * 1e-9)

    # compute signal-idler angle from perp. phase matching conditions
    omega = np.arcsin(k_p / k_i * np.sin(alpha))

    # compute parallel wavevector mismatch in 1/m
    delta_k = k_p * np.cos(alpha) - k_s - k_i * np.cos(omega)

    return delta_k


def minimize_k_mismatch(lmd_s, alpha, lmd_p=400, type='ooe'):
    """
    Minimizes the wavevector mismatch for NOPA phase matching, given signal wavelength and pump-signal angle.
    The propagation angle is varied to minimize the parallel wavevector mismatch.

    Args:
        lmd_s (float): Signal wavelength in nm.
        alpha (float): Pump-signal angle in radians.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
    """
    # Initial guess for the propagation angle (in radians)
    initial_theta = np.pi / 4  # 45 degrees

    # Define the objective function to minimize
    def objective(theta):
        delta_k = compute_k_mismatch(theta, lmd_s, alpha, lmd_p, type)
        return np.abs(delta_k)

    # Minimize the objective function
    result = minimize_scalar(objective, bounds=(0, np.pi / 2), method='bounded')

    # Return the optimal propagation angle and the corresponding wavevector mismatch
    return result.x, result.fun

def phase_matching_array(lmd_s_array, alpha, lmd_p=400, type='ooe'):
    """
    Computes the optimal propagation angle and wavevector mismatch for an array of signal wavelengths.
    Returns two arrays: one for the optimal angles and one for the corresponding wavevector mismatches.
    Args:
        lmd_s_array (array-like): Array of signal wavelengths in nm.
        alpha (float): Pump-signal angle in radians.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
    """
    theta_array = np.zeros_like(lmd_s_array)
    delta_k_array = np.zeros_like(lmd_s_array)

    for i, lmd_s in enumerate(lmd_s_array):
        theta_opt, delta_k_opt = minimize_k_mismatch(lmd_s, alpha, lmd_p, type)
        theta_array[i] = theta_opt
        delta_k_array[i] = delta_k_opt

    return theta_array, delta_k_array

if __name__ == "__main__":
    # Example usage
    alpha = np.radians(3.7)  # Pump-signal angle in radians

    lmd_s_array = np.linspace(800, 450, 100)  # Signal wavelengths from 500 nm to 800 nm
    theta_array, delta_k_array = phase_matching_array(lmd_s_array, alpha)

    # Plotting the results and save to image file
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.degrees(theta_array), lmd_s_array)
    plt.title('Optimal Propagation Angle vs Signal Wavelength')
    plt.ylabel('Signal Wavelength (nm)')
    plt.xlabel('Optimal Propagation Angle (degrees)')
    plt.xlim(22, 34)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(lmd_s_array, delta_k_array)
    plt.title('Wavevector Mismatch vs Signal Wavelength')
    plt.xlabel('Signal Wavelength (nm)')
    plt.ylabel(r'Wavevector Mismatch (2$\pi$/m)')
    plt.grid()

    plt.tight_layout()
    plt.savefig('phase_matching_results.png')