import numpy as np
from scipy import constants as const
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from materials import n_BBO, v_g_BBO


# functions used for calculating phase matching

# CONVENTIONS:
# type: 'ooe' corresponds to signal ordinary, idler ordinary, pump extraordinary. While different types are supported, 
#   this only affects the refractive index calculations. Importently, the propagation angle theta is assumed to be the same for all three waves.
#   This might not be the case if the indices of refraction for the generated waves very with theta, as is the case for type II phase matching. 
# angles: alpha is pump-signal angle, theta is propagation angle relative to crystal axis
# all angles are expected in radians
# all wavelengths are expected in nm
# everything else in SI units (m, s, W, etc.)
# nothing has been explicitely vectorized, numpy arrays might work but not guaranteed

def group_velocity_mismatch(lmd_s, theta=0, alpha=0, lmd_p=400, type='ooe'):
    """
    Calculates the group velocity mismatch (GVM) between signal and idler in BBO for given parameters.
    GVM is defined as the difference in inverse group velocities (1/v_g) of signal and idler.
    
    Args:
        lmd_s (float): Signal wavelength in nm.
        theta (float): Propagation angle in radians, determining respective indices of refraction. Defaults to 0.
        alpha (float): Pump-signal angle in radians. If non-zero, the projected GVM (idler projected onto signal) is computed. Defaults to 0.
        lmd_p (float): Pump wavelength in nm. Defaults to 400.
        type (str): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
        
    Returns:
        tuple: GVM_is (s/m), GVM_ps (s/m), GVM_pi (s/m)
            GVM_is: Group velocity mismatch between idler and signal, projected onto signal direction
            GVM_ps: Group velocity mismatch between pump and signal, projected onto signal direction
            GVM_pi: Group velocity mismatch between pump and idler, projected onto idler direction

    Note: For NOPA phase matching, GVM_is should vanish for the phase matched wavelength.
    """
    # frequencies in rad/s
    w_s = 2 * np.pi * const.c / (lmd_s * 1e-9)
    w_p = 2 * np.pi * const.c / (lmd_p * 1e-9)
    w_i = w_p - w_s

    # wavelengths in nm
    lmd_i = 2 * np.pi * const.c / w_i * 1e9

    # refractive indices and group velocities (in m/s)
    type_boolean = [type[i] == 'e' for i in range(3)]
    n_s = n_BBO(lmd_s, type_boolean[0], theta)
    n_i = n_BBO(lmd_i, type_boolean[1], theta)
    n_p = n_BBO(lmd_p, type_boolean[2], theta)

    v_g_s = v_g_BBO(lmd_s, type_boolean[0], theta)
    v_g_i = v_g_BBO(lmd_i, type_boolean[1], theta)
    v_g_p = v_g_BBO(lmd_p, type_boolean[2], theta)

    # compute idler-pump angle from perpendicular phase matching condition
    k_s = 2 * np.pi * n_s / (lmd_s * 1e-9)
    k_i = 2 * np.pi * n_i / (lmd_i * 1e-9)
    beta = np.arcsin(k_s * np.sin(alpha) / k_i) # idler-pump angle
    Omega = beta + alpha # signal-idler angle

    # compute projected GVM in s/m
    GVM_is = 1 / (v_g_i * np.cos(Omega)) - 1 / v_g_s # idler projected onto signal
    GVM_ps = 1 / (v_g_p * np.cos(alpha)) - 1 / v_g_s # pump projected onto signal
    GVM_pi = 1 / (v_g_p * np.cos(beta)) - 1 / v_g_i  # pump projected onto idler

    return GVM_is, GVM_ps, GVM_pi

def compute_k_mismatch(theta, lmd_s, alpha, lmd_p=400, type='ooe'):

    """
    Computes the wavevector mismatch for NOPA phase matching, given propagation angle, signal wavelength and pump-signal angle.
    Idler-pump angle is computed from the perpendicular phase matching condition. 
    As a consequence, the perpendicular mismatch is zero by construction.
    Physically, this represents the fact that the idler direction is determined by the polarization response direction, which is fixed by the pump-signal angle.
    The parallel mismatch is then computed from the parallel phase matching condition, and returned (e.g. to be minimized).
    Pump wavelength can be specified, default is 400 nm (2nd harmonic of 800 nm).

    Args:
        theta (float): Propagation angle in radians.
        lmd_s (float): Signal wavelength in nm.
        alpha (float): Pump-signal angle in radians.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.

    Returns:
        float: Parallel wavevector mismatch in 1/m. Corresponds to full wavevector mismatch, by construction.
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

    # compute idler-pump angle from perpendicular phase matching condition
    beta = np.arcsin(k_s * np.sin(alpha) / k_i)

    # compute parallel wavevector mismatch in 1/m
    delta_k_par = k_s * np.cos(alpha) - k_p + k_i * np.cos(beta)

    return delta_k_par

def minimize_k_mismatch(lmd_s, alpha, lmd_p=400, type='ooe'):
    """
    Minimizes the wavevector mismatch for NOPA phase matching, given signal wavelength and pump-signal angle.
    The propagation angle is varied to minimize the parallel wavevector mismatch.
    Perpendicular mismatch is zero due to the idler-pump angle. See compute_k_mismatch() for details.

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
        return abs(delta_k)

    # Minimize the objective function
    result = minimize_scalar(objective, bounds=(0, np.pi / 2), method='bounded')

    # Return the optimal propagation angle and the corresponding wavevector mismatch
    return result.x, compute_k_mismatch(result.x, lmd_s, alpha, lmd_p, type)

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

def optimize_alpha(lmd_s_range, lmd_s_center=None, lmd_p=400, type='ooe', bounds=(0, np.pi/2), metric='theta_std'):
    """
    Optimizes the pump-signal angle for a given signal wavelength to minimize the wavevector mismatch.
    Returns the optimal angle and the corresponding propagation angle and wavevector mismatch.

    Args:
        lmd_s_range (tuple): Tuple (lmd_s_min, lmd_s_max) defining the range of signal wavelengths in nm.
        lmd_s_center (float, optional): Center signal wavelength in nm. If None, uses the midpoint of lmd_s_range. Defaults to None.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.

    Returns:
        tuple: Optimal pump-signal angle (in radians), corresponding propagation angle (in radians), 
               and wavevector mismatch (in 1/m), at center signal wavelength.
    """
    if lmd_s_center is None:
        lmd_s_center = (lmd_s_range[0] + lmd_s_range[1]) / 2

    lmd_s_array = np.linspace(lmd_s_range[0], lmd_s_range[1], 100)

    def objective(alpha):
        
        if metric == 'theta_std':
            # compute phase matching over the signal wavelength range
            theta_array, _ = phase_matching_array(lmd_s_array, alpha, lmd_p, type)

            # define metric of dependence of theta on lmd_s (to be minimized)
            metric_value = np.std(theta_array)  # standard deviation of theta over the wavelength range
        
        elif metric == 'delta_k_squares':
            # compute wavevector mismatch over the signal wavelength range, with phase matching at center wavelength
            theta_m, delta_k_m = minimize_k_mismatch(lmd_s_center, alpha, lmd_p, type)
            delta_k_array_m = np.array([compute_k_mismatch(theta_m, lmd_s, alpha, lmd_p, type) for lmd_s in lmd_s_array])

            # define metric of deviation of delta_k from zero (to be minimized)
            metric_value = np.sum(delta_k_array_m**2)

        return metric_value

    # Minimize the objective function
    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    optimal_alpha = result.x

    # compute phase matching at center signal wavelength
    theta_m, delta_k_m = minimize_k_mismatch(lmd_s_center, optimal_alpha, lmd_p, type)

    return optimal_alpha, theta_m, delta_k_m

def OPA_gain(theta, lmd_s, alpha, I_p, L, lmd_p=400, type='ooe', dB=True):
    """
    Calculates the OPA gain for given parameters.
    
    Args:
        theta (float): Propagation angle in radians.
        lmd_s (float): Signal wavelength in nm.
        alpha (float): Pump-signal angle in radians.
        I_p (float): Pump intensity in W/m^2.
        L (float): Interaction length in m.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.

    Returns:
        float: OPA gain (in dB).
    """
    # Compute wavevector mismatch
    delta_k = compute_k_mismatch(theta, lmd_s, alpha, lmd_p, type)

    # compute frequencies in rad/s
    w_p, w_s = [2 * np.pi * const.c / (lmd * 1e-9) for lmd in (lmd_p, lmd_s)]
    w_i = w_p - w_s
    lmd_i = 2 * np.pi * const.c / w_i * 1e9 # idler wavelength in nm

    # refractive indices using sellmeier equations
    n_s, n_i, n_p = [n_BBO(lmd, type[i] == 'e', theta) for i, lmd in enumerate((lmd_s, lmd_i, lmd_p))]

    # relevant nonlinear coefficients for BBO
    # values from https://doi.org/10.1016/S0925-3467(02)00360-9
    d22 = 2.11e-12 # m/V
    d31 = 0.26e-12 # m/V

    # assuming below that crystal orientation is optimized for given theta
    if type in ["ooe", "eoo", "oeo"]:
        d_eff = np.abs(d31 * np.cos(theta) + d22 * np.cos(theta))
    elif type in ["eeo", "oee", "eoe"]:
        d_eff = np.abs(d22 * np.cos(theta)**2)
    else:
        raise ValueError("Invalid type. Must be one of 'ooe', 'eoo', 'oeo', 'eeo', 'oee', 'eoe'.")
    
    # hardcoded value for testing
    # d_eff = 2e-12 # m/V

    # calculate gain
    Gamma_squared = 2 * w_i * w_s * d_eff**2 * I_p / (n_i * n_s * n_p * const.c**3 * const.epsilon_0)
    g = np.sqrt(Gamma_squared - (delta_k / 2)**2)
    gain = w_i / w_s * Gamma_squared / g**2 * np.sinh(g * L)**2
    
    if dB:
        gain_db = 10 * np.log10(gain)
        gain_db = np.nan_to_num(gain_db, nan=0.0)
        return gain_db
    else:
        gain = np.nan_to_num(gain, nan=0.0)
        return gain

def effective_alpha(alpha, theta, lmd_s, lmd_p=400, type='ooe', n_external=1.0, normal='pump'):
    """
    Calculates the effective pump-signal angle outside the crystal, given the desired angle inside the crystal.
    Assumes that either pump or signal is normal to the crystal surface (choose with 'normal' parameter).

    Args:
        alpha (float): internal pump-signal angle in radians.
        theta (float): internal propagation angle relative to crystal axis, in radians.
        lmd_s (float): Signal wavelength in nm.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
        n_external (float, optional): Refractive index of the external medium. Defaults to 1.0 (air).
        normal (str, optional): Which beam is normal to the crystal surface ('pump' or 'signal'). Defaults to 'pump'.

    Returns:
        float: Effective pump-signal angle in radians.
    """
    # Compute refractive indices
    n_s = n_BBO(lmd_s, theta=theta, extraordinary=(type[0] == 'e'))
    n_p = n_BBO(lmd_p, theta=theta, extraordinary=(type[2] == 'e'))

    # Calculate effective angle, from Snell's law n_ext*sin(alpha_eff) = n_int*sin(alpha)
    if normal == 'pump':
        # signal is refracted
        alpha_eff = np.arcsin(n_s * np.sin(alpha) / n_external)
    elif normal == 'signal':
        # pump is refracted
        alpha_eff = np.arcsin(n_p * np.sin(alpha) / n_external)

    return alpha_eff

def pulse_front_tilt_angle(phi, theta, n_prism_func, theta_apex, f1, f2, lmd_p=400, ret_ext=False):
    """
    Calculates the pulse front tilt angle inside the BBO crystal introduced by a prism + telescope setup, 
    depending on the incidence angle on the telescope.

    Args:
        phi (float): Incidence angle on the telescope in radians. Can be a numpy array.
        theta (float): Propagation angle in the crystal relative to crystal axis, in radians.
        n_prism_func (float): Function that takes wavelength in nm and returns refractive index of the prism material.
        theta_apex (float): Apex angle of the prism in radians.
        f1 (float): Focal length of the first lens in the telescope in m.
        f2 (float): Focal length of the second lens in the telescope in m.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        ret_ext (bool, optional): Whether to return the external pulse front tilt angle. Defaults to False.

    Returns:
        float: Pulse front tilt angle inside the crystal in radians.
        If ret_ext is True, also returns the external pulse front tilt angle in radians.

    """

    # compute angle of refraction in the prism using Snell's law
    n_prism = n_prism_func(lmd_p)
    phi_r = np.arcsin(np.sin(phi) / n_prism)

    # compute angle of refraction at prism exit face
    phi_t = np.arcsin(n_prism * np.sin(theta_apex - phi_r))

    # compute pulse front tilt outside the prism
    dn_dlambda = (n_prism_func(lmd_p + 1) - n_prism_func(lmd_p - 1)) / 2  # numerical derivative, in 1/nm
    tan_gamma_prism = - np.sin(theta_apex) / (np.cos(phi_r) * np.cos(phi_t)) * lmd_p * dn_dlambda

    # compute pulse front tilt after telescope
    tan_gamma_ext = f1 / f2 * tan_gamma_prism

    # compute pulse front tilt inside the crystal
    v_g = v_g_BBO(lmd_p, extraordinary=True, theta=theta)

    tan_gamma_int = (v_g / const.c) * tan_gamma_ext

    if ret_ext:
        return np.arctan(tan_gamma_int), np.arctan(tan_gamma_ext)
    else:
        return np.arctan(tan_gamma_int)

    # Plotting the results and save to image file
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lmd_s_array, np.degrees(theta_array))
    plt.title('Optimal Propagation Angle vs Signal Wavelength')
    plt.xlabel('Signal Wavelength (nm)')
    plt.ylabel('Optimal Propagation Angle (degrees)')
    #plt.xlim(22, 34)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(lmd_s_array, delta_k_array*1e-6)  # Convert to rad/mm
    plt.title('Wavevector Mismatch vs Signal Wavelength')
    plt.xlabel('Signal Wavelength (nm)')
    plt.ylabel(r'Wavevector Mismatch (rad/mm)')
    plt.grid()

    plt.tight_layout()
    plt.savefig('phase_matching_results.png')