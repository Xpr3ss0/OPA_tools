import numpy as np
from scipy import constants as const
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from materials import n_BBO, v_g_BBO


# functions used for calculating phase matching

# CONVENTIONS:

# type: 'ooe' corresponds to signal ordinary, idler ordinary, pump extraordinary. While different types are supported, 
#   this only affects the refractive index calculations. However, the crystal angle theta is assumed to be independent of the non-collinear angle alpha.
#   This is only true if there is only one extraordinary beam (the pump), oriented at theta to the crystal axis. Then, signal and idler can be oriented at alpha relative to the pump, without impacting phase-matching.
#   For other phase matching types, the angle of more than one beam relative to the crystal axis is relevant, which means alpha and theta are not independent. 

# angles: alpha is pump-signal angle, theta is propagation angle relative to crystal axis
# all angles are expected in radians
# all wavelengths are expected in nm
# everything else in SI units (m, s, W, etc.)
# nothing has been explicitely vectorized, numpy arrays might work but not guaranteed

def get_k_vectors(lmd_s, theta=0, alpha=0, lmd_p=400, type='ooe'):
    """
    Get the wavevectors for the signal, idler, and pump beams.

    Args:
        lmd_s (float): Signal wavelength in nm.
        theta (float): Propagation angle in radians, determining respective indices of refraction. Defaults to 0.
        alpha (float): Non-collinear pump-signal angle in radians. Defaults to 0.
        lmd_p (float): Pump wavelength in nm. Defaults to 400.
        type (str): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.

    Returns:
        tuple: k_s, k_i, k_p (wavevectors in 1/m)
    """
    # frequencies in rad/s
    w_s = 2 * np.pi * const.c / (lmd_s * 1e-9)
    w_p = 2 * np.pi * const.c / (lmd_p * 1e-9)
    w_i = w_p - w_s

    # wavelengths in nm
    lmd_i = 2 * np.pi * const.c / w_i * 1e9

    # refractive indices and group velocities (in m/s) and wavevectors (abs. magnitude, in 1/m)
    type_boolean = [type[i] == 'e' for i in range(3)]
    n_s = n_BBO(lmd_s, type_boolean[0], theta)
    n_i = n_BBO(lmd_i, type_boolean[1], theta)
    n_p = n_BBO(lmd_p, type_boolean[2], theta)

    k_s = 2 * np.pi * n_s / (lmd_s * 1e-9)
    k_i = 2 * np.pi * n_i / (lmd_i * 1e-9)
    k_p = 2 * np.pi * n_p / (lmd_p * 1e-9)

    return k_s, k_i, k_p

def get_idler_angle(k_s, k_i, k_p, alpha, return_delta_k=False, method='analytical'):
    """
    Compute the idler angle. Can use two methods. Result should be the same.

    Args:
        k_s (float): Signal wavevector in 1/m.
        k_i (float): Idler wavevector in 1/m.
        k_p (float): Pump wavevector in 1/m.
        alpha (float): Pump-signal angle in radians.
        return_delta_k (bool): If True, also return the wavevector mismatch delta_k. Defaults to False.
        method (str): 'analytical' or 'numerical' method to compute the angles. Defaults to 'analytical'.

    Returns:
        tuple: omega (idler-signal angle in radians), beta (idler-pump angle in radians), delta_k (wavevector mismatch in 1/m, if return_delta_k is True)
    """

    if method == 'analytical':
        def objective(Omega):
            delta_k_par = k_p * np.cos(alpha) - k_s - k_i * np.cos(Omega)
            delta_k_perp = k_p * np.sin(alpha) - k_i * np.sin(Omega)
            delta_k = np.sqrt(delta_k_par**2 + delta_k_perp**2)
            return delta_k

        result = minimize_scalar(objective, bounds=(0, np.pi / 2), method='bounded')
        omega = result.x
        beta = result.x - alpha
        delta_k = objective(result.x)
    elif method == 'numerical':
        # compute vector (k_p - k_s) components (pump is along x-axis)
        p_minus_s_x = k_p - k_s * np.cos(alpha)
        p_minus_s_y = - k_s * np.sin(alpha)

        # angle between pump (x-axis) and (k_p - k_s): use arctan2(|y|, x)
        # take absolute value of y because beta is defined as a positive angle magnitude
        beta = np.arctan2(np.abs(p_minus_s_y), p_minus_s_x)
        omega = beta + alpha

        if return_delta_k:
            delta_k_par = k_p * np.cos(alpha) - k_s - k_i * np.cos(omega)
            delta_k_perp = k_p * np.sin(alpha) - k_i * np.sin(omega)
            delta_k = np.sqrt(delta_k_par**2 + delta_k_perp**2)

    if return_delta_k:
        return omega, beta, delta_k
    return omega, beta

def group_velocity_mismatch(lmd_s, theta=0, alpha=0, lmd_p=400, type='ooe', project=False):
    """
    Calculates the group velocity mismatch (GVM) between signal and idler in BBO for given parameters.
    GVM is defined as the difference in inverse group velocities (1/v_g) of signal and idler.
    
    Args:
        lmd_s (float): Signal wavelength in nm.
        theta (float): Propagation angle in radians, determining respective indices of refraction. Defaults to 0.
        alpha (float): Pump-signal angle in radians. If non-zero, the projected GVM (idler projected onto signal) is computed. Defaults to 0.
        lmd_p (float): Pump wavelength in nm. Defaults to 400.
        type (str): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
        project (bool or str): If false, no projection is done and GVM is computed in terms of absolute group velocities. 
                               If set to 'signal' or 'pump', the GVM is projected onto the signal or pump direction, respectively. Defaults to False.
                               Projection makes sense if the beam waist is large compared to the pulse length, as in this case the walk-off occurs mainly orthogonal to the pulse front (which should be tilted in this case).
                               The projection should then match the direction of the pulse front tilt (i.e. signal for a tilted pump).
        
    Returns:
        tuple: GVM_ps (s/m), GVM_pi (s/m), GVM_si (s/m)
            GVM_ps: Group velocity mismatch between pump and signal, projected onto signal direction
            GVM_pi: Group velocity mismatch between pump and idler, projected onto signal direction
            GVM_si: Group velocity mismatch between signal and idler, projected onto signal direction
    """
    k_s, k_i, k_p = get_k_vectors(lmd_s, theta, alpha, lmd_p, type)
    lmd_i = 2 * np.pi / k_i * 1e9

    v_g_s = v_g_BBO(lmd_s, type[0]=='e', theta)
    v_g_i = v_g_BBO(lmd_i, type[1]=='e', theta)
    v_g_p = v_g_BBO(lmd_p, type[2]=='e', theta)

    # find beta (idler-pump angle) that minimizes wavevector mismatch
    omega, beta, delta_k = get_idler_angle(k_s, k_i, k_p, alpha, return_delta_k=True)

    # compute projected GVM in s/m
    if project == 'signal':
        d_p = 1 / (v_g_p * np.cos(alpha))
        d_s = 1 / v_g_s
        d_i = 1 / (v_g_i * np.cos(omega))
    elif project == 'pump':
        d_p = 1 / v_g_p
        d_s = 1 / (v_g_s * np.cos(alpha))
        d_i = 1 / (v_g_i * np.cos(beta))
    else:
        d_p = 1 / v_g_p
        d_s = 1 / v_g_s
        d_i = 1 / v_g_i

    GVM_ps = d_p - d_s
    GVM_pi = d_p - d_i
    GVM_si = d_s - d_i

    return GVM_ps, GVM_pi, GVM_si, delta_k

def compute_k_mismatch(theta, lmd_s, alpha, lmd_p=400, type='ooe', method="analytical"):

    """
    Computes the wavevector mismatch for (NOPA) phase matching, given propagation angle, signal wavelength and pump-signal (NC) angle.
    The mismatch is computed as the difference in magnitude of the polarization response k-vector (vectorial difference k_p - k_s, DFG),
    and the idler k_vector.
    This represents the fact that the idler is emitted in the direction of the polarization response.
    A positive mismatch implies k_i > k_pol.

    Args:
        theta (float): Propagation angle in radians.
        lmd_s (float): Signal wavelength in nm.
        alpha (float): Pump-signal angle in radians.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
        method (str, optional): Method to compute idler angle ('analytical' or 'numerical'). Defaults to 'analytical'. As both methods should give the same result, there is no reason for numerical at the moment.

    Returns:
        float: Wavevector mismatch in 1/m. 
    """

    # frequencies in rad/s
    k_s, k_i, k_p = get_k_vectors(lmd_s, theta, alpha, lmd_p, type)

    omega, beta, delta_k = get_idler_angle(k_s, k_i, k_p, alpha, return_delta_k=True, method=method)

    return delta_k

def compute_k_mismatch_dumb(theta, lmd_s, alpha, lmd_p=400, type='ooe'):

    """
    Like compute_k_mismatch(), but computes the wavevector mismatch in a more "dumb" way, by assuming parallel mismatch is zero and computing 
    only the perpendicular component. The angle is computed from the assumption of parallel phase matching.

    Args:
        theta (float): Propagation angle in radians.
        lmd_s (float): Signal wavelength in nm.
        alpha (float): Pump-signal angle in radians.
        lmd_p (float, optional): Pump wavelength in nm. Defaults to 400.
        type (str, optional): Type of phase matching ('ooe' or 'eoo'). Defaults to 'ooe'.
        method (str, optional): Method to compute idler angle ('analytical' or 'numerical'). Defaults to 'analytical'. As both methods should give the same result, there is no reason for numerical at the moment.

    Returns:
        float: Wavevector mismatch in 1/m. 
    """

    # frequencies in rad/s
    k_s, k_i, k_p = get_k_vectors(lmd_s, theta, alpha, lmd_p, type)

    beta = np.arcsin( k_s * np.sin(alpha) / k_i ) # from k_p = k_s cos(alpha) + k_i cos(beta), assuming parallel mismatch is zero

    delta_k = k_p -k_s * np.cos(alpha) - k_i * np.cos(beta)

    return delta_k

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

def OPA_gain(theta, lmd_s, alpha, I_p, L, lmd_p=400, type='ooe', dB=True, dumb=False):
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
    if dumb:
        delta_k = compute_k_mismatch_dumb(theta, lmd_s, alpha, lmd_p, type)
    else:
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
    gain = Gamma_squared / g**2 * np.exp(2 * g * L) / 4
    
    if dB:
        gain_db = 10 * np.log10(gain)
        # gain_db = np.nan_to_num(gain_db, nan=0.0)
        return gain_db
    else:
        # gain = np.nan_to_num(gain, nan=1)
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

def pulse_front_tilt_angle(phi, theta, n_prism_func, theta_apex, f1, f2, lmd_p_prism=800, lmd_p_crystal=400, full_return=True):
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
        lmd_p_prism (float, optional): Pump wavelength in nm. Defaults to 800.
        lmd_p_crystal (float, optional): Pump wavelength in nm. Defaults to 400.
        ret_ext (bool, optional): Whether to return the external pulse front tilt angle. Defaults to False.

    Returns:
        float or dict: If full_return is False, returns the pulse front tilt angle inside the crystal in radians.
                       If full_return is True, returns a dictionary with the following keys:
                           'internal tilt': pulse front tilt angle inside the crystal in radians,
                            'external tilt': pulse front tilt angle before the crystal (after telescope) in radians,
                            'prism tilt': pulse front tilt angle introduced by the prism (before telescope) in radians,
                            'prism exit angle': angle of the beam exiting the prism in radians,
                            'prism refraction angle 1': angle of refraction (to surface normal) at the first prism surface in radians,
                            'prism refraction angle 2': angle of refraction (to surface normal) at the second prism surface in radians,
                            'prism incidence angle': angle of incidence (to surface normal) at the first prism surface in radians (argument phi),
                            'angle change 1': angle change at the first prism surface in radians,
                            'angle change 2': angle change at the second prism surface in radians,
                            'total angle change': total angle change through the prism in radians
    """

    # compute angle of refraction after first surface in the prism using Snell's law
    n_prism = n_prism_func(lmd_p_prism)
    phi_r = np.arcsin(np.sin(phi) / n_prism)

    # compute angle of incidence at prism exit face
    phi_i = theta_apex - phi_r

    # compute angle of refraction at prism exit face
    phi_t = np.arcsin(n_prism * np.sin(phi_i))

    # compute pulse front tilt outside the prism
    dn_dlambda = (n_prism_func(lmd_p_prism + 1) - n_prism_func(lmd_p_prism - 1)) / 2  # numerical derivative, in 1/nm
    tan_gamma_prism = - np.sin(theta_apex) / (np.cos(phi_r) * np.cos(phi_t)) * lmd_p_prism * dn_dlambda

    # compute pulse front tilt after telescope
    tan_gamma_ext = f1 / f2 * tan_gamma_prism

    # compute pulse front tilt inside the crystal
    v_g = v_g_BBO(lmd_p_crystal, extraordinary=True, theta=theta)

    tan_gamma_int = (v_g / const.c) * tan_gamma_ext

    if not full_return:
        return np.arctan(tan_gamma_int)
    else:

        angle_change_2 = phi_t - phi_i  # angle change at prism exit face
        angle_change_1 = phi - phi_r  # angle change at prism entry face
        angle_change = angle_change_1 + angle_change_2 # total angle change through prism

        result = {"internal tilt": np.arctan(tan_gamma_int),
                  "external tilt": np.arctan(tan_gamma_ext),
                  "prism tilt": np.arctan(tan_gamma_prism),
                  "prism exit angle": phi_t,
                  "prism refraction angle 1": phi_r,
                  "prism refraction angle 2": phi_i,
                  "prism incidence angle": phi,
                  "angle change 1": angle_change_1,
                  "angle change 2": angle_change_2,
                  "total angle change": angle_change}
        
        return result