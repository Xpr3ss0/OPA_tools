import numpy as np
from scipy import constants as const
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from materials import n_BBO, v_g_BBO


# tools for calculating pulse front tilt angle

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
                            'prism refraction angle 2': angle of refraction (to surface normal) at the second prism surface in radians, or equivalent angle of incidence at second prism surface,
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