import numpy as np
from scipy import constants as const
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from materials import n_BBO, v_g_BBO


# tools for calculating pulse front tilt angle
def PFT_change_interface(gamma_ext=None, gamma_int=None, lmd=400, theta=0, return_diff=True):
    """
    Computes the change in pulse front tilt angle when going through an interface from air into BBO.
    
    Args:
        gamma_ext (float): External pulse front tilt angle in radians. Must provide either this or gamma_int.
        gamma_int (float): Internal pulse front tilt angle in radians. Must provide either this or gamma_ext.
        lmd (float, optional): Wavelength in nm. Defaults to 400.
        theta (float, optional): Angle of propagation in the crystal relative to crystal axis, in radians. Defaults to 0.
        return_diff (bool, optional): Whether to return only the change in angle, as gamma_ext - gamma_int. Defaults to True.
    
    Returns:
        float: Change in pulse front tilt angle in radians.
    """

    if gamma_ext is None and gamma_int is None:
        raise ValueError("Either gamma_ext or gamma_int must be provided.")
    elif gamma_int is None:
        tan_gamma_int = (v_g_BBO(lmd, extraordinary=True) / const.c) * np.tan(gamma_ext)
        gamma_int = np.arctan(tan_gamma_int)
        if return_diff:
            return gamma_ext - gamma_int
        else:
            return gamma_int
    else:
        tan_gamma_ext = (const.c / v_g_BBO(lmd, extraordinary=True, theta=theta)) * np.tan(gamma_int)
        gamma_ext = np.arctan(tan_gamma_ext)
        if return_diff:
            return gamma_ext - gamma_int
        else:
            return gamma_ext


def PFT_prism_sym(theta_apex, n_prism_func, lmd=800):
    """
    Calculates the pulse front tilt angle introduced by a prism surrounded by air, given the apex angle and refractive index function.
    The setup is assumed to be symmetric, i.e. the angle of incidence equals the angle of exit.

    Args:
        n_prism_func (function): Function that takes wavelength in nm and returns refractive index of the prism material.
        theta_apex (float): Apex angle of the prism in radians.
        lmd (float, optional): Wavelength in nm. Defaults to 800.

    Returns:
        float: Pulse front tilt angle introduced by the prism in radians.
    """

    # compute angle of refraction at prism exit face
    n_prism = n_prism_func(lmd)
    phi_e = np.arcsin(n_prism * np.sin(theta_apex / 2))

    # compute pulse front tilt introduced by the prism
    dn_dlambda = (n_prism_func(lmd + 1) - n_prism_func(lmd - 1)) / 2  # numerical derivative, in 1/nm
    tan_gamma_prism = 2 * np.sin(theta_apex/2) /np.cos(phi_e) * lmd * np.abs(dn_dlambda)

    return np.arctan(tan_gamma_prism)


def PFT_prism_sym_test(theta_apex, n_prism_func, lmd=800):
    """
    Calculates the pulse front tilt angle introduced by a prism surrounded by air, given the apex angle and refractive index function.
    The setup is assumed to be symmetric, i.e. the angle of incidence equals the angle of exit.

    Args:
        n_prism_func (function): Function that takes wavelength in nm and returns refractive index of the prism material.
        theta_apex (float): Apex angle of the prism in radians.
        lmd (float, optional): Wavelength in nm. Defaults to 800.

    Returns:
        float: Pulse front tilt angle introduced by the prism in radians.
    """

    # compute angle of refraction at prism exit face
    n_prism = n_prism_func(lmd)

    # compute pulse front tilt introduced by the prism
    dn_dlambda = (n_prism_func(lmd + 1) - n_prism_func(lmd - 1)) / 2  # numerical derivative, in 1/nm
    tan_gamma_prism = 2 * np.tan(theta_apex/2) * np.abs(dn_dlambda) # this is the wrong formula for testing if I get Giovanni's numbers

    return np.arctan(tan_gamma_prism)


def PFT_prism_sym_test2(theta_apex, n_prism_func, lmd=800):
    """
    Calculates the pulse front tilt angle introduced by a prism surrounded by air, given the apex angle and refractive index function.
    The setup is assumed to be symmetric, i.e. the angle of incidence equals the angle of exit.

    Args:
        n_prism_func (function): Function that takes wavelength in nm and returns refractive index of the prism material.
        theta_apex (float): Apex angle of the prism in radians.
        lmd (float, optional): Wavelength in nm. Defaults to 800.

    Returns:
        float: Pulse front tilt angle introduced by the prism in radians.
    """

    # compute angle of refraction at prism exit face
    term1 = np.tan(theta_apex / 2)
    dn_dlambda = (n_prism_func(lmd + 1) - n_prism_func(lmd - 1)) / 2  # numerical derivative, in 1/nm
    tanPFT = term1 * 2 * lmd * dn_dlambda
    return np.arctan(tanPFT)

def PFT_telescope(gamma_in, f1=None, f2=None, M=None):
    """
    Calculates the pulse front tilt angle after a telescope.

    Args:
        gamma_in (float): Input pulse front tilt angle in radians.
        f1 (float, optional): Focal length of the first lens in the telescope in m.
        f2 (float, optional): Focal length of the second lens in the telescope in m.
        M (float, optional): Magnification of the telescope. If provided, f1 and f2 are ignored.

    Returns:
        float: Pulse front tilt angle after the telescope in radians.
    """

    if M is None:
        if f1 is None or f2 is None:
            raise ValueError("Either magnification M or both focal lengths f1 and f2 must be provided.")
        M = f2 / f1

    tan_gamma_out = (1 / M) * np.tan(gamma_in)

    return np.arctan(tan_gamma_out)



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