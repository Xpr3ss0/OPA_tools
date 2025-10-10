import numpy as np
from scipy import constants as const


def n_CaF2(lmd):
    """
    Refractive index of CaF2, from Malitson, JOSA 1963
    Valid for 0.18 to 8 um at 24°C
    Args:
        lmd (float or array): Wavelength in nm
    Returns:
        float or array: Refractive index
    """
    # convert nm to um
    lmd = lmd / 1000

    n_squared = 1 + 0.5675888 * lmd**2 / (lmd**2 - 0.050263605**2) + \
                0.4710914 * lmd**2 / (lmd**2 - 0.1003909**2) + \
                3.8484723 * lmd**2 / (lmd**2 - 34.649040**2)
    
    return np.sqrt(n_squared)

def n_fused_silica(lmd):
    """
    Refractive index of fused silica, from Malitson, JOSA 1965
    Valid for 0.21 to 3.71 um at 20°C
    Args:
        lmd (float or array): Wavelength in nm
    Returns:
        float or array: Refractive index
    """

    # convert nm to um
    lmd = lmd / 1000

    n_squared = 1 + 0.6961663 * lmd**2 / (lmd**2 - 0.0684043**2) + \
                0.4079426 * lmd**2 / (lmd**2 - 0.1162414**2) + \
                0.8974794 * lmd**2 / (lmd**2 - 9.896161**2)

    return np.sqrt(n_squared)

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
        extraordinary (bool): Whether to calculate the group velocity for the extraordinary refractive index. Defaults to False.
        theta (float): Angle in radians to the optical axis for extraordinary index. Defaults to 0.
        
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