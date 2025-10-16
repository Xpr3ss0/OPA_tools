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

def n_BBO_Tamosauskas(wavelength, extraordinary=False, theta=0):
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
        Source of coefficients: 
            https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Tamosauskas-o for ordinary
            https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Tamosauskas-e for extraordinary
    """
    wl = wavelength / 1000  # Convert nm to micrometers

    if not extraordinary:
        n_o = np.sqrt(1 + 0.90291 * wl**2 / (wl**2 - 0.003926) + 0.83155 * wl**2 / (wl**2 - 0.018786) + 0.76536 * wl**2 / (wl**2 - 60.01))
        return n_o
    elif theta == 0:
        n_e = np.sqrt(1 + 1.151075 * wl**2 / (wl**2 - 0.007142) + 0.21803 * wl**2 / (wl**2 - 0.02259) + 0.656 * wl**2 / (wl**2 - 263))
        return n_e
    else:
        n_e = n_BBO_Tamosauskas(wavelength, extraordinary=True) # extraordinary index at theta=0
        n_o = n_BBO_Tamosauskas(wavelength, extraordinary=False)
        n_theta = np.sqrt(1 / (np.sin(theta)**2 / n_e**2 + np.cos(theta)**2 / n_o**2))
        return n_theta

def n_BBO_Zhang(wavelength, extraordinary=False, theta=0):
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
        Source of coefficients: 
            https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Zhang-o for ordinary
            https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Zhang-e for extraordinary
            These coefficients are also used in LWE.
    """
    wl = wavelength / 1000  # Convert nm to micrometers

    if not extraordinary:
        n_o = np.sqrt(2.7359 + 0.01878 / (wl**2 - 0.01822) - 0.01471 * wl**2 + 0.0006081 * wl**4 - 0.00006740*wl**6)
        return n_o
    elif theta == 0:
        n_e = np.sqrt(2.3753 + 0.01224 / (wl**2 - 0.01667) - 0.01627 * wl**2 + 0.0005716 * wl**4 - 0.00006305 * wl**6)
        return n_e
    else:
        n_e = n_BBO_Zhang(wavelength, extraordinary=True) # extraordinary index at theta=0
        n_o = n_BBO_Zhang(wavelength, extraordinary=False)
        n_theta = np.sqrt(1 / (np.sin(theta)**2 / n_e**2 + np.cos(theta)**2 / n_o**2))
        return n_theta

# alias for current use
n_BBO = n_BBO_Zhang

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