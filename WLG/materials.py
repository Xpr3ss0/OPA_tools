from refractiveindex import RefractiveIndexMaterial

# collect needed materials
Al2O3_o = RefractiveIndexMaterial(shelf='main', book='Al2O3', page='Querry-o')
Al2O3_e = RefractiveIndexMaterial(shelf='main', book='Al2O3', page='Querry-e')

def F_th(U_g, tau_p):
    """
    Calculates the fluence threshold F_th in J/cm^2, see https://doi.org/10.1103/PhysRevB.71.115109.
    
    Args:
        U_g (float): Bandgap energy in eV.
        tau_p (float): Pulse duration in fs.

    Returns:
        float: Fluence threshold F_th in J/cm^2.
    """
    c1 = -0.16 # J/cm^2/fs^-kappa
    c2 = 0.074 # J/cm^2/fs^-kappa/eV
    kappa = 0.3
    return (c1 + c2 * U_g) * (tau_p ** kappa)

def P_cr(wavelength_nm, n_func, n2):
    """
    Calculates the critical power (in W) for self-focusing in Al2O3 (sapphire) at the given wavelength in nm.
    
    Args:
        wavelength_nm (float): Wavelength in nanometers.
        n_func (function): Function that returns the refractive index at the given wavelength.
        n2 (float): Nonlinear refractive index in m^2/W.
    
    """
    n0 = n_func(wavelength_nm)
    lmd = wavelength_nm * 1e-9 # convert to m
    P_cr = (3.77 * lmd**2) / (8 * 3.14159 * n0 * n2)  # in W
    return P_cr

def n_Al2O3(wavelength_nm, extraordinary=False):
    """Returns the refractive index of Al2O3 (sapphire) at the given wavelength in nm.
    
    Parameters:
        wavelength_nm (float): Wavelength in nanometers.
        extraordinary (bool): If True, returns the extraordinary refractive index. Defaults to False (ordinary index).
    """
    if extraordinary:
        return Al2O3_e.get_refractive_index(wavelength_nm)
    return Al2O3_o.get_refractive_index(wavelength_nm)

# constants (approximate)
n2_AlO3 = 3.1e-16  # cm^2/W, at 800 nm, but approximate for 1030 nm as well
U_g_Al2O3 = 9.9 # eV, bandgap energy of Al2O3

def P_cr_Al2O3(wavelength_nm):
    """Calculates the critical power (in W) for self-focusing in Al2O3 (sapphire) at the given wavelength in nm."""
    n0 = n_Al2O3(wavelength_nm)
    n2 = n2_AlO3 * 1e-4 # convert to m^2/W
    lmd = wavelength_nm * 1e-9 # convert to m
    return P_cr(wavelength_nm, n_Al2O3, n2)
