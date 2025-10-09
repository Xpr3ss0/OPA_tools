import numpy as np
from scipy import constants as const
from tools import n_BBO


# script for retrieving d_eff from gain calculation in https://doi.org/10.1088/2040-8978/18/10/103501

# parameters as given in paper
G = 6
L = 1e-3
I_p = 25e13
lmd_p = 800e-9
lmd_s = 1.2e-6
lmd_i = 1/(1/lmd_p - 1/lmd_s)

# calculate from sellmeier
n_s, n_i, n_p = n_BBO(lmd_s*1e9), n_BBO(lmd_i*1e9), n_BBO(lmd_p*1e9)

Gamma = np.log(4*G)/(2*L)

d_eff_squared = Gamma**2 * const.c * const.epsilon_0 * n_s * n_i * n_p * lmd_s* lmd_i / (8*np.pi**2 * I_p)


print(Gamma)
print(np.sqrt(d_eff_squared))