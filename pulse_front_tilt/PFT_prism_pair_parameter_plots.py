from tools import PFT_change_interface, PFT_prism_sym, PFT_prism_sym_test, PFT_prism_sym_test2
from materials import n_BBO, n_CaF2, n_fused_silica, n_SF11
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


######################
## SETUP PARAMETERS ##
######################

theta_1_range_deg = (10, 60)                            # apex angle of first prism in degrees
lmd = 400                                               # wavelength in nm
n1_func = n_SF11                                        # refractive index function of first prism
n2_func = n_fused_silica                                # refractive index function of second prism
theta_1_test_vals = [np.radians(v) for v in [6, 10, 14, 18]]                 # test apex angle of first prism in radians
detuning_range = (-5, 5)

##########
# SCRIPT #
##########

# refractive indices
n1 = n1_func(lmd)                           # refractive index of first prism
n2 = n2_func(lmd)                           # refractive index of second prism



# setup parameter grid
theta_1_array = np.linspace(np.radians(theta_1_range_deg[0]), np.radians(theta_1_range_deg[1]), 100)
theta_2_array = np.zeros_like(theta_1_array)
error_array = np.zeros_like(theta_1_array)
PFT_1_array = np.zeros_like(theta_1_array)
PFT_2_array = np.zeros_like(theta_1_array)
PFT_total_array = np.zeros_like(theta_1_array)
PFT_total_simple_array = np.zeros_like(theta_1_array)
theta_2_test_vals = []
theta_1_test_arrays = []
diff_test_arrays = []
delta_phi_i1_arrays = []
delta_phi_e2_arrays = []
PFT_detune_arrays = []


def angle_detuning(delta_phi_i1, theta_1, theta_2, n1, n2):
    """
    Calculate the variation of the exit angle (2nd prism) from the symmetry angle, given the
    respective variation of the incident angle (1st prism).
    """

    phi_1 = np.arcsin(n1 * np.sin(theta_1 /2))
    phi_2 = np.arcsin(n2 * np.sin(theta_2 /2))

    # incident interface of 1st prism
    delta_eps_i1 = np.arcsin(1/n1 * np.sin(phi_1 + delta_phi_i1)) - theta_1/2

    # exit interface of 1st prism
    delta_eps_e1 = - delta_eps_i1
    delta_phi_e1 = np.arcsin(n1 * np.sin(theta_1/2 + delta_eps_e1)) - phi_1

    # incident interface of 2nd prism
    delta_phi_i2 = delta_phi_e1
    delta_eps_i2 = np.arcsin(1/n2 * np.sin(phi_2 + delta_phi_i2)) - theta_2/2

    # exit interface of 2nd prism
    delta_eps_e2 = - delta_eps_i2
    delta_phi_e2 = np.arcsin(n2 * np.sin(theta_2/2 + delta_eps_e2)) - phi_2

    return delta_phi_e2
    

def PFT_detuning(delta_phi_i1, theta_1, theta_2, n1_func, n2_func, lmd):
    """
    Calculate the variation of the pulse front tilt angle from the symmetric configuration, given the
    respective variation of the incident angle (1st prism).

    This function then calculates all necessary angle variations and returns the total PFT variation for the setup.
    """

    n1, n2 = n1_func(lmd), n2_func(lmd)

    phi_1 = np.arcsin(n1 * np.sin(theta_1 /2))
    phi_2 = np.arcsin(n2 * np.sin(theta_2 /2))

    # incident interface of 1st prism
    delta_eps_i1 = np.arcsin(1/n1 * np.sin(phi_1 + delta_phi_i1)) - theta_1/2

    # exit interface of 1st prism
    delta_eps_e1 = - delta_eps_i1
    delta_phi_e1 = np.arcsin(n1 * np.sin(theta_1/2 + delta_eps_e1)) - phi_1

    # incident interface of 2nd prism
    delta_phi_i2 = delta_phi_e1
    delta_eps_i2 = np.arcsin(1/n2 * np.sin(phi_2 + delta_phi_i2)) - theta_2/2

    # exit interface of 2nd prism
    delta_eps_e2 = - delta_eps_i2
    delta_phi_e2 = np.arcsin(n2 * np.sin(theta_2/2 + delta_eps_e2)) - phi_2

    # compute PFT without detuning
    PFT_1 = PFT_prism_sym(theta_1, n1_func, lmd=lmd)
    PFT_2 = PFT_prism_sym(theta_2, n2_func, lmd=lmd)

    # correct PFT with detuning
    PFT_1_detuned = PFT_1 * np.cos(phi_1) / ((np.sin(delta_eps_i1)*np.tan(theta_1/2) + np.cos(delta_eps_i1))*np.cos(phi_1 + delta_phi_e1))
    PFT_2_detuned = PFT_2 * np.cos(phi_2) / ((np.sin(delta_eps_i2)*np.tan(theta_2/2) + np.cos(delta_eps_i2))*np.cos(phi_2 + delta_phi_e2))

    delta_PFT_1 = PFT_1 - PFT_1_detuned
    delta_PFT_2 = PFT_2 - PFT_2_detuned

    return delta_PFT_1 - delta_PFT_2

    

def get_objective_1(theta_1):

    def objective(delta_theta):
        phi_i_2 = np.arcsin(n2 * np.sin(theta_1 /2 + delta_theta /2))
        phi_e_1 = np.arcsin(n1 * np.sin(theta_1 /2))
        diff = np.abs(phi_e_1 - (phi_i_2 - delta_theta/2))
        return diff
    
    return objective

def get_objective_2(theta_1):
    
    phi_e_1 = np.arcsin(n1 * np.sin(theta_1 /2))

    def objective(delta_theta):
        theta_2 = theta_1 + delta_theta
        phi_i_2 = np.arcsin(n2 * np.sin(theta_2 /2))
        diff = np.abs(2*phi_e_1 - theta_1 - (2*phi_i_2 - theta_2))
        return diff

    return objective


get_objective_used = get_objective_1

# scan second prism apex angle for theta_1_test
for theta_1 in theta_1_test_vals:
    
    delta_theta_test_array = np.linspace(-theta_1, np.pi/2 - theta_1, 400)
    objective_test = get_objective_used(theta_1)
    diff_test_array = objective_test(delta_theta_test_array)
    theta_1_test_arrays.append(delta_theta_test_array)
    diff_test_arrays.append(diff_test_array)

    delta_theta_min = - theta_1
    delta_theta_max = np.pi/2 - theta_1
    res = minimize_scalar(objective_test, bounds=(delta_theta_min, delta_theta_max), method='bounded')
    theta_2_test_vals.append(theta_1 + res.x)
    
# compute theta_2 for all theta_1 values
for i, theta_1 in enumerate(tqdm(theta_1_array)):

    # objective for finding theta_2
    objective = get_objective_used(theta_1)

    delta_theta_min = - theta_1
    delta_theta_max = np.pi/2 - theta_1
    res = minimize_scalar(objective, bounds=(delta_theta_min, delta_theta_max), method='bounded')
    theta_2_array[i] = theta_1 + res.x
    error_array[i] = res.fun

    # compute pulse front tilt angles
    PFT_1 = PFT_prism_sym(theta_1, n1_func, lmd=lmd)
    PFT_2 = PFT_prism_sym(theta_2_array[i], n2_func, lmd=lmd)
    PFT_1_array[i] = PFT_1
    PFT_2_array[i] = PFT_2
    PFT_total_array[i] = np.arctan(np.tan(PFT_1) - np.tan(PFT_2))
    PFT_total_simple_array[i] = PFT_1 - PFT_2

# compute angle detuning for theta_1_test_vals and theta_2_test_vals
for theta_1, theta_2 in zip(theta_1_test_vals, theta_2_test_vals):
    delta_phi_i1_array = np.linspace(np.radians(detuning_range[0]), np.radians(detuning_range[1]), 200)
    delta_phi_e2_array = angle_detuning(delta_phi_i1_array, theta_1, theta_2, n1, n2)
    delta_phi_i1_arrays.append(delta_phi_i1_array)
    delta_phi_e2_arrays.append(delta_phi_e2_array)
    PFT_detune_arrays.append(PFT_detuning(delta_phi_i1_array, theta_1, theta_2, n1_func, n2_func, lmd))



# plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(12, 9), nrows=2, ncols=2)

ax5 = ax2.twinx()

# plot angles
ax1.plot(np.degrees(theta_1_array), np.degrees(theta_2_array), label='Second prism apex angle (deg)')
ax1.set_xlabel('First prism apex angle (deg)')
ax1.set_ylabel('Second prism apex angle (deg)')

# plot lines for ordered prism
ax1.axhline(y=23.4, color='red', linestyle='--', label='ordered FS prism (23.4°)')
ax1.axvline(x=13.2, color='green', linestyle='--', label='ordered SF11 prism (13.2°)')

ax1.legend()
ax1.grid()

# plot detuning error
for dphi1_arr, dphi2_arr, theta_1, theta_2 in zip(delta_phi_i1_arrays, delta_phi_e2_arrays, theta_1_test_vals, theta_2_test_vals):
    ax2.plot(np.degrees(dphi1_arr), np.degrees(dphi2_arr - dphi1_arr), label=f'$\\theta_1$={np.degrees(theta_1):.1f}$\\degree$, $\\theta_2$={np.degrees(theta_2):.1f}$\\degree$')

ax2.set_xlabel(r'$\Delta \phi_{i,1}$ (deg)')
ax2.set_ylabel(r'$\Delta \phi_{e,2} - \Delta \phi_{i,1}$ (deg)')
ax2.legend(loc='upper left')
ax2.grid()

# plot PFT detuning
for dphi1_arr, PFT_detune_arr, theta_1, theta_2 in zip(delta_phi_i1_arrays, PFT_detune_arrays, theta_1_test_vals, theta_2_test_vals):
    ax5.plot(np.degrees(dphi1_arr), np.degrees(PFT_detune_arr), ls='--', label=f'$\\theta_1$={np.degrees(theta_1):.1f}$\\degree$, $\\theta_2$={np.degrees(theta_2):.1f}$\\degree$')

ax5.set_ylabel('PFT detuning (deg)')
ax5.legend(loc='lower right')

# plot pulse front tilt angles
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_1_array), label='PFT from first prism')
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_2_array), label='PFT from second prism')
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_total_array), label='Total PFT (arctan diff)')
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_total_simple_array), '--', label='Total PFT (simple diff)')
ax3.axvline(x=13.2, color='black', linestyle='--', label='ordered SF11 prism (13.2°)')

ax3.set_xlabel('First prism apex angle (deg)')
ax3.set_ylabel('Pulse front tilt angle (deg)')
ax3.legend()
ax3.grid()

# plot test case
for delta_theta_test_array, diff_test_array, theta_1 in zip(theta_1_test_arrays, diff_test_arrays, theta_1_test_vals):
    ax4.plot(np.degrees(delta_theta_test_array), np.degrees(diff_test_array), label=f'Angle deviation at $\\theta_1$={np.degrees(theta_1):.1f}$\\degree$')
ax4.set_xlabel(r'$\Delta \theta$ (deg)')
ax4.set_ylabel('Angle deviation (deg)')
ax4.legend()
ax4.grid()

plt.tight_layout()

# plot error
fig_inspect, ax_inspect = plt.subplots(figsize=(6, 4))
ax_inspect.plot(np.degrees(theta_1_array), np.degrees(error_array), color='red', label='Solver angle deviation')
ax_inspect.set_xlabel('First prism apex angle (deg)')
ax_inspect.set_ylabel('Angle deviation (deg)')
ax_inspect.legend()
ax_inspect.grid()
plt.tight_layout()

plt.show()