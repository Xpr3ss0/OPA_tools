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
theta_1_test_arrays = []
diff_test_arrays = []

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
for test_angle in theta_1_test_vals:
    
    delta_theta_test_array = np.linspace(-test_angle, np.pi/2 - test_angle, 400)
    objective_test = get_objective_used(test_angle)
    diff_test_array = objective_test(delta_theta_test_array)
    theta_1_test_arrays.append(delta_theta_test_array)
    diff_test_arrays.append(diff_test_array)
    

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

# plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(12, 9), nrows=2, ncols=2)

# plot angles
ax1.plot(np.degrees(theta_1_array), np.degrees(theta_2_array), label='Second prism apex angle (deg)')
ax1.set_xlabel('First prism apex angle (deg)')
ax1.set_ylabel('Second prism apex angle (deg)')
ax1.legend()
ax1.grid()

# plot error
ax2.plot(np.degrees(theta_1_array), np.degrees(error_array), color='red', label='Angle deviation')
ax2.set_xlabel('First prism apex angle (deg)')
ax2.set_ylabel('Angle deviation (deg)')
ax2.legend()
ax2.grid()

# plot pulse front tilt angles
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_1_array), label='PFT from first prism')
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_2_array), label='PFT from second prism')
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_total_array), label='Total PFT (arctan diff)')
ax3.plot(np.degrees(theta_1_array), np.degrees(PFT_total_simple_array), '--', label='Total PFT (simple diff)')
ax3.set_xlabel('First prism apex angle (deg)')
ax3.set_ylabel('Pulse front tilt angle (deg)')
ax3.legend()
ax3.grid()

# plot test case
for delta_theta_test_array, diff_test_array, test_angle in zip(theta_1_test_arrays, diff_test_arrays, theta_1_test_vals):
    ax4.plot(np.degrees(delta_theta_test_array), np.degrees(diff_test_array), label=f'Angle deviation at $\\theta_1$={np.degrees(test_angle):.1f}$\\degree$')
ax4.set_xlabel('Delta theta (deg)')
ax4.set_ylabel('Angle deviation (deg)')
ax4.legend()
ax4.grid()

plt.tight_layout()
plt.show()