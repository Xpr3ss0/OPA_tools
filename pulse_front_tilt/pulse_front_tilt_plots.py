import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tools import pulse_front_tilt_angle
from materials import n_CaF2, n_fused_silica

# script for plotting pulse front tilt angle and required incidence angle on prism

# Parameters obtained from phase matching
alpha_degree = 3.7  # Pump-signal angle in degrees
theta = np.radians(31.2) # critical phase matching angle in radians

# Prism parameters
lmd_p_prism = 800 # pump wavelength in nm, do PFM before SHG
lmd_p_BBO = 800/2 # pump wavelength in nm in BBO
f1_telescope = 200e-3 # focal length of first telescope lens in m
f2_telescope = 50e-3 # focal length of second telescope lens in m

# plot parameters
phi_range = (0, np.pi/2)
apex_angles = [40, 43, 46, 49] # apex angles to consider, degrees


if __name__ == "__main__":

    phi_array = np.linspace(phi_range[0], phi_range[1], 100)
    plt.figure(figsize=(12, 4*len(apex_angles)))
    apex_angles = [np.radians(a) for a in apex_angles]

    n = 1
    
    for theta_apex in apex_angles:

        

        for n_func, material in [(n_CaF2, 'CaF₂ (GVD: 68 fs²/mm)'), (n_fused_silica, 'Fused Silica (GVD: 98 fs²/mm)')]:

            result = pulse_front_tilt_angle(phi_array, theta, n_func, 
                                                                      theta_apex, f1_telescope, f2_telescope,
                                                                      lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=True)
            
            def objective(phi):
          
                alpha_tilt = pulse_front_tilt_angle(phi, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=False)
                return np.abs(alpha_tilt - np.radians(alpha_degree))
            
            
            # dirty way to find minimum incidence angle
            phi_test_array = np.linspace(0, np.pi/2, 1000)
            alpha_test_array = pulse_front_tilt_angle(phi_test_array, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=False)

            # phi_min: minimum incidence angle where alpha is valid (not NaN or infinite)
            phi_min = phi_test_array[np.where(np.isfinite(alpha_test_array))[0][0]]

            result_opt = minimize_scalar(objective, bounds=(phi_min, np.pi/2), method='bounded')
            phi_opt = result_opt.x

            gamma_int_array, gamma_ext_array = result["internal tilt"], result["external tilt"]

            plt.subplot(len(apex_angles), 2, n)
            plt.plot(np.degrees(phi_array), np.degrees(gamma_int_array), label='Inside BBO')
            plt.plot(np.degrees(phi_array), np.degrees(gamma_ext_array), label='Outside BBO')
            plt.axhline(alpha_degree, color='red', linestyle='--', label=r'$\alpha, \Delta \alpha_\mathrm{opt}: %.2f$' % np.degrees(result_opt.fun))
            plt.axvline(np.degrees(phi_opt), color='green', linestyle=':', label=r'$\phi_\mathrm{opt}: %.1f \degree$' % np.degrees(phi_opt))
            plt.axvline(np.degrees(phi_min), color='orange', linestyle='-.', label=r'$\phi_\mathrm{min}: %.1f \degree$' % np.degrees(phi_min))
            plt.ylim(0, 15)
            plt.xlabel(r'Incidence Angle $\phi$ (degrees)')
            plt.ylabel(r'Pulse Front Tilt Angle $\gamma$ (degrees)')
            plt.title(f'{material} Prism, {np.degrees(theta_apex):.1f}° Apex Angle')
            plt.legend()
            plt.grid()
            
            n += 1

    plt.tight_layout()

    # plt.savefig('phase_matching/plots/pulse_front_tilt.png', dpi=300)
    plt.show()
