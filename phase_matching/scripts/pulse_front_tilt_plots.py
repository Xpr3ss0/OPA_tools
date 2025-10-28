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
theta_apex = np.radians(60) # apex angle of the prism in radians
lmd_p = 800 # pump wavelength in nm
f1_telescope = 300e-3 # focal length of first telescope lens in m
f2_telescope = 30e-3 # focal length of second telescope lens in m

# plot parameters
phi_range = (0, np.pi/2)
apex_angles = [60, 45, 40] # apex angles to consider, degrees


if __name__ == "__main__":

    phi_array = np.linspace(phi_range[0], phi_range[1], 100)
    plt.figure(figsize=(12, 4*len(apex_angles)))
    apex_angles = [np.radians(a) for a in apex_angles]

    n = 1
    
    for theta_apex in apex_angles:

        

        for n_func, material in [(n_CaF2, 'CaF₂ (GVD: 68 fs²/mm)'), (n_fused_silica, 'Fused Silica (GVD: 98 fs²/mm)')]:

            result = pulse_front_tilt_angle(phi_array, theta, n_func, 
                                                                      theta_apex, f1_telescope, f2_telescope,
                                                                      lmd_p=lmd_p, full_return=True)
            
            def objective(phi):
                alpha_tilt = pulse_front_tilt_angle(phi, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p=800, full_return=False)
                return np.abs(alpha_tilt - np.radians(alpha_degree))
            
            
            result_opt = minimize_scalar(objective, bounds=(0.05, np.pi/3), method='bounded')
            phi_opt = result_opt.x

            gamma_int_array, gamma_ext_array = result["internal tilt"], result["external tilt"]

            plt.subplot(len(apex_angles), 2, n)
            plt.plot(np.degrees(phi_array), np.degrees(gamma_int_array), label='Inside BBO')
            plt.plot(np.degrees(phi_array), np.degrees(gamma_ext_array), label='Outside BBO')
            plt.axhline(alpha_degree, color='red', linestyle='--', label=r'$\alpha, \Delta \alpha_\mathrm{opt}: %.2f$' % np.degrees(result_opt.fun))
            plt.axvline(np.degrees(phi_opt), color='green', linestyle=':', label=r'$\phi_\mathrm{opt}: %.1f \degree$' % np.degrees(phi_opt))
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
