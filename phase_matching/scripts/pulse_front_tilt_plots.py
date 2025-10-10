import numpy as np
import matplotlib.pyplot as plt

from tools import pulse_front_tilt_angle
from materials import n_CaF2, n_fused_silica

# script for plotting pulse front tilt angle and required incidence angle on prism

# Parameters obtained from phase matching
alpha_degree = 3.93  # Pump-signal angle in degrees
theta = np.radians(30.42) # critical phase matching angle in radians

# Prism parameters
theta_apex = np.radians(60) # apex angle of the prism in radians
lmd_p = 395 # pump wavelength in nm
f1_telescope = 120e-3 # focal length of first telescope lens in m
f2_telescope = 50e-3 # focal length of second telescope lens in m

# plot parameters
phi_range = (0, np.pi/2)
apex_angles = [60, 45] # apex angles to consider, degrees


if __name__ == "__main__":

    phi_array = np.linspace(phi_range[0], phi_range[1], 100)
    plt.figure(figsize=(12, 4*len(apex_angles)))
    apex_angles = [np.radians(a) for a in apex_angles]

    n = 1
    
    for theta_apex in apex_angles:
        for n_func, material in [(n_CaF2, 'CaF₂ (GVD: 68 fs²/mm)'), (n_fused_silica, 'Fused Silica (GVD: 98 fs²/mm)')]:

            gamma_int_array, gamma_ext_array = pulse_front_tilt_angle(phi_array, theta, n_func, 
                                                                      theta_apex, f1_telescope, f2_telescope,
                                                                      ret_ext=True, lmd_p=lmd_p)
            plt.subplot(len(apex_angles), 2, n)
            plt.plot(np.degrees(phi_array), np.degrees(gamma_int_array), label='Inside BBO')
            plt.plot(np.degrees(phi_array), np.degrees(gamma_ext_array), label='Outside BBO')
            plt.axhline(alpha_degree, color='red', linestyle='--', label=r'$\alpha$')
            plt.ylim(0, 15)
            plt.xlabel(r'Incidence Angle $\phi$ (degrees)')
            plt.ylabel(r'Pulse Front Tilt Angle $\gamma$ (degrees)')
            plt.title(f'{material} Prism, {np.degrees(theta_apex):.1f}° Apex Angle')
            plt.legend()
            plt.grid()
            
            n += 1

    plt.tight_layout()

    plt.savefig('phase_matching/plots/pulse_front_tilt.png', dpi=300)
