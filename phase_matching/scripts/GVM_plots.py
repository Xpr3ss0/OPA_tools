import matplotlib.pyplot as plt
import numpy as np
from tools import group_velocity_mismatch, compute_k_mismatch

# Parameters
alpha_deg = 3.56  # degrees
theta_deg = 31.06  # degrees
lmd_p = 400  # nm
signal_range = (470, 800)  # nm

# Convert degrees to radians
alpha = np.radians(alpha_deg)
theta = np.radians(theta_deg)

# Wavelengths
lmd_s = np.linspace(signal_range[0], signal_range[1], 300)  # nm

if __name__ == "__main__":
    GVM_ps_array = []
    GVM_pi_array = []
    GVM_si_array = []
    k_mismatch_array = []

    for lmd in lmd_s:
        GVM_ps, GVM_pi, GVM_si, delta_k = group_velocity_mismatch(lmd, theta, alpha, lmd_p=lmd_p)
        GVM_ps_array.append(GVM_ps * 1e12)  # convert to fs/mm
        GVM_pi_array.append(GVM_pi * 1e12)  # convert to fs/mm
        GVM_si_array.append(GVM_si * 1e12)  # convert to fs/mm
        k_mismatch_array.append(delta_k * 1e-3)  # convert to mm^-1

    # convert lists to numpy arrays
    GVM_ps_array = np.array(GVM_ps_array)
    GVM_pi_array = np.array(GVM_pi_array)
    GVM_si_array = np.array(GVM_si_array)
    k_mismatch_array = np.array(k_mismatch_array)
    

    # plot results
    fig, (ax_ps, ax_si) = plt.subplots(figsize=(12, 5), ncols=2)
    ax_km = ax_si.twinx()

    ax_ps.plot(lmd_s, GVM_ps_array, label=r'GVM$_{ps}$ (pump-signal)', color='blue')
    ax_ps.plot(lmd_s, GVM_pi_array, label=r'GVM$_{pi}$ (pump-idler)', color='red')
    ax_si.plot(lmd_s, GVM_si_array, label=r'GVM$_{si}$ (signal-idler)', color='green')
    ax_km.plot(lmd_s, k_mismatch_array, label=r'$\Delta k$', color='orange')
    ax_si.plot([], [], label=r'$\Delta k$', color='orange')
    # ax_ps.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    fig.suptitle(f'Group Velocity Mismatch vs Signal Wavelength\n(α={alpha_deg}°, θ={theta_deg}°)')
    
    ax_ps.set_xlabel('Signal Wavelength (nm)')
    ax_ps.set_ylabel(r'GVM (fs/mm)')
    ax_ps.legend()
    ax_ps.grid()
    ax_si.set_ylabel(r'GVM (fs/mm)', color='green')
    ax_si.legend()
    ax_si.grid()
    ax_km.set_ylabel(r'$\Delta k$ (mm$^{-1}$)', color='orange')
    plt.tight_layout()
    plt.show()