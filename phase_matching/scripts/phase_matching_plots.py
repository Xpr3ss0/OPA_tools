import numpy as np
import matplotlib.pyplot as plt
from tools import phase_matching_array, optimize_alpha, OPA_gain, compute_k_mismatch

# Parameters
alpha_values = [3.9, 4.0, 4.1] # degrees
signal_range = (500, 700) # nm
signal_lmd_m = 550
lmd_p = 400 # nm
L = 1e-3 # 1 mm
I_p = 50e13 # 25 GW/cm^2 = 25e13 W/m^2
gain_in_dB = False
type = 'ooe' # phase matching type, in principle all should be supported

# fine tuning
alpha_optimization = 'delta_k_squares' # 'theta_std' or 'delta_k_squares', chooses the metric for alpha optimization

angle_detuning_array = np.array([-0.1, 0, 0.1]) # degrees, detunes angle from optimized value
angle_detuning_single = 0 # degrees, single detuning value

detuning_mode = 'alpha' # 'alpha' or 'theta', chooses whether to detune alpha over array.

# nonlinear coefficients, not used here for now, hard-coded in OPA_gain function
# values from https://doi.org/10.1016/S0925-3467(02)00360-9
d22 = 2.11e-12 # m/V
d31 = 0.26e-12 # m/V


if __name__ == "__main__":

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    lmd_s_array = np.linspace(signal_range[0], signal_range[1], 100)

    # make theta plots for different alpha values
    for alpha_deg in alpha_values:
        alpha_rad = np.radians(alpha_deg)
        theta_array, delta_k_array = phase_matching_array(lmd_s_array, alpha_rad, lmd_p=lmd_p, type=type)

        plt.sca(ax1)
        plt.plot(lmd_s_array, np.degrees(theta_array), label=f'α={alpha_deg}°')

        plt.sca(ax2)
        plt.plot(lmd_s_array, delta_k_array * 1e-3, label=f'α={alpha_deg}°') # Convert to mm^-1

    # make theta plot for optimal alpha, theta_opt and delta_k_opt correspond to center wavelength (default)
    alpha_opt, theta_opt, delta_k_opt = optimize_alpha(signal_range, bounds=(0, np.radians(5)), 
                                                       lmd_p=lmd_p, metric=alpha_optimization, type=type, lmd_s_center=signal_lmd_m)

    plt.sca(ax1)
    theta_array, delta_k_array = phase_matching_array(lmd_s_array, alpha_opt, lmd_p=lmd_p, type=type)

    plt.plot(lmd_s_array, np.degrees(theta_array), label=f'α={np.degrees(alpha_opt):.2f}° (opt.)\n$\\theta_c$={np.degrees(theta_opt):.2f}°', linestyle='--')
    plt.sca(ax2)
    plt.plot(lmd_s_array, delta_k_array * 1e-3, label=f'α={np.degrees(alpha_opt):.2f}° (opt.)', linestyle='--')

    ax1.legend()
    ax1.grid()
    ax2.legend()
    ax2.grid()
    ax1.set_xlabel('Signal Wavelength (nm)')
    ax1.set_ylabel('Optimal Propagation Angle (degrees)')
    ax1.set_title('Optimal Propagation Angle vs Signal Wavelength')
    ax2.set_xlabel('Signal Wavelength (nm)')
    ax2.set_ylabel('Wavevector Mismatch (mm$^{-1}$)')
    ax2.set_title('Wavevector Mismatch vs Signal Wavelength')

    # make gain plot for optimal alpha and phase matching angle of center wavelength
    plt.sca(ax3)

    for angle_d in np.radians(angle_detuning_array):

        if detuning_mode == 'alpha':
            alpha_opt_detuned = alpha_opt + angle_d
            theta_opt_detuned = theta_opt + np.radians(angle_detuning_single)
        elif detuning_mode == 'theta':
            alpha_opt_detuned = alpha_opt + np.radians(angle_detuning_single)
            theta_opt_detuned = theta_opt + angle_d
        else:
            raise ValueError("detuning_mode must be 'alpha' or 'theta'")


        gain_array = np.zeros_like(lmd_s_array)
        for i, lmd_s in enumerate(lmd_s_array):
            gain_array[i] = OPA_gain(theta=theta_opt_detuned, lmd_s=lmd_s, alpha=alpha_opt_detuned, L=L, I_p=I_p, 
                                     lmd_p=lmd_p, dB=gain_in_dB, type=type)

        plt.plot(lmd_s_array, gain_array, 
                 label=f'$\\Delta\\alpha$={np.degrees(alpha_opt_detuned - alpha_opt):.2f}°,' 
                       f'$\\Delta\\theta$={np.degrees(theta_opt_detuned - theta_opt):.2f}°')

    plt.title(f'Parametric Gain for L={L*1e3:.1f} mm, $I_p$={I_p*1e-13:.1f} GW/cm²')
    plt.xlabel(r'$\lambda_s$ (nm)')
    plt.ylabel('Gain (dB)' if gain_in_dB else 'Gain (linear)')
    plt.grid()
    plt.legend()

    # make wavevector mismatch plot for optimal alpha and phase matching angle of center wavelength
    plt.sca(ax4)
    
    for angle_d in np.radians(angle_detuning_array):

        if detuning_mode == 'alpha':
            alpha_opt_detuned = alpha_opt + angle_d
            theta_opt_detuned = theta_opt
        elif detuning_mode == 'theta':
            alpha_opt_detuned = alpha_opt
            theta_opt_detuned = theta_opt + angle_d
        else:
            raise ValueError("detuning_mode must be 'alpha' or 'theta'")
        delta_k_array_opt = np.zeros_like(lmd_s_array)
        
        for i, lmd_s in enumerate(lmd_s_array):
            delta_k_array_opt[i] = compute_k_mismatch(theta_opt_detuned, lmd_s, alpha_opt_detuned, lmd_p=lmd_p, type=type)

        plt.plot(lmd_s_array, delta_k_array_opt*1e-3, 
                 label=f'$\\Delta\\alpha$={np.degrees(alpha_opt_detuned - alpha_opt):.2f}°,' 
                       f'$\\Delta\\theta$={np.degrees(theta_opt_detuned - theta_opt):.2f}°') # Convert to mm^-1
    
    plt.title(f'Wavevector Mismatch for L={L*1e3:.1f} mm, $I_p$={I_p*1e-13:.1f} GW/cm²')
    plt.xlabel(r'$\lambda_s$ (nm)')
    plt.ylabel(r'$\Delta k$ (mm$^{-1}$)')
    plt.grid()
    plt.legend()

    

    plt.tight_layout()
    plt.show()
    # plt.savefig(fname='phase_matching/plots/phase_matching_plots.png', dpi=300)