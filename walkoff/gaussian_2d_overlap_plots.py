from tools import overlap_function, propagating_pulse
import numpy as np
import matplotlib.pyplot as plt
from materials import v_g_BBO
from scipy import constants as const

# Define parameters for two pulses

lmd_p = 400  # nm
lmd_s = 700  # nm
theta = np.radians(31) # angle to optical axis for extraordinary index
alpha = np.radians(3.7)  # propagation angle relative to z-axis
tilt_angle_p = 0
tilt_angle_s = alpha

v_g_p = v_g_BBO(lmd_p, extraordinary=True, theta=theta) # m/s
v_g_s = v_g_BBO(lmd_s, extraordinary=False) # m/s

# pulse durations
tau_p = 150e-15  # s
tau_s = 150e-15  # s

# convert to spatial widths (propagation along z)
sigma_z_p = v_g_p * tau_p
sigma_z_s = v_g_s * tau_s

# pulse widths in x
sigma_x_p = 5e-3  # in m
sigma_x_s = 5e-3  # in m

# print parameters
c = const.c
print(f"Pump v_g/c: {v_g_p/c:.3f} , sigma_z: {sigma_z_p*1e6:.2f} um")
print(f"Signal v_g/c: {v_g_s/c:.3f} , sigma_z: {sigma_z_s*1e6:.2f} um")

pulse_1_params = {"x0": 0.0, "z0": 0.0, "sigma_x": sigma_x_p, "sigma_z": sigma_z_p, "v_g": v_g_p, "gamma": tilt_angle_p, "alpha": 0}
pulse_2_params = {"x0": 0.0, "z0": 0.0, "sigma_x": sigma_x_s, "sigma_z": sigma_z_s, "v_g": v_g_s, "gamma": tilt_angle_s, "alpha": alpha}

x_range = (-20*sigma_x_p, 20*sigma_x_p, 1000) # x_min, x_max, num_points
z_range = (-5e-3, 5e-3, 1000) # z_min, z_max, num_points
t = np.linspace(z_range[0] / v_g_p / 2, z_range[1] / v_g_p / 2, 100)

if __name__ == "__main__":

    # compute overlap values
    overlap_values = overlap_function(pulse_1_params, pulse_2_params, x_range, z_range, t)

    # compute fwhm of overlap values
    half_max = np.max(overlap_values) / 2
    fwhm_indices = np.where(overlap_values >= half_max)[0].min(), np.where(overlap_values >= half_max)[0].max()
    t_fwhm = t[fwhm_indices[1]] - t[fwhm_indices[0]]

    # compute corresponding z positions
    z_positions_p = t * v_g_p
    l_split_p = z_positions_p[fwhm_indices[1]] - z_positions_p[fwhm_indices[0]]
    z_positions_s = t * v_g_s * np.cos(alpha)
    l_split_s = z_positions_s[fwhm_indices[1]] - z_positions_s[fwhm_indices[0]]

    # print fwhm results
    print(f"Temporal FWHM of overlap: {t_fwhm*1e12:.2f} ps")
    print(f"Spatial FWHM of overlap (pump frame): {l_split_p*1e3:.2f} mm")
    print(f"Spatial FWHM of overlap (signal frame): {l_split_s*1e3:.2f} mm")

    # compute sum and product of pulses at t=t[0], t=0, t=t[-1] for visualization
    x = np.linspace(*x_range)
    z = np.linspace(*z_range)
    X, Z = np.meshgrid(x, z)

    def pulse_1(t):
        return propagating_pulse(X, pulse_1_params['x0'], Z, pulse_1_params['z0'],
                                 pulse_1_params['sigma_x'], pulse_1_params['sigma_z'],
                                 t, pulse_1_params['v_g'],
                                 pulse_1_params['gamma'],
                                 pulse_1_params['alpha'])
    def pulse_2(t):
        return propagating_pulse(X, pulse_2_params['x0'], Z, pulse_2_params['z0'],
                                 pulse_2_params['sigma_x'], pulse_2_params['sigma_z'],
                                 t, pulse_2_params['v_g'],
                                 pulse_2_params['gamma'],
                                 pulse_2_params['alpha'])

    pulse_sum_tstart = pulse_1(t[0]) + pulse_2(t[0])
    pulse_sum_tmid = pulse_1(0.0) + pulse_2(0.0)
    pulse_sum_tend = pulse_1(t[-1]) + pulse_2(t[-1])

    pulse_prod_tstart = pulse_1(t[0]) * pulse_2(t[0])
    pulse_prod_tmid = pulse_1(0.0) * pulse_2(0.0)
    pulse_prod_tend = pulse_1(t[-1]) * pulse_2(t[-1])

    # plot overlap values and fwhm
    fig, axs = plt.subplots(nrows=3, figsize=(6, 10))
    axs[0].plot(z_positions_p*1e3, overlap_values)
    axs[0].plot([z_positions_p[fwhm_indices[0]]*1e3, z_positions_p[fwhm_indices[1]]*1e3], [half_max, half_max], 'r--', label=f"FWHM = {l_split_p*1e3:.2f} mm")
    axs[1].plot(z_positions_s*1e3, overlap_values)
    axs[1].plot([z_positions_s[fwhm_indices[0]]*1e3, z_positions_s[fwhm_indices[1]]*1e3], [half_max, half_max], 'r--', label=f"FWHM = {l_split_s*1e3:.2f} mm")
    axs[2].plot(t*1e12, overlap_values)
    axs[2].plot([t[fwhm_indices[0]]*1e12, t[fwhm_indices[1]]*1e12], [half_max, half_max], 'r--', label=f"FWHM = {t_fwhm*1e12:.2f} ps")
    axs[0].set_xlabel("Pump Position (mm)")
    axs[1].set_xlabel("Signal Position (mm)")
    axs[2].set_xlabel("Time (ps)")
    plt.suptitle("Overlap Integral Between Two 2D Gaussian Pulses Over Time")

    for ax in axs:
        ax.set_ylabel("Overlap Integral")
        ax.grid()

    plt.tight_layout()

    # plot pulse sums
    fig, (axs_sums, axs_prods) = plt.subplots(2, 3, figsize=(8, 8), sharex=True, sharey=True)
    im1 = axs_sums[0].imshow(pulse_sum_tstart, extent=(x_range[0]*1e3, x_range[1]*1e3, z_range[0]*1e3, z_range[1]*1e3), origin='lower', cmap='viridis', interpolation='nearest')
    axs_sums[0].set_title(f"t={t[0]*1e12:.2f} ps\n")
    im2 = axs_sums[1].imshow(pulse_sum_tmid, extent=(x_range[0]*1e3, x_range[1]*1e3, z_range[0]*1e3, z_range[1]*1e3), origin='lower', cmap='viridis', interpolation='nearest')
    axs_sums[1].set_title(f"t=0 ps\nSum of Amplitudes")
    im3 = axs_sums[2].imshow(pulse_sum_tend, extent=(x_range[0]*1e3, x_range[1]*1e3, z_range[0]*1e3, z_range[1]*1e3), origin='lower', cmap='viridis', interpolation='nearest')
    axs_sums[2].set_title(f"t={t[-1]*1e12:.2f} ps\n")

    plt.suptitle(f"$\\alpha={np.degrees(alpha):.1f}$, $\\theta={np.degrees(theta):.1f}$ deg")

    for ax in axs_sums:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")


    # plot pulse products, using the same color scale for comparison, using log scale
    pmax = max(pulse_prod_tstart.max(), pulse_prod_tmid.max(), pulse_prod_tend.max())
    pmin = min(pulse_prod_tstart[pulse_prod_tstart>0].min(), pulse_prod_tmid[pulse_prod_tmid>0].min(), pulse_prod_tend[pulse_prod_tend>0].min())
    im4 = axs_prods[0].imshow(pulse_prod_tstart, extent=(x_range[0]*1e3, x_range[1]*1e3, z_range[0]*1e3, z_range[1]*1e3), origin='lower', cmap='plasma', vmin=pmin, vmax=pmax, interpolation='nearest')
    im5 = axs_prods[1].imshow(pulse_prod_tmid, extent=(x_range[0]*1e3, x_range[1]*1e3, z_range[0]*1e3, z_range[1]*1e3), origin='lower', cmap='plasma', vmin=pmin, vmax=pmax, interpolation='nearest')
    im6 = axs_prods[2].imshow(pulse_prod_tend, extent=(x_range[0]*1e3, x_range[1]*1e3, z_range[0]*1e3, z_range[1]*1e3), origin='lower', cmap='plasma', vmin=pmin, vmax=pmax, interpolation='nearest')

    axs_prods[1].set_title(f"Product of Amplitudes")

    for ax in axs_prods:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        
    plt.tight_layout()
    plt.show()