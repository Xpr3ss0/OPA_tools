import numpy as np
import matplotlib.pyplot as plt


def grating_angles(lmd, d, theta_i):
    """
    Calculate the 1st order diffraction angle and resulting PFT angle for a transmission grating using the grating equation.
    Args:
        lmd (float or array): Wavelength in nm
        d (float): Grating spacing in mm (1 / lines per mm)
        theta_i (float): Incident angle in radians

    Returns:
        float or array: Diffraction angle in radians
    """

    # Convert wavelength from nm to m
    lmd_m = lmd * 1e-9
    d_m = d * 1e-3

    # Calculate diffraction angle using the grating equation
    theta_d = np.arcsin(np.sin(theta_i) - (lmd_m / d_m))

    ang_disp = 1 / (d_m * np.cos(theta_d))  # angular dispersion in rad/m

    pft_angle = np.arctan(ang_disp * lmd_m)  # PFT angle in radians

    return theta_d, pft_angle, ang_disp


if __name__ == "__main__":


    ##############
    # PARAMETERS #
    ##############
    lmd = 800  # wavelength in nm
    groove_densities = [100, 200, 300]  # lines per mm
    theta_i_range = (0, 60) # incident angle range in degrees


    # set up grid
    theta_i_array = np.radians(np.linspace(theta_i_range[0], theta_i_range[1], 100))  # incident angle array in radians

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    fig3, ax5 = plt.subplots(1, 1, figsize=(6, 5))

    for groove_density in groove_densities:
        d = 1 / groove_density  # grating spacing in mm

        theta_d_array, pft_angle_array, ang_disp_array = grating_angles(lmd, d, theta_i_array)
        theta_deflect_array = theta_d_array - theta_i_array  # deflection angle in radians
        d_deflect_d_theta_i = np.gradient(theta_deflect_array, theta_i_array)  # derivative of deflection angle with respect to incident angle

        plt.sca(ax1)
        plt.plot(np.degrees(theta_i_array), np.degrees(theta_d_array), label=f'{groove_density} lines/mm')

        plt.sca(ax3)
        plt.plot(np.degrees(theta_i_array), np.degrees(theta_deflect_array), label=f'{groove_density} lines/mm')

        plt.sca(ax4)
        plt.plot(np.degrees(theta_i_array), d_deflect_d_theta_i, label=f'{groove_density} lines/mm')

        plt.sca(ax2)
        plt.plot(np.degrees(theta_i_array), np.degrees(pft_angle_array), label=f'{groove_density} lines/mm')

        plt.sca(ax5)
        plt.plot(np.degrees(theta_i_array), np.degrees(ang_disp_array * 1e-9), label=f'{groove_density} lines/mm')

    plt.sca(ax1)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Diffraction Angle (degrees)')
    plt.title('Diffraction Angles vs Incident Angle')
    plt.grid()
    plt.legend()

    plt.sca(ax2)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Pulse Front Tilt Angle (degrees)')
    plt.title('Pulse Front Tilt Angle vs Incident Angle')
    plt.grid()
    plt.legend()

    plt.sca(ax3)
    plt.title('Deflection Angle vs Incident Angle')
    plt.grid()
    plt.legend()
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Deflection Angle (degrees)')

    plt.sca(ax4)
    plt.title('Derivative of Deflection Angle vs Incident Angle')
    plt.grid()
    plt.legend()
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Derivative (deg/deg)')

    plt.sca(ax5)
    plt.title('Angular Dispersion vs Incident Angle')
    plt.grid()
    plt.legend()
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Angular Dispersion (deg/nm)')


    fig.suptitle(f'Grating Diffraction and Pulse Front Tilt for {lmd} nm Wavelength')
    plt.tight_layout()
    plt.show()

