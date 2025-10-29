import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm


def standard_gaussian(x: np.ndarray, onedim: bool = False):

    """Returns a standard Gaussian function evaluated at x.

    Args:
        x (np.ndarray): Input array. See onedim parameter for shape.
        onedim (bool, optional): If True, x is treated as a single variable, possibly with batch dimensions.
                                 If False, x is treated as a multi-dimensional variable where the first axis represents dimensions.
                                 Defaults to False.
                                 This is so that the case of [x1,x2,x3,...] (1d gaussian sampled at n points) and [x1,y1,z1,...] (nd gaussian sampled at single point) can both be distinguished.

    Returns:
        np.ndarray: Standard Gaussian evaluated at x.
    """
    if onedim:
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    else:
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.sum(x**2, axis=0))
    

def gaussian_2d(x: np.ndarray|float, x0: np.ndarray|float, z0: np.ndarray|float, z: np.ndarray|float, sigma_x: float, sigma_z: float, gamma: float = 0.0) -> np.ndarray:

    """Returns a 2D Gaussian function evaluated at (x,z) with given standard deviations and rotation angle.

    Args:
        x (np.ndarray): x-coordinates where the Gaussian is evaluated.
        z (np.ndarray): z-coordinates where the Gaussian is evaluated.
        sigma_x (float): Standard deviation along the x-axis.
        sigma_z (float): Standard deviation along the z-axis.
        gamma (float): Rotation angle in radians.

    Returns:
        np.ndarray: 2D Gaussian evaluated at (x,z).
    """

    # compute normalized input vector
    x_vector = np.array([(x - x0), (z - z0)])
    rotation_mat = np.array([[np.cos(gamma), np.sin(gamma)],
                             [-np.sin(gamma), np.cos(gamma)]])
    x_rotated = np.tensordot(rotation_mat, x_vector, axes=1)
    x_standard = np.array([x_rotated[0] / sigma_x, x_rotated[1] / sigma_z])

    result = standard_gaussian(x_standard)

    return result


def propagating_pulse(x: np.ndarray|float, x0: np.ndarray|float, z: np.ndarray|float, z0: np.ndarray|float, sigma_x: float, sigma_z: float, t: float, v_g: float, gamma: float = 0.0, alpha: float = 0.0) -> np.ndarray:
    """
    Returns a 2D Gaussian pulse propagating at group velocity v_g at angle phi relative to the z-axis.
    
    Args:
        x (np.ndarray): x-coordinates where the Gaussian is evaluated.
        z (np.ndarray): z-coordinates where the Gaussian is evaluated.
        sigma_x (float): Standard deviation along the x-axis.
        sigma_z (float): Standard deviation along the z-axis.
        t (float): Time at which the pulse is evaluated.
        v_g (float): Group velocity of the pulse.
        gamma (float): Rotation angle of the Gaussian in radians.
        alpha (float): Propagation angle relative to the z-axis in radians.
    """

    # Compute the effective group velocity components
    v_g_z = v_g * np.cos(alpha)
    v_g_x = v_g * np.sin(alpha)

    # Compute the pulse shape at time t (t=0 is centered at (x0, z0))
    pulse = gaussian_2d(x, x0 + v_g_x * t, z, z0 + v_g_z * t, sigma_x, sigma_z, gamma)

    return pulse


def compute_overlap(pulse_1: np.ndarray, pulse_2: np.ndarray, dx: float, dz: float) -> float:
    """
    Computes the overlap integral between two 2D Gaussian pulses using np.trapezoid. Units are arbitrary.

    Args:
        pulse1 (np.ndarray): First pulse array. Assumed to be 2D.
        pulse2 (np.ndarray): Second pulse array. Assumed to be 2D and same shape as pulse1.

    Returns:
        float: Overlap integral value.
    """

    overlap = np.trapezoid(np.trapezoid(pulse_1 * pulse_2, axis=0, dx=dx), axis=0, dx=dz)
    return overlap


def overlap_function(pulse1_params: dict, pulse2_params: dict, x_range, z_range, t: np.ndarray) -> np.ndarray:
    """
    Computes the overlap integral between two propagating 2D Gaussian pulses over time, sampled at given time points.

    Args: 
        pulse1_params (dict): Parameters for the first pulse. Should contain keys 'x0', 'z0', 'sigma_x', 'sigma_z', 'v_g', 'theta', 'phi'; x0 and z0 refer to center at t=0.
        pulse2_params (dict): Parameters for the second pulse. Same keys as pulse1_params.
        x_range (tuple): (x_min, x_max) or (x_min, x_max, num_points) for x-axis sampling.
        z_range (tuple): (z_min, z_max) or (z_min, z_max, num_points) for z-axis sampling.
        t (np.ndarray): 1D array of time points at which to compute the overlap.

    Returns:
        np.ndarray: Overlap integral values at each time point. Same shape as t.
    """

    x = np.linspace(*x_range) if len(x_range) == 3 else np.linspace(x_range[0], x_range[1], 200)
    dx = x[1] - x[0]
    z = np.linspace(*z_range) if len(z_range) == 3 else np.linspace(z_range[0], z_range[1], 200)
    dz = z[1] - z[0]
    X, Z = np.meshgrid(x, z)
    overlap_values = np.zeros_like(t)
    
    def pulse_1(t):
        return propagating_pulse(X, pulse1_params['x0'], Z, pulse1_params['z0'],
                                 pulse1_params['sigma_x'], pulse1_params['sigma_z'],
                                 t, pulse1_params['v_g'],
                                 pulse1_params['gamma'],
                                 pulse1_params['alpha'])

    def pulse_2(t):
        return propagating_pulse(X, pulse2_params['x0'], Z, pulse2_params['z0'],
                                 pulse2_params['sigma_x'], pulse2_params['sigma_z'],
                                 t, pulse2_params['v_g'],
                                 pulse2_params['gamma'],
                                 pulse2_params['alpha'])

    for i, time in tqdm(enumerate(t), desc="Computing overlap", total=len(t)):
        p1 = pulse_1(time)
        p2 = pulse_2(time)
        overlap_values[i] = compute_overlap(p1, p2, dx, dz)
    return overlap_values


# Test the Gaussian functions with interactive plots
if __name__ == "__main__":
    x = np.linspace(-5, 5, 200)
    z = np.linspace(-5, 5, 200)
    X, Z = np.meshgrid(x, z)

    sigma_x = 1.0
    sigma_z = 2.0
    theta = np.radians(30)  # Rotate by 30 degrees
    phi = np.radians(15)    # Propagate at 15 degrees to z-axis

    # Test propagating pulse with interactive slider for time
    # Select theta and phi using input fields
    fig, ax = plt.subplots(figsize=(7, 9))
    plt.subplots_adjust(bottom=0.4)
    t0 = 0.0
    v_g = 1.0
    pulse_data = propagating_pulse(X, 0.0, Z, 0.0, sigma_x, sigma_z, t0, v_g, theta, alpha=phi)
    pulse_plot = ax.imshow(pulse_data, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis')
    ax.set_title('Propagating 2D Gaussian Pulse')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.axis('equal')
    cbar = plt.colorbar(pulse_plot)
    cbar.set_label('Amplitude')
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(ax_time, 'Time', -5.0, 5.0, valinit=t0)

    # add interactive angle selection
    ax_theta = plt.axes([0.25, 0.15, 0.65, 0.03])
    theta_slider = Slider(ax_theta, 'Theta', 0.0, 90.0, valinit=np.degrees(theta))
    ax_phi = plt.axes([0.25, 0.2, 0.65, 0.03])
    phi_slider = Slider(ax_phi, 'Phi', 0.0, 90.0, valinit=np.degrees(phi))

    def update(val):
        t = time_slider.val
        theta = np.radians(theta_slider.val)
        phi = np.radians(phi_slider.val)
        pulse_data = propagating_pulse(X, 0.0, Z, 0.0, sigma_x, sigma_z, t, v_g, theta, alpha=phi)
        pulse_plot.set_data(pulse_data)
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    theta_slider.on_changed(update)
    phi_slider.on_changed(update)
    plt.show()