import numpy as np
import matplotlib.pyplot as plt
from tools import pulse_front_tilt_angle
from materials import n_CaF2, n_FS
from scipy.optimize import minimize_scalar


# Parameters
theta_apex = np.radians(60) # apex angle of the prism in radians
lmd_p_prism = 800 # pump wavelength in nm, do PFM before SHG
lmd_p_BBO = 800/2 # pump wavelength in nm in BBO
f1_telescope = 200e-3 # focal length of first telescope lens in m
f2_telescope = 50e-3 # focal length of second telescope lens in m
alpha_target = np.radians(3.7)  # target pulse front tilt angle in radians
theta = np.radians(31.2) # critical phase matching angle in radians
n_func = n_CaF2


def get_rotation_matrix(angle):
    """
    Returns a 2D rotation matrix for a given angle in radians.
    """
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])

def compute_triangle_points(apex, tilt, base_length=1.0, v_offset=0.0):
    """
    Computes the 2d points of a triangle given the apex point, tilt angle, base length, and vertical offset.
    """
    # compute un-tilded, with apex at origin
    A = np.array([0.0, 0.0])
    B = np.array([base_length * np.sin(apex/2), -base_length * np.cos(apex/2)])
    C = np.array([-base_length * np.sin(apex/2), -base_length * np.cos(apex/2)])

    # apply vertical offset
    A[1] += v_offset
    B[1] += v_offset
    C[1] += v_offset

    # rotate by tilt angle
    rotation_matrix = get_rotation_matrix(tilt)
    
    A = rotation_matrix @ A
    B = rotation_matrix @ B
    C = rotation_matrix @ C

    return A, B, C



if __name__ == "__main__":


    # find required incidence angle
    def objective(phi):
        alpha_tilt = pulse_front_tilt_angle(phi, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=False)
        return np.abs(alpha_tilt - alpha_target)
    
    
    # dirty way to find minimum incidence angle
    phi_test_array = np.linspace(0, np.pi/2, 1000)
    alpha_test_array = pulse_front_tilt_angle(phi_test_array, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=False)

    # phi_min: minimum incidence angle where alpha is valid (not NaN or infinite)
    phi_min = phi_test_array[np.where(np.isfinite(alpha_test_array))[0][0]]
    result = minimize_scalar(objective, bounds=(phi_min, np.pi/2), method='bounded')
    phi_opt = result.x

    result_tilt = pulse_front_tilt_angle(phi_opt, theta, n_func, theta_apex, f1_telescope, f2_telescope, lmd_p_prism=lmd_p_prism, lmd_p_crystal=lmd_p_BBO, full_return=True)

    for key, value in result_tilt.items():
        print(f"{key}: {np.degrees(value):.2f}")

    tilt_internal_deg = np.degrees(result_tilt["internal tilt"])
    
    # compute angle between incidence and exit beam
    prism_incidence = result_tilt["prism incidence angle"] # incidence angle normal to prism surface
    prism_exit = result_tilt["prism exit angle"] # exit angle normal to prism surface
    angle_tilt = - theta_apex / 2 + prism_exit

    # plot geometry
    plt.figure(figsize=(8, 6))

    # plot prism with equal aspect ratio
    plt.axis('equal')
    v_offset = 0.3
    # angle_tilt = 0
    base_length = 1.0
    A, B, C = compute_triangle_points(theta_apex, angle_tilt, base_length=base_length, v_offset=v_offset)
    plt.plot([A[0], B[0]], [A[1], B[1]], 'k-')
    plt.plot([A[0], C[0]], [A[1], C[1]], 'k-')
    plt.plot([B[0], C[0]], [B[1], C[1]], 'k-')

    # plot outgoing beam (horizontal at y=0)
    beam_length = 1
    mat = get_rotation_matrix(angle_tilt)

    # exit beam points
    P1 = A + 0.3 * (B - A)
    P2 = P1 + beam_length * np.array([1.0, 0.0])

    # entry beam point
    b0 = 0.3 * base_length # prism horizontal half-width at entry face

    # define coefficients for quadratic equation
    gamma = prism_exit - result_tilt["prism refraction angle 2"] # angle between beam in prism and beam exiting prism

    # getting length of beam inside prism numerically because I'm too dumb to do it analytically
    def objective(l):

        def objective_2(s):
            P3 = P1 + l * np.array([-np.cos(gamma), -np.sin(gamma)])
            P_closest = A + s * (C - A)
            return np.linalg.norm(P3 - P_closest)
        
        res = minimize_scalar(objective_2, bounds=(0, 1), method='bounded')
        s_opt = res.x
        P3 = P1 + l * np.array([-np.cos(gamma), -np.sin(gamma)])
        P_closest = A + s_opt * (C - A)
        return np.linalg.norm(P3 - P_closest)
    
    res = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
    l_opt = res.x
    P3 = P1 + l_opt * np.array([-np.cos(gamma), -np.sin(gamma)])

    beta = phi_opt - result_tilt["prism refraction angle 1"]
    angle_change = gamma + beta

    P4 = P3 + beam_length * np.array([-np.cos(angle_change), -np.sin(angle_change)])


    plt.scatter([A[0]], [A[1]], color='r')
    plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'b-')
    plt.plot([P3[0], P1[0]], [P3[1], P1[1]], 'b-')
    plt.plot([P4[0], P3[0]], [P4[1], P3[1]], 'b-')

    # plot incoming beam (at pi - angle_change to horizontal)
    incoming_angle = np.pi - angle_change

    plt.show()

