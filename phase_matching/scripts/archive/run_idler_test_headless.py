from idler_angle_test import test_angle_methods
import numpy as np

alpha_deg = 3.69
theta_deg = 31.21
lmd_p = 400
alpha = np.radians(alpha_deg)
theta = np.radians(theta_deg)

lmd_s = 650
omega_list, beta_list, mismatch_list = test_angle_methods(lmd_s, theta, alpha, lmd_p=lmd_p)

for i, (omega, beta, dk) in enumerate(zip(omega_list, beta_list, mismatch_list), start=1):
    print(f"Method {i}: omega (deg) = {np.degrees(omega):.6f}, beta (deg) = {np.degrees(beta):.6f}, delta_k (1/m) = {dk:.6e}")
