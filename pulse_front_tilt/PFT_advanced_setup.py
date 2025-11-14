import numpy as np
from tools import PFT_change_interface, PFT_prism, PFT_telescope
from materials import n_BBO

alpha_deg = 3.7  # non-collinear angle in degrees
theta_deg = 31.2 # critical phasematching angle


# convert all parameters here
alpha = np.radians(alpha_deg)  # non-collinear angle in radians
theta = np.radians(theta_deg)  # critical phasematching angle in radians


# compute required external pulse front tilt angle
pft_diff = PFT_change_interface(gamma_int=alpha, theta=theta)
gamma_ext = alpha + pft_diff


results = {"alpha_deg": alpha_deg,
           "theta_deg": theta_deg,
           "n_BBO": f"{n_BBO(400):.3f}",
           "gamma_ext_deg": f"{np.degrees(gamma_ext):.2f}"}

print("\n\nAdvanced Pulse Front Tilt Setup Results:\n")
for key, value in results.items():
    print(f"{key}: {value}")

# test stuff
gamma_int_test = PFT_change_interface(gamma_ext=np.radians(6.15), theta=theta, return_diff=False)

print(f"\nTest gamma_int: {np.degrees(gamma_int_test):.2f} deg (should be close to {alpha_deg} deg)\n")