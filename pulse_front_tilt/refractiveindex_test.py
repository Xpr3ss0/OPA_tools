import numpy as np
from refractiveindex import RefractiveIndexMaterial


SF11 = RefractiveIndexMaterial(shelf="specs", book="SCHOTT-optical", page="N-FK58")

print(SF11.get_refractive_index(589))  # Calculate refractive index at wavelength 0.589 micrometers