import LightwaveExplorer as lwe
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const

lwe_results_filename = r"C:\Users\juliu\Documents\VSCode Projects\OPA_tools\LWE_results\IR_reference_sim_05.txt"

if __name__=="__main__":

    result = lwe.load(lwe_results_filename)

    for i in range(5):

        fig, ax = plt.subplots()
        
        spec = result.spectrum_y[i]
        freqs = result.frequencyVectorSpectrum
        theta = result.batchVector[i]
        lmd = const.c / freqs * 1e9
        plt.plot(lmd, spec)
        plt.xlim(600, 1100)
        plt.title(f"$\\theta = {np.degrees(theta):.2f}$")
        plt.grid(axis='y')
        plt.show()