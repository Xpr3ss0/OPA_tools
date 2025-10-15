import numpy as np
import matplotlib.pyplot as plt
from tools import phase_matching_array, optimize_alpha, OPA_gain, compute_k_mismatch
import LightwaveExplorer as lwe
from scipy import constants as const
from tqdm import tqdm
from tools import OPA_gain


# this script plots the gain from a lwe simulation for a batch of simulations
# it assumes that batch 1 is over the crystal length starting with 0
# batch two is over any other parameter


lwe_results_filename = r"C:\Users\juliu\Documents\VSCode Projects\OPA_tools\LWE_results\large_theta_alpha_scan_03.txt"


def band_total_power(spectrum, freqVector, band=(430, 570), log=True):

    filter = (freqVector > band[0]*1e12) & (freqVector < band[1]*1e12)
    power = np.sum(spectrum[filter])
    return np.log(power) if log else power

def band_avg_power_density(spectrum, freqVector, band=(430, 570)):

    filter = (freqVector > band[0]*1e12) & (freqVector < band[1]*1e12)
    powerDensity = np.mean(spectrum[filter])
    return powerDensity

def band_std(spectrum, freqVector, band=(430, 570)):

    filter = (freqVector > band[0]*1e12) & (freqVector < band[1]*1e12)
    std = np.std(spectrum[filter])
    return std



map_func = band_total_power
args = {'band': (430, 570), 'log': True}
title = 'Log Total Power (a.u.) in 430-570 THz band'

if __name__=="__main__":


    # load gain from lwe simulation
    results = lwe.load(lwe_results_filename)

    param_batch_1 = results.batchVector
    param_batch_2 = results.batchVector2

    heat_map = np.zeros((len(param_batch_2), len(param_batch_1)))
    CW_map = np.zeros((len(param_batch_2), len(param_batch_1)))

    for i, param_value_2 in enumerate(param_batch_2):
        for j, param_value_1 in enumerate(param_batch_1):


            # get signal spectrum
            signal_spectrum = results.spectrum_y[i, j]
            signal_freq = results.frequencyVectorSpectrum

            # compute map value using specified function
            power = map_func(signal_spectrum, signal_freq, **args)
            heat_map[i, j] = power

    fig, ax = plt.subplots()
    im = ax.imshow(heat_map)
    ax.set_xticks(np.arange(len(param_batch_1)), labels=[f"{np.degrees(v):.1f}°" for v in param_batch_1])
    ax.set_yticks(np.arange(len(param_batch_2)), labels=[f"{np.degrees(v):.1f}°" for v in param_batch_2])
    ax.set_xlabel(r'Crystal Angle $\theta$ (degrees)')
    ax.set_ylabel(r'Propagation Angle $\alpha$ (degrees)')
    plt.title(title)
    plt.tight_layout()

    plt.show()



