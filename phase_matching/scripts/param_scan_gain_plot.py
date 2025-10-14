import numpy as np
import matplotlib.pyplot as plt
from tools import phase_matching_array, optimize_alpha, OPA_gain, compute_k_mismatch
import LightwaveExplorer as lwe
from scipy import constants as const


# this script plots the gain from a lwe simulation for a batch of simulations
# it assumes that batch 1 is over the crystal length starting with 0
# batch two is over any other parameter


lwe_results_filename = r"C:\Users\juliu\Documents\VSCode Projects\OPA_tools\LWE_results\NOPA_NC_scan_08.txt"


# this function builds the legend string, taking all relevant parameters from the lwe result, except the scanned one
def legend_formatter(lwe_result, index, scan_param='propagationAngle2'):
    
    # allow for .2f formatting
    format_dict = {
        'crystalTheta': [r'$\theta=%.1f\degree$', np.degrees],
        'propagationAngle2': [r'$\alpha=%.1f\degree$', np.degrees],
    }

    legend_str = ""
    for attr in ['crystalTheta', 'propagationAngle2']:
        if attr == scan_param:
            continue
        legend_str += format_dict[attr][0] % format_dict[attr][1](getattr(lwe_result, attr)) + ' '

    legend_str += format_dict[scan_param][0] % format_dict[scan_param][1](lwe_result.batchVector2[index])

    return legend_str

def plot_references(pumpFreq, mode='nu'):
    
    degenFreq = pumpFreq / 2
    
    if mode=='nu':
        plt.axvline(degenFreq*1e-12, label='degeneracy', ls='--')
        plt.axvline(400, label='target region', color='black')
        plt.axvline(700, color='black')
    elif mode=='lmd':
        degenLmd = const.c / degenFreq
        plt.axvline(degenLmd*1e6, label='degeneracy', ls='--')
        plt.axvline(const.c / 400e12 * 1e6, label='target region', color='black')
        plt.axvline(const.c / 700e12 * 1e6, color='black')


if __name__=="__main__":


    # load gain from lwe simulation
    results = lwe.load(lwe_results_filename)

    param_batch = results.batchVector2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

    for i, param_value in enumerate(param_batch):
        spec_final = results.spectrum_y[i, -1]
        spec_0 = results.spectrum_y[i, 0]
        freqs = results.frequencyVectorSpectrum

        # filter for better automatic plot scaling
        filter = (freqs < 800e12) & (freqs > 80e12) 

        gain = spec_final / spec_0
        lmd = const.c / freqs * 1e9

        # load relevant parameters from lwe results
        pulseEmergy = results.pulseEnergy1
        pulseBandwidth = results.bandwidth1
        pulseWaist = results.beamwaist1
        pumpFreq = results.frequency1

        # compute pump intensity
        pulseArea = np.pi * pulseWaist**2
        TBP = 0.441
        tau_TF = TBP / pulseBandwidth
        P_p = pulseEmergy / tau_TF
        I_p = P_p / pulseArea
    
        # assume batch mode scanning over max z, from 0 to end
        z_list = results.batchVector
        length = z_list[-1]

        legend_str = legend_formatter(results, i, scan_param='propagationAngle2')
        ax1.plot(lmd[filter]*1e-3, gain[filter], label=legend_str)
        ax2.plot(freqs[filter]*1e-12, gain[filter], label=legend_str)
        ax3.plot(lmd[filter]*1e-3, spec_final[filter]*1e12, label=legend_str)
        ax4.plot(freqs[filter]*1e-12, spec_final[filter]*1e12, label=legend_str)

    plt.suptitle(f"$I_p={I_p*1e-13:.2f}$ GW/cm$^2$, L={length*1e3:.2f}mm", fontsize=15)
    plt.sca(ax1)
    plot_references(pumpFreq, mode='lmd')
    plt.xlabel(r'$\lambda$ ($\mathrm{\mu m}$)')
    plt.ylabel(r'gain')
    plt.legend(loc='upper right')
    plt.yscale('log')

    plt.sca(ax2)
    plot_references(pumpFreq, mode='nu')
    plt.xlabel(r"$\nu$ (THz)")
    plt.ylabel(r'gain')
    plt.legend(loc='upper left')
    plt.yscale('log')

    plt.sca(ax3)
    plot_references(pumpFreq, mode='lmd')
    plt.xlabel(r'$\lambda$ ($\mathrm{\mu m}$)')
    plt.ylabel(r"S (J/THz)")
    plt.legend(loc='upper right')

    plt.sca(ax4)
    plot_references(pumpFreq, mode='nu')
    plt.xlabel(r"$\nu$ (THz)")
    plt.ylabel(r"S (J/THz)")
    plt.legend(loc='upper left')

    plt.tight_layout()

    plt.show()
