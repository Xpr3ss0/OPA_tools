import numpy as np
import matplotlib.pyplot as plt
from tools import phase_matching_array, optimize_alpha, OPA_gain, compute_k_mismatch
import LightwaveExplorer as lwe
from scipy import constants as const

lwe_results_filename = r"D:\VSCode Projects\OPA_tools\LWE_sims\result_files\NOPA_results\NOPA_results.txt"

if __name__=="__main__":


    # load gain from lwe simulation
    results = lwe.load(lwe_results_filename)
    spec_final = results.spectrum_y[-1]
    spec_0 = results.spectrum_y[0]
    freqs = results.frequencyVectorSpectrum
    gain = spec_final / spec_0
    lmd = const.c / freqs * 1e9

    pulseEmergy = results.pulseEnergy1
    pulseBandwidth = results.bandwidth1
    pulseWaist = results.beamwaist1
    pulseArea = np.pi * pulseWaist**2
    TBP = 0.441
    tau_TF = TBP / pulseBandwidth
    P_p = pulseEmergy / tau_TF
    I_p = P_p / pulseArea

    print(f"{I_p:.2e}")

    # compute gain using linear theory
    theta = results.crystalTheta
    alpha = results.propagationAngle2

    gain_linear = np.zeros_like(lmd)
    for i, lmd_s in enumerate(lmd):
        gain_linear[i] = OPA_gain(theta=theta, lmd_s=lmd_s, alpha=alpha, L=1e-3, I_p=I_p, 
                                    lmd_p=400, dB=False)

    plt.plot(lmd, spec_final / spec_0, label='lwe')
    plt.plot(lmd, gain_linear)
    plt.title(f"$\\theta={np.degrees(theta):.1f}\\degree$, $\\alpha = {np.degrees(alpha):.1f}\\degree$, $I_p={I_p*1e-13:.2f}$ GW/cm$^2$")
    plt.xlim(300, 1000)
    plt.legend()
    plt.yscale('log')
    plt.show()
