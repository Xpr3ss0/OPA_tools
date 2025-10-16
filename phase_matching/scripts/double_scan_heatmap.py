import numpy as np
import matplotlib.pyplot as plt
from tools import phase_matching_array, optimize_alpha, OPA_gain, compute_k_mismatch
import LightwaveExplorer as lwe
from scipy import constants as const
from tools import OPA_gain


# this script plots the gain from a lwe simulation for a batch of simulations
# it assumes that batch 1 is over the crystal length starting with 0
# batch two is over any other parameter


lwe_results_filename = r"D:\VSCode Projects\OPA_tools\LWE_results\large_theta_alpha_scan_06\large_theta_alpha_scan_06.txt"


def band_total_power(spectrum, freqVector, band=(430, 570), log=True):

    filter = (freqVector > band[0]*1e12) & (freqVector < band[1]*1e12)
    power = np.sum(spectrum[filter])
    return np.log(power) if log else power

def band_avg_power_density(spectrum, freqVector, band=(430, 570)):

    filter = (freqVector > band[0]*1e12) & (freqVector < band[1]*1e12)
    powerDensity = np.mean(spectrum[filter])
    return powerDensity

def recip_band_std(spectrum, freqVector, band=(430, 570)):

    filter = (freqVector > band[0]*1e12) & (freqVector < band[1]*1e12)
    std = np.std(spectrum[filter])
    return std

def recip_rel_std(spectrum, freqVector, band=(430, 570)):

    filter = (freqVector > band[0]*1e12) & (freqVector < band[1]*1e12)
    mean = np.mean(spectrum[filter])
    std = np.std(spectrum[filter])
    return mean**2 / std



map_func = band_total_power
args = {'band': (const.c / 680e-9 * 1e-12, const.c / 530e-9 * 1e-12)}
title = 'log total power (a.u.) in 530nm - 680nm'

if __name__=="__main__":


    # load gain from lwe simulation
    results = lwe.load(lwe_results_filename)

    param_batch_1 = results.batchVector
    param_batch_2 = results.batchVector2

    # create heta map
    heat_map = np.zeros((len(param_batch_2), len(param_batch_1)))

    signal_freq = results.frequencyVectorSpectrum
    signal_lmd = const.c / signal_freq
    lmd_filter = (signal_lmd < 3000e-9) & (signal_lmd > 300e-9)

    for i, param_value_2 in enumerate(param_batch_2):
        for j, param_value_1 in enumerate(param_batch_1):


            # get signal spectrum
            signal_spectrum = results.spectrum_y[i, j]
            
            # compute map value using specified function
            power = map_func(signal_spectrum, signal_freq, **args)
            heat_map[i, j] = power

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    heatmap_plot = ax1.imshow(heat_map)
    ax1.set_xticks(np.arange(len(param_batch_1)), labels=[f"{np.degrees(v):.1f}°" for v in param_batch_1])
    ax1.set_yticks(np.arange(len(param_batch_2)), labels=[f"{np.degrees(v):.1f}°" for v in param_batch_2])
    ax1.set_xlabel(r'Crystal Angle $\theta$ (degrees)')
    ax1.set_ylabel(r'Propagation Angle $\alpha$ (degrees)')
    ax1.set_title(title)
    ax2.set_title("Spectral Power Density")
    ax2.set_xlabel("$\\nu$ (THz)    (press b to toggle scale)")
    ax2.set_ylabel("$S$ ( J / THz )")
    ax2.set_title("Spectral Power Density")
    plt.grid(True)

    highlights = []
    lines = []
    selected_indices = []
    x_scale = 'frequency'

    def on_button_press(event):

        if event.key == "b": 
            global x_scale
            global lines

            if x_scale == 'frequency':
                x_scale = 'wavelength'
                for line in lines:
                    xdata = signal_lmd[lmd_filter] * 1e9
                    line.set_xdata(xdata)
                    ax2.set_xlim(xdata.min(), xdata.max())
                    ax2.set_xlabel("$\\lambda$ (nm)    (press b to toggle scale)")
            else:
                x_scale = 'frequency'
                for line in lines:
                    xdata = signal_freq[lmd_filter] * 1e-12
                    line.set_xdata(xdata)
                    ax2.set_xlim(xdata.min(), xdata.max())
                    ax2.set_xlabel("$\\nu$ (THz)    (press b to toggle scale)")

            ax2.figure.canvas.draw()    

    def add_plot(index):

        global highlights
        global lines
        global x_scale

        x, y = index

        # get xdata
        if x_scale == "frequency":
            xdata = signal_freq[lmd_filter] * 1e-12
        else:
            xdata = signal_lmd[lmd_filter] * 1e9

        # find corresponding batch parameter indices
        param_x = param_batch_1[x]
        param_y = param_batch_2[y]

        new_highlight = ax1.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                                                edgecolor='red', facecolor='none', lw=2))
        highlights.append(new_highlight)
        selected_spectrum = results.spectrum_y[y, x][lmd_filter]
        line,  = ax2.plot(xdata, selected_spectrum*1e12, label=f"$\\theta={np.degrees(param_x):.2f}$, $\\alpha={np.degrees(param_y):.2f}$")
        lines.append(line)
        ax2.legend()


    def on_click(event):
        global highlights
        global lines
        global selected_indices

        if event.inaxes == ax1:
            x, y = int(round(event.xdata)), int(round(event.ydata))

            ctrl_pressed = event.key == 'control'

            # plotting logic depends on ctrl
            if not ctrl_pressed:

                # set selected indices to only the new one
                selected_indices = [(x, y)]

                # remove previous highlights and lines from plot
                for i in range(len(highlights)):
                    highlights.pop(0).remove()
                    lines.pop(0).remove()

                # add new line and highlight
                add_plot((x, y))
            else:

                # if already plotted, remove
                if (x, y) in selected_indices:
                    loc = selected_indices.index((x, y))
                    lines.pop(loc).remove()
                    highlights.pop(loc).remove()
                    selected_indices.pop(loc)
                # if not plotted, add plot
                else:
                    # add new line and highlight
                    add_plot((x, y))
                    selected_indices.append((x, y))

            ax1.figure.canvas.draw()
            ax2.figure.canvas.draw()

    plt.tight_layout()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_button_press)

    plt.show()



