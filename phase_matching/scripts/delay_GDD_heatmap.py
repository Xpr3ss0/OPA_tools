import numpy as np
import matplotlib.pyplot as plt
from tools import phase_matching_array, optimize_alpha, OPA_gain, compute_k_mismatch, minimize_k_mismatch
import LightwaveExplorer as lwe
from scipy import constants as const
from tools import OPA_gain
from tqdm import tqdm


# this script plots the gain from a lwe simulations for a batch of simulations
# it assumes that batch 1 is over the crystal length starting with 0
# batch two is over any other parameter
# should not load more than 4 datasets at once (loading time and plotting space)

data_sets = [1]

lwe_results_filenames = [r"LWE_results/delay_GDD_scans/revised_parameters\delay_GDD_scan_0{}.txt".format(data_set) for data_set in data_sets]

#####################
# Utility functions #
#####################

def band_total_power(spectrum, freqVector, band=(430, 570), log=True):
    """Computes total power in a given frequency band (THz). Returns log(power) if log=True."""

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

def gaussian_profile(freq, central_frequency, bandwidth, pulse_energy=1):
    """Generates a Gaussian profile in frequency domain. Amplitude is computed from pulse energy."""

    # compute amplitude from power (area under curve)
    amplitude = pulse_energy / (bandwidth * np.sqrt(2 * np.pi))
    return amplitude * np.exp(-0.5 * ((freq - central_frequency) / bandwidth) ** 2)

#######################
# Configuration setup #
#######################

map_func = band_total_power
use_gain = False # if true, map_func is applied to gain spectrum instead of output spectrum
band = (400, 600) # frequency band in THz for map function
title = f'log total power (a.u.) in {band[0]:.0f}THz - {band[1]:.0f}THz'

if __name__=="__main__":


    #######################
    # Loading LWE results #
    #######################
    print("Loading LWE results...")
    results = [lwe.load(lwe_results_filename) for lwe_results_filename in tqdm(lwe_results_filenames)]

    param_batches_1 = [result.batchVector for result in results]
    param_batches_2 = [result.batchVector2 for result in results]

    # load spectral seed profile
    seed_bandwidths = [result.bandwidth2 for result in results]
    seed_freqs = [result.frequency2 for result in results]
    seed_energies = [result.pulseEnergy2 for result in results]
    pump_energies = [result.pulseEnergy1 for result in results]
    pump_focal_area = [np.pi * result.beamwaist1**2 for result in results]
    pump_bandwidths = [result.bandwidth1 for result in results]
    pump_FL_durations = [0.44 / bandwidth for bandwidth in pump_bandwidths] # assuming Gaussian pulses
    pump_I_p = [energy / (duration * area) * 1e-13 for energy, duration, area in zip(pump_energies, pump_FL_durations, pump_focal_area)] # in GW/cm^2

    # print loaded parameters
    for i in range(len(results)):
        print(f"\nLoaded LWE results from {lwe_results_filenames[i]}\n"
              f"Signal center frequency: {seed_freqs[i]*1e-12:.2f} THz\n"
              f"Signal bandwidth: {seed_bandwidths[i]*1e-12:.2f} THz\n"
              f"Pump energy: {pump_energies[i]*1e6:.2f} uJ\n"
              f"Pump intensity: {pump_I_p[i]:.2f} GW/cm^2\n")


    ######################
    # Creating heat maps #
    ######################

    heat_maps = []

    # initialize empty heat maps
    for i in range(len(results)):
        heat_map = np.zeros((len(param_batches_2[i]), len(param_batches_1[i])))
        heat_maps.append(heat_map)

    signal_freqs = [result.frequencyVectorSpectrum for result in results]
    with np.errstate(divide='ignore'):
        signal_lmds = [const.c / freq for freq in signal_freqs]
    lmd_filters = [(lmd < 3000e-9) & (lmd > 300e-9) for lmd in signal_lmds]

    # compute heat map values
    print("Computing heat maps...")
    for k in tqdm(range(len(results))):
        for i, param_value_2 in enumerate(param_batches_2[k]): # delay
            for j, param_value_1 in enumerate(param_batches_1[k]): # GDD

                # get signal spectrum
                signal_spectrum = results[k].spectrum_y[i, j]

                if use_gain:
                    # compute gain spectrum
                    input_spectrum = gaussian_profile(signal_freqs[k], seed_freqs[k], seed_bandwidths[k], seed_energies[k])
                    signal_spectrum = signal_spectrum / input_spectrum
                
                # compute map value using specified function
                power = map_func(signal_spectrum, signal_freqs[k], band=band)
                heat_maps[k][i, j] = power

                
    ################
    # Set up plots #
    ################

    # set up backend and figure
    plt.switch_backend('QT5Agg')
    fig, axs = plt.subplot_mosaic([[f'map_{i}', f'map_{i+1}' if i+1 < len(results) else f'map_{i}', 'selected_plot', 'selected_plot'] 
                                   for i in range(0, len(results), 2)])

    # plot heat maps and theta_cw lines
    for i in range(len(results)):
        ax = axs[f'map_{i}']
        heatmap_plot = ax.imshow(heat_maps[i])
        ax.set_xticks(np.arange(len(param_batches_1[i])), labels=[f"{tau*1e15:.1f}" for tau in param_batches_1[i]])
        ax.set_yticks(np.arange(len(param_batches_2[i])), labels=[f"{GDD*1e30:.1f}" for GDD in param_batches_2[i]])
        ax.set_xlabel(r'Delay (fs)')
        ax.set_ylabel(r'Seed GDD (fs$^2$)')
        ax.set_title(title + f'\n$I_p$={pump_I_p[i]:.1f} GW/cm²')

    # initialize plot for selected spectra
    ax2 = axs['selected_plot']
    ax2.set_title("Spectral Power Density")
    ax2.set_xlabel("$\\nu$ (THz)    (press b to toggle scale)")
    ax2.set_ylabel("Power Spectrum ( J / THz )    (press n to toggle mode)")
    ax2.set_title("Spectral Power Density")

    ##################
    # Event handlers #
    ##################

    highlights = []
    lines = []
    selected_tiles = []
    x_scale = 'frequency' # 'frequency' or 'wavelength'
    y_mode = 'spectrum' # 'spectrum' or 'gain'

    def on_button_press(event):

        if event.key == "b":
            global x_scale

            if x_scale == 'frequency':
                x_scale = 'wavelength'
                for line in lines:
                    old_xdata = line.get_xdata() # frequency in THz
                    old_xlims = ax2.get_xlim()
                    new_xdata = const.c / (old_xdata * 1e12) * 1e9 # wavelength in nm
                    new_xlims = (const.c / (old_xlims[1] * 1e12) * 1e9, const.c / (old_xlims[0] * 1e12) * 1e9)
                    line.set_xdata(new_xdata)
                    ax2.set_xlim(new_xlims)
                    ax2.set_xlabel("$\\lambda$ (nm)    (press b to toggle scale)")
            else:
                x_scale = 'frequency'
                for line in lines:
                    old_xdata = line.get_xdata() # wavelength in nm
                    old_xlims = ax2.get_xlim()
                    new_xdata = const.c / (old_xdata * 1e-9) * 1e-12 # frequency in THz
                    new_xlims = (const.c / (old_xlims[1] * 1e-9) * 1e-12, const.c / (old_xlims[0] * 1e-9) * 1e-12)
                    line.set_xdata(new_xdata)
                    ax2.set_xlim(new_xlims)
                    ax2.set_xlabel("$\\nu$ (THz)    (press b to toggle scale)")
            ax2.figure.canvas.draw()

        elif event.key == "n":
            global y_mode

            y_max = -np.inf
            y_min = np.inf

            y_mode = 'spectrum' if y_mode == 'gain' else 'gain'

            for line, (x, y, ax_index) in zip(lines, selected_tiles):

                if y_mode == 'gain':
                    output_spectrum = results[ax_index].spectrum_y[y, x][lmd_filters[ax_index]]
                    input_spectrum = gaussian_profile(signal_freqs[ax_index][lmd_filters[ax_index]], seed_freqs[ax_index], seed_bandwidths[ax_index], seed_energies[ax_index])
                    ydata = output_spectrum / input_spectrum
                    ax2.set_ylabel("gain    (press n to toggle mode)")
                    
                else:
                    ydata = results[ax_index].spectrum_y[y, x][lmd_filters[ax_index]] * 1e12 # convert to J/THz
                    ax2.set_ylabel("Power Spectrum ( J / THz )    (press n to toggle mode)")
                        
                y_max = max(y_max, ydata.max())
                y_min = min(y_min, ydata.min())   
                line.set_ydata(ydata)
                

            abs_range = y_max - y_min
            ax2.set_ylim(y_min - abs_range * 0.05, y_max + abs_range * 0.08)
            ax2.legend()
            ax2.figure.canvas.draw() 

        elif event.key == "a":
            # autoscale y-axis
            ax2.relim(visible_only=True)
            ax2.autoscale_view()
            ax2.figure.canvas.draw()

        elif event.key == "r":
            plt.tight_layout()
            plt.draw()
    
    def add_plot(index):

        global highlights
        global lines
        global x_scale
        global y_mode

        x, y, ax_index = index

        signal_freq = signal_freqs[ax_index]
        signal_lmd = signal_lmds[ax_index]
        lmd_filter = lmd_filters[ax_index]
        param_batch_1 = param_batches_1[ax_index]
        param_batch_2 = param_batches_2[ax_index]
        seed_freq = seed_freqs[ax_index]
        seed_bandwidth = seed_bandwidths[ax_index]
        seed_energy = seed_energies[ax_index]
        result = results[ax_index]

        # get xdata
        if x_scale == "frequency":
            xdata = signal_freq[lmd_filter] * 1e-12
        else:
            xdata = signal_lmd[lmd_filter] * 1e9

        # get ydata
        if y_mode == 'spectrum':
            ydata = result.spectrum_y[y, x][lmd_filter] * 1e12 # convert to J/THz
        elif y_mode == 'gain':
            output_spectrum = result.spectrum_y[y, x][lmd_filter]
            input_spectrum = gaussian_profile(signal_freq[lmd_filter], seed_freq, seed_bandwidth, seed_energy)
            ydata = output_spectrum / input_spectrum

        # find corresponding batch parameter indices
        param_x = param_batch_1[x]
        param_y = param_batch_2[y]

        new_highlight = axs[f"map_{ax_index}"].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                                                edgecolor='red', facecolor='none', lw=2))
        highlights.append(new_highlight)
        line,  = ax2.plot(xdata, ydata, label=f"$\\tau={param_x*1e15:.1f}$fs, GDD={param_y*1e30:.2f}fs$^2$")
        lines.append(line)

        if ax2.get_ylim()[1] < ydata.max():
            ax2.set_ylim(ax2.get_ylim()[0], ydata.max() * 1.1) 
        ax2.legend()

    def on_click(event):
        global highlights
        global lines
        global selected_tiles

        if event.inaxes in [axs[f'map_{i}'] for i in range(len(results))]:

            x, y = int(round(event.xdata)), int(round(event.ydata))
            ax_index = [axs[f'map_{i}'] for i in range(len(results))].index(event.inaxes)

            ctrl_pressed = event.key == 'control'

            # plotting logic depends on ctrl
            if not ctrl_pressed:

                # set selected indices to only the new one
                selected_tiles = [(x, y, ax_index)]

                # remove previous highlights and lines from plot
                for i in range(len(highlights)):
                    highlights.pop(0).remove()
                    lines.pop(0).remove()

                # add new line and highlight
                add_plot((x, y, ax_index))
            else:

                # if already plotted, remove
                if (x, y, ax_index) in selected_tiles:
                    loc = selected_tiles.index((x, y, ax_index))
                    lines.pop(loc).remove()
                    highlights.pop(loc).remove()
                    selected_tiles.pop(loc)
                # if not plotted, add plot
                else:
                    # add new line and highlight
                    add_plot((x, y, ax_index)) # appends corresponding line to lines and highlight to highlights
                    selected_tiles.append((x, y, ax_index))

            ax2.legend()
            axs[f"map_{ax_index}"].figure.canvas.draw()
            ax2.figure.canvas.draw()

    
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_button_press)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(0.01)
    plt.tight_layout()
    plt.draw()
    plt.show()



