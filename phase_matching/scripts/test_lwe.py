import LightwaveExplorer as lwe
import numpy as np
import matplotlib.pyplot as plt


lwe_results_filename = r"C:\Users\juliu\Documents\VSCode Projects\OPA_tools\LWE_results\NOPA_NC_scan.txt"

if __name__=="__main__":
    result = lwe.load(lwe_results_filename)
    spec = result.spectrum_y[0, 1]
    print(result.spectrum_y.shape)
    plt.plot(spec)
    plt.show()