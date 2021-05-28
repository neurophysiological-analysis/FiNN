'''
Created on Jun 2, 2020

@author: voodoocode
'''

import numpy as np

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import finn.same_frequency_coupling.time_domain.directional_absolute_coherency as dac

def main():
    data = np.load("/mnt/data/AnalysisFramework/beta2/demo_data/dac/demo_data.npy")
    frequency_sampling = 5500
    frequency_peak = 30
    
    noise_weight = 0.2
    
    phase_min = -90
    phase_max = 270
    phase_step = 4
    
    fmin = 28
    fmax = 33
    nperseg = frequency_sampling
    nfft = frequency_sampling
    return_signed_conn = True
    minimal_angle_thresh = 4
    
    #Generate data
    offset = int(np.ceil(frequency_sampling/frequency_peak))
    loc_data = data[offset:]
    signal_1 = np.zeros((loc_data).shape)
    signal_1 += loc_data
    signal_1 += np.random.random(len(loc_data)) * noise_weight
    
    conn_vals = list()
    fig = plt.figure()
    for phase_shift in np.arange(phase_min, phase_max, phase_step):
        loc_offset = offset - int(np.ceil(frequency_sampling/frequency_peak * phase_shift/360))
        loc_data = data[(loc_offset):]
        signal_2 = np.zeros(loc_data.shape)
        signal_2 += loc_data
        signal_2 += np.random.random(len(loc_data)) * noise_weight
        
        plt.cla()
        plt.plot(signal_1[:500], color = "blue")
        plt.plot(signal_2[:500], color = "red")
        plt.title("Signal shifted by %2.f degree around %2.2fHz" % (float(phase_shift), float(frequency_peak)))
        plt.show(block = False)
        plt.pause(0.001)
        
        dac_val = dac.run(signal_1, signal_2, fmin , fmax, frequency_sampling, nperseg, nfft, return_signed_conn, minimal_angle_thresh)[1]
        conn_vals.append(dac_val if (np.isnan(dac_val) == False) else 0)
    
    plt.close(fig)
    
    plt.figure()
    plt.scatter(np.arange(phase_min, phase_max, phase_step), conn_vals)
    plt.show(block = True)

main()











