'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np
import scipy.signal
import scipy.stats

def run(low_freq_data, high_freq_data, frequency_window_half_size = 10, frequency_step_width = 20):
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, frequency_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, frequency_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < frequency_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(high_freq_data[phase_indices]))
        
    amplitude_signal = np.concatenate((amplitude_signal, amplitude_signal))
    amplitude_signal /= np.nansum(amplitude_signal)
    
    len_signal = len(amplitude_signal)
    uniform_signal = np.random.uniform(np.nanmin(amplitude_signal), np.nanmax(amplitude_signal), len_signal)
    
    score = (scipy.stats.entropy(amplitude_signal, uniform_signal, len_signal))/np.log(len_signal)

    return score