'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np
import scipy.signal

def run(low_freq_data, high_freq_data):
    phase_low_freq_signal = np.angle(scipy.signal.hilbert(low_freq_data))
    amplitude_high_freq_signal = np.abs(scipy.signal.hilbert(high_freq_data))
    
    vectors = amplitude_high_freq_signal * np.exp(np.complex(0, 1) * phase_low_freq_signal)
    
    score = np.abs(np.average(vectors))
    
    return score