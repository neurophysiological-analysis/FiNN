'''
Created on Jun 11, 2020

@author: voodoocode
'''

import numpy as np
import finn.same_frequency_coupling.time_domain.complex_coherency as td_cc

def run(data_1, data_2, fs, nperseg, nfft):
    """
    Calculates the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from time domain.
  
    @param data_1: First dataset from time domain; vector of samples
    @param data_2: Second dataset from time domain; vector of samples
    @param fs: Sampling frequency
    @param nperseg: Size of individual segments in fft
    @param nfft: fft window size
    
    @return (bins, conn) - Frequency bins and corresponding same_frequency_coupling values
    """
    
    (bins, coh) = td_cc.run(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, np.square(np.abs(coh)))








