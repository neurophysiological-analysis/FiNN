'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np

import finn.same_frequency_coupling.__calc_phase_slope_index as calc
import finn.same_frequency_coupling.time_domain.complex_coherency as td_cc

def run(data_X, data_Y, nperseg_outer, fs, nperseg_inner, nfft, window, pad_type, f_min, f_max, f_step_sz = 1):
    
    data_coh = list()
    
    for idx_start in np.arange(0, len(data_X), nperseg_outer):
        
        (bins, cc) = td_cc.run(data_X[idx_start:(idx_start + nperseg_outer)], data_Y[idx_start:(idx_start + nperseg_outer)], nperseg_inner, pad_type, fs, nfft, window)
        
        data_coh.append(cc)
    
    return calc.run(data_coh, bins, f_min, f_max, f_step_sz)