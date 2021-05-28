'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np
import finn.same_frequency_coupling.__misc as misc
import finn.same_frequency_coupling.__calc_weighted_phase_lag_index as calc

def run(data1, data2, fs, nperseg, nfft, window = "hann", pad_type = "zero"):
    s_xy = list()
    for block_start in np.arange(0, np.min([len(data1), len(data2)]) - nperseg, nperseg):
        loc_data1 = data1[block_start:(block_start + nperseg)]
        loc_data2 = data2[block_start:(block_start + nperseg)]
        
        seg_data_X = misc.__segment_data(loc_data1, nperseg, pad_type)
        seg_data_Y = misc.__segment_data(loc_data2, nperseg, pad_type)
    
        (_, f_data_X) = misc.__calc_FFT(seg_data_X, fs, nfft, window)
        (_,    f_data_Y) = misc.__calc_FFT(seg_data_Y, fs, nfft, window)
    
        s_xy.append((np.conjugate(f_data_X[0, :]) * f_data_Y[0, :] * 2))

    s_xy = np.asarray(s_xy)
    
    return calc.run(s_xy)