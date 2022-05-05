'''
Created on Jan 4, 2021

@author: voodoocode
'''

import numpy as np

import finn.sfc.td as td
import finn.sfc.fd as fd

import finn.sfc._misc as misc
import demo_data.demo_data_paths as paths

def main():
    data = np.load(paths.fct_sfc_data)
    frequency_sampling = 5500
    frequency_peak = 30
    
    noise_weight = 0.2
    
    phase_shift = 153
    
    nperseg = frequency_sampling
    nfft = frequency_sampling
    
    #Generate data
    offset = int(np.ceil(frequency_sampling/frequency_peak))
    loc_data = data[offset:]
    signal_1 = np.zeros((loc_data).shape)
    signal_1 += loc_data
    signal_1 += np.random.random(len(loc_data)) * noise_weight
    
    loc_offset = offset - int(np.ceil(frequency_sampling/frequency_peak * phase_shift/360))
    loc_data = data[(loc_offset):]
    signal_2 = np.zeros(loc_data.shape)
    signal_2 += loc_data
    signal_2 += np.random.random(len(loc_data)) * noise_weight
    
    (bins, cc_td) = calc_from_time_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft)
    cc_fd = calc_from_frequency_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft)
    
    if ((cc_fd == cc_td).all() == False):
        print("Error")
    
def calc_from_time_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft, window = "hann", pad_type = "zero"):
    return td.run_cc(signal_1, signal_2, nperseg, pad_type, frequency_sampling, nfft, window)

def calc_from_frequency_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft, window = "hann", pad_type = "zero"):
    seg_data_X = misc._segment_data(signal_1, nperseg, pad_type)
    seg_data_Y = misc._segment_data(signal_2, nperseg, pad_type)

    (bins, fd_signal_1) = misc._calc_FFT(seg_data_X, frequency_sampling, nfft)
    (_,    fd_signal_2) = misc._calc_FFT(seg_data_Y, frequency_sampling, nfft)
    
    return fd.run_cc(fd_signal_1, fd_signal_2)
    
    
main()
print("terminated succesfully")