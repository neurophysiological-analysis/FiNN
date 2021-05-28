'''
Created on Jun 11, 2020

@author: voodoocode
'''

import finn.same_frequency_coupling.frequency_domain.complex_coherency as fd_cc
import finn.same_frequency_coupling.__misc as misc

def run(data_X, data_Y, nperseg, pad_type, fs, nfft, window):
    """
    Calculate complex coherency from time domain data.
    
    @param data_X: data set X from time domain.
    @param data_Y: data set Y from time domain.
    @param nperseg: number of samples in a fft segment.
    @param pad_type: padding type to be applied.
    @param fs: Sampling frequency.
    @param nfft: Size of fft window.
    @param window: Type of windowing function (for fft).
    
    @return: Complex coherency
    """

    seg_data_X = misc.__segment_data(data_X, nperseg, pad_type)
    seg_data_Y = misc.__segment_data(data_Y, nperseg, pad_type)

    (bins, f_data_X) = misc.__calc_FFT(seg_data_X, fs, nfft, window)
    (_,    f_data_Y) = misc.__calc_FFT(seg_data_Y, fs, nfft, window)

    return (bins, fd_cc.run(f_data_X, f_data_Y))















