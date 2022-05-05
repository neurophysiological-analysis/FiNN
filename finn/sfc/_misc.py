'''
Created on Dec 29, 2020

This module implements a number of functions used in across several sfc metrics.

@author: voodoocode
'''

import numpy as np
import scipy.signal

def _segment_data(data, nperseg, pad_type = "zero"):
    """
    Chop data into segments.
    
    @param data: Input data; single vector of samples.
    @param nperseg: Length of individual segments.
    @param pad_type: Type of applied padding.
    
    @return: Segmented data.
    """
    
    seg_cnt = int(len(data)/nperseg)
    pad_width = nperseg - (len(data) - (seg_cnt * nperseg))
    
    if (pad_width != 0):
        if (pad_type == "zero"):
            s_data = np.pad(data, (0, pad_width), "constant", constant_values = 0)
        else:
            raise NotImplementedError("Error, only supports zero padding")
        seg_cnt += 1
        
    return np.reshape(s_data, (seg_cnt, nperseg))[:int(len(data)/nperseg), :]

def _calc_FFT(data, fs, nfft, window = "hanning"):
    """
    Calculate fft from data
    
    @param data: Input data; single vector of samples.
    @param fs: Sampling frequency.
    @param nfft: FFT window size.
    @param window: Window type applied during fft.
    
    @return: (bins, f_data) - frequency bins and corresponding complex fft information.
    """
    m_data = data - np.mean(data)
    m_data = data - np.repeat(np.expand_dims(np.mean(data, axis = 1), axis = 1), data.shape[1], axis = 1)

    if (window == "hanning" or window == "hann"):
        win = np.hanning(data.shape[1])
    else:
        win = np.concatenate((scipy.signal.get_window(window, data.shape[1] - 1, fftbins = True), [0]))
    w_data = m_data * win

    if(np.complex128 == data.dtype or np.complex256 == data.dtype or np.complex64 == data.dtype):
        f_data = np.fft.fft(w_data, n = nfft, axis = 1); f_data = f_data[:, :int(f_data.shape[1]/2 + 1)]
    else:
        f_data = np.fft.rfft(w_data, n = nfft, axis = 1)

    bins = np.arange(0, f_data.shape[1], 1) * fs/nfft

    return(bins, f_data)








