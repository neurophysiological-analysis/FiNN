"""
Created on Aug 1, 2025.

@author: voodoocode
"""

import scipy.signal

def get_pwr(data, fs, window = "hann", nperseg = None, noverlap = None, nfft = None, detrend = "linear"):
    """
    
    Wraps scipy's implementation of Welch's method to calculate spectral power
    
    Parameters
    ----------
    
    data : np.ndarray, shape(samp_cnt,)
           Input data.
    fs : int
         Sampling frequency.
    window : string 
             Windowing function to be used, defaults to "hann".
    nperseg : int
              Number of datapoints in each segment, defaults to fs for 1 Hz bins.
    noverlap : int
               Number of overlapping samples.
    nfft : int
           Number of datapoints in each FFT, defaults to fs for 1 Hz bins.
    detrend : string
              Flag for different detrending options ('None', 'constant', linear'), defaults to 'linear'.
    
    Returns
    -------
    tuple of (list, list)
        - bins : list
                 Frequency bins.
        - conn : list
                 Power values of the respective frequency bins.
    
    """
    return scipy.signal.welch(data, fs, window, nperseg, noverlap, nfft, detrend)

