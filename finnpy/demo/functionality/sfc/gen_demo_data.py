"""
Created on Aug 1, 2025.

@author: voodoocode
"""

import numpy as np
import finnpy.filters.frequency as ff  # @UnresolvedImport

def _sine(freq, amp, phase, fs, size):
    """
    
    Creates a specified sine wave.
    
    Parameters
    ----------
    
    freq : float
           Frequency.
    amp : float
          Amplitude.
    phase : float
    fs : float
         Sampling frequency.
    samples : int
              Size of the returned signal.
    
    """
    
    phase = 0
    x = np.arange(0, size)
    
    return amp * np.sin(2 * np.pi * freq * (x / fs - (phase / 360) / freq))

def gen_gauss_signal(f_min, f_max, f_step_width, fs, size):
    """
    Generates a signal with spectrally Gaussian activity in a specific frequency range.
    
    Parameters
    ----------
    
    f_min : float
            Minimum frequency
    f_max : float
            Maximimum freuqency.
    f_step_width : float
                   Step width between f_min and f_max
    fs : float
         Sampling frequency
    size : int
           Lenght of the signal
    """

    f_std = np.sqrt(f_max - f_min)
    f_mean = (f_max + f_min)/2
    
    sines = list()
    for curr_f in np.arange(f_min, f_max, f_step_width):
        amp = 1/(f_std * np.sqrt(2*np.pi)) * np.exp(-np.pow((curr_f - f_mean), 2)/(2*f_std*f_std))
        sines.append(_sine(curr_f, amp, 0, fs, size))
    
    signal = np.sum(np.asarray(sines), axis = 0)
    
    signal = signal[int(2*fs):-int(2*fs)]

    return signal

def gen_wn_signal(f_min, f_max, fs, size):
    
    signal = np.random.randn(size)
    
    signal = ff.fir(np.copy(signal), f_min, f_max, 1, fs, ripple_pass_band = 10e-2, stop_band_suppression = 10e-2, fft_win_sz = 2, pad_type = "zero", mode = "fast")
    
    #------------------------------------------- import matplotlib.pyplot as plt
    #------------------------------------------------------- import scipy.signal
    # (bins, pwr) = scipy.signal.welch(signal, fs, "hann", fs, fs // 2, fs, "linear")
    # plt.plot(bins[np.argmin(np.abs(bins - 5)):np.argmin(np.abs(bins - 50))], pwr[np.argmin(np.abs(bins - 5)):np.argmin(np.abs(bins - 50))])
    #---------------------------------------------------- plt.show(block = True)

    return signal
