"""
Created on Dec 29, 2020.

This module provides different functions to estimate sfc between two signals from the time domain.

@author: voodoocode

"""

import numpy as np
import finnpy.sfc._misc as misc  # @UnresolvedImport
import finnpy.sfc.fd as fd  # @UnresolvedImport
import finnpy.sfc._wpli as calc_wpli  # @UnresolvedImport
import finnpy.sfc._psi as calc_psi  # @UnresolvedImport
import finnpy.sfc._dac as calc_dac  # @UnresolvedImport

def run_dac(data_1, data_2, f_min, f_max, fs, nperseg, nfft, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
    """
    Calculate the directional absolute coherence between two signals. Assumes data_1 and data_2 to be from time domain.
    
    As the coherence is similar to the Fouier Transform of the Pearson correlation coefficient, the magnitude informs of the strength of the correlation and whereas the sign of the imaginary part informs on the direction.
    
    Important design decision:
    - In case data_2 happens before data_1, the sign of the psi (used to gain directional information) is defined to be positive.
    - In case data_1 happens before data_2, the sign of the psi (used to gain directional information)  is defined to be negative.
    
    The sign of the imaginary part of the coherence is a sine with a frequency of f = 1 in [-180°, 180°]. Naturally, there are two roots of this sine, one at 0° and another at -180°/180°. Around these root phase shifts, the calculated sign is proportionally more sensetive to noise in the signal. Therefore, in case of phase shifts from [-thresh°, +thresh°] the amplitude is corrected to 0. Furthermore, any same_frequency_coupling with a phase shift of ~0° is (mostly) indistingusihable from volume conduction effects.
    
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    f_min : float
           Minimum frequency of the frequency range on which coherency gets evaluated.
    f_max : float
           Maximum frequency of the frequency range on which coherency gets evaluated.
    fs : float
         Sampling frequency
    nperseg : float
              Size of individual segments in fft.
    nfft : float
           FFT window size.
    return_signed_conn : boolean
                         Flag whether the absolute coherence should be multiplied with [-1, 1] for directional information
    minimal_angle_thresh : float
                           The minimal angle (phase shift) to evaluate in this analysis. Any angle smaller than the angle defined by minimal_angle_thresh is considered volume conduction and therefore replace with np.nan.
    volume_conductance_ratio : float
                               Defines the ratio of below threshold connectivity values to identify volume conductance.
    
    Returns
    -------
    float
        Connectivity between data_1 and data_2 measured using the directionalized absolute coherence.
    """
    (bins, coh) = run_cc(data_1, data_2, nperseg, "zero", fs, nfft, "hann")
    
    return calc_dac.run(coh, bins, f_min, f_max, return_signed_conn, minimal_angle_thresh, volume_conductance_ratio)
    
def run_wpli(data_1, data_2, fs, nperseg, nfft, window = "hann", pad_type = "zero"):
    """
    Calculate the weighted phase lag index between two signals.
    
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    fs : float
         Sampling frequency
    nperseg : float
              Size of individual segments in fft.
    nfft : float
           FFT window size.
    window : str
             FFT window type. Supported window types are listed at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.
    pad_type : str
               Padding type, currently only "zero" padding is supported.
    
    Returns
    -------
    tuple of (list, list)
        - bins : list
                 Frequency bins.
        - conn : list
                 Coherence values of the respective frequency bins measured via weighted phase lag index.
    """
    s_xy = list()
    for block_start in np.arange(0, np.min([len(data_1), len(data_2)]) - nperseg, nperseg):
        loc_data1 = data_1[block_start:(block_start + nperseg)]
        loc_data2 = data_2[block_start:(block_start + nperseg)]
        
        seg_data_1 = misc.segment_data(loc_data1, nperseg, pad_type)
        seg_data_2 = misc.segment_data(loc_data2, nperseg, pad_type)
    
        (bins, f_data_1) = misc.calc_FFT(seg_data_1, fs, nfft, window)
        (_,    f_data_2) = misc.calc_FFT(seg_data_2, fs, nfft, window)  # noqa: E241
    
        s_xy.append((np.conjugate(f_data_1[0, :]) * f_data_2[0, :] * 2))

    s_xy = np.asarray(s_xy)
    
    return (bins, calc_wpli.run(s_xy))

def run_psi(data_1, data_2, nperseg_outer, fs, nperseg_inner, nfft, window, pad_type, f_min, f_max, f_step_sz = 1, normalize = True):
    """
    Calculate the phase slope index between two signals. Assumes data_1 and data_2 to be from time domain.
  
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    nperseg_outer : int
                    Outer window size. If normalize = False, this parameter is not used.
    fs : float
         Sampling frequency
    nperseg_inner : int
                    Inner window size.
    nfft : float
           FFT window size.
    window : str
             FFT window type. Supported window types are listed at
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.
    pad_type : str
               Padding type, currently only "zero" padding is supported.
    f_min : float
           Minimum frequency of the frequency range on which coherency gets evaluated.
    f_max : float
           Maximum frequency of the frequency range on which coherency gets evaluated.
    f_step_sz : float
                Frequency step size in the evaluated interval.
    normalize : boolean
                Determines whether to normalize by dividing through the variance.
    
    Returns
    -------
    float
        Connectivity between data_1 and data_2 measured using the phase slope index.
    """
    if (normalize is True):
        data_coh = list()
        
        for idx_start in np.arange(0, len(data_1), nperseg_outer):
            
            (bins, cc) = run_cc(data_1[idx_start:(idx_start + nperseg_outer)], data_2[idx_start:(idx_start + nperseg_outer)], nperseg_inner, pad_type, fs, nfft, window)
            
            data_coh.append(cc)
    else:
        (bins, tmp) = run_cc(data_1, data_2, nperseg_inner, "zero", fs, nfft, "hann")
        data_coh = [tmp]
    
    return calc_psi.run(data_coh, bins, f_min, f_max, f_step_sz)
    
def run_ic(data_1, data_2, fs, nperseg, nfft):
    """
    Calculate the imaginary coherency between two signals. Assumes data_1 and data_2 to be from time domain.
    
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    fs : float
         Sampling frequency
    nperseg : float
              Size of individual segments in fft.
    nfft : float
           FFT window size.
    
    Returns
    -------
    tuple of (list, list)
        - bins : list
                 Frequency bins.
        - conn : list
                 Coherence values of the respective frequency bins measured via imaginary coherence.
    """
    (bins, coh) = run_cc(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, np.imag(coh))
    
def run_msc(data_1, data_2, fs, nperseg, nfft):
    """
    Calculate the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from time domain.
  
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    fs : float
         Sampling frequency
    nperseg : float
              Size of individual segments in fft.
    nfft : float
           FFT window size.
    
    Returns
    -------
    tuple of (list, list)
        - bins : list
                 Frequency bins.
        - conn : list
                 Coherence values of the respective frequency bins measured via magnitude squared coherence.
    """
    (bins, coh) = run_cc(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, np.square(np.abs(coh)))
    
def run_cc(data_1, data_2, nperseg, pad_type, fs, nfft, window):
    """
    Calculate complex coherency from time domain data.
    
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    nperseg : float
              Size of individual segments in fft.
    pad_type : str
               Padding type, currently only "zero" padding is supported.
    fs : float
         Sampling frequency
    nfft : float
           FFT window size.
    window : str
             FFT window type. Supported window types are listed at
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.
    
    Returns
    -------
    tuple of (list, list)
        - bins : list
                 Frequency bins.
        - conn : list
                 Coherence values of the respective frequency bins measured via complex coherence.
    """
    seg_data_1 = misc.segment_data(data_1, nperseg, pad_type)
    seg_data_2 = misc.segment_data(data_2, nperseg, pad_type)
    
    seg_data_1 = seg_data_1[:seg_data_2.shape[0], :]
    seg_data_2 = seg_data_2[:seg_data_1.shape[0], :]

    (bins, f_data_1) = misc.calc_FFT(seg_data_1, fs, nfft, window)
    (_,    f_data_2) = misc.calc_FFT(seg_data_2, fs, nfft, window)  # noqa: E241

    return (bins, fd.run_cc(f_data_1, f_data_2))
