"""
Created on Dec 29, 2020.

This module provides different functions to estimate sfc between two signals from the time domain.

@author: voodoocode

"""

import numpy as np
import scipy.signal

def dac_td(data_1, data_2, f_min, f_max, fs, nperseg, nfft, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
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
    (bins, coh) = cc_td(data_1, data_2, nperseg, "zero", fs, nfft, "hann")
    
    return dac_cc(coh, bins, f_min, f_max, return_signed_conn, minimal_angle_thresh, volume_conductance_ratio)
    
def wpli_td(data_1, data_2, fs, nperseg, nfft, window = "hann", pad_type = "zero"):
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
        
        seg_data_1 = _segment_data(loc_data1, nperseg, pad_type)
        seg_data_2 = _segment_data(loc_data2, nperseg, pad_type)
    
        (bins, f_data_1) = _calc_FFT(seg_data_1, fs, nfft, window)
        (_,    f_data_2) = _calc_FFT(seg_data_2, fs, nfft, window)  # noqa: E241
    
        s_xy.append((np.conjugate(f_data_1[0, :]) * f_data_2[0, :] * 2))

    s_xy = np.asarray(s_xy)
    
    return (bins, wpli_cc(s_xy))

def psi_td(data_1, data_2, nperseg_outer, fs, nperseg_inner, nfft, window, pad_type, f_min, f_max, f_step_sz = 1, normalize = True):
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
            
            (bins, cc) = cc_td(data_1[idx_start:(idx_start + nperseg_outer)], data_2[idx_start:(idx_start + nperseg_outer)], nperseg_inner, pad_type, fs, nfft, window)
            
            data_coh.append(cc)
    else:
        (bins, tmp) = cc_td(data_1, data_2, nperseg_inner, "zero", fs, nfft, "hann")
        data_coh = [tmp]
    
    return psi_cc(data_coh, bins, f_min, f_max, f_step_sz)
    
def ic_td(data_1, data_2, fs, nperseg, nfft):
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
    (bins, coh) = cc_td(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, np.imag(coh))
    
def msc_td(data_1, data_2, fs, nperseg, nfft):
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
    (bins, coh) = cc_td(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, np.square(np.abs(coh)))
    
def cc_td(data_1, data_2, nperseg, pad_type, fs, nfft, window):
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
    seg_data_1 = _segment_data(data_1, nperseg, pad_type)
    seg_data_2 = _segment_data(data_2, nperseg, pad_type)
    
    seg_data_1 = seg_data_1[:seg_data_2.shape[0], :]
    seg_data_2 = seg_data_2[:seg_data_1.shape[0], :]

    (bins, f_data_1) = _calc_FFT(seg_data_1, fs, nfft, window)
    (_,    f_data_2) = _calc_FFT(seg_data_2, fs, nfft, window)  # noqa: E241

    return (bins, cc_fd(f_data_1, f_data_2))


def dac_fd(data_1, data_2, bins, f_min, f_max, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
    """
    Calculate the directional absolute coherence between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
    
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
    bins : np.ndarray or list, len(bin_cnt)
           Frequency bins.
    f_min : float
           Minimum frequency of the frequency range on which coherency gets evaluated.
    f_max : float
           Maximum frequency of the frequency range on which coherency gets evaluated.
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
    coh = cc_fd(data_1, data_2)
    
    return dac_cc(coh, bins, f_min, f_max, return_signed_conn, minimal_angle_thresh, volume_conductance_ratio)

def wpli_fd(data_1, data_2):
    """
    Calculate the weighted phase lag index between two signals.
    
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
             
    Returns
    -------
    list
        Connectivity between data_1 and data_2 measured using the weighted phase lag index.
    """
    s_xy = list()
    for block_idx in np.arange(len(data_1)):
        s_xy.append((np.conjugate(data_1[block_idx]) * data_2[block_idx] * 2))

    s_xy = np.asarray(s_xy)
    
    return wpli_cc(s_xy)

def psi_fd(data_1, data_2, bins, f_min, f_max, f_step_sz = 1):
    """
    Calculate the phase slope index between two signals. Assumes data_1 and data_2 to be from time domain.
  
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    bins : np.ndarray or list, len(bin_cnt)
           Frequency bins.
    f_min : float
           Minimum frequency of the frequency range on which coherency gets evaluated.
    f_max : float
           Maximum frequency of the frequency range on which coherency gets evaluated.
    f_step_sz : float
                Frequency step size in the evaluated interval.
    
    Returns
    -------
    float
        Connectivity between data_1 and data_2 measured using the phase slope index.
    """
    data_coh = list()
    for (outer_window_idx, _) in enumerate(data_1):
        data_coh.append(cc_fd(data_1[outer_window_idx], data_2[outer_window_idx]))
    
    return psi_cc(data_coh, bins, f_min, f_max, f_step_sz)

def ic_fd(data_1, data_2):
    """
    Calculate the imaginary coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    
    Returns
    -------
    list
        Connectivity between data_1 and data_2 measured using imaginary coherence.
    """
    return np.imag(cc_fd(data_1, data_2))

def msc_fd(data_1, data_2, bins):
    """
    Calculate the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    bins : np.ndarray or list, len(bin_cnt)
           Frequency bins.
    
    Returns
    -------
    list
        Connectivity between data_1 and data_2 measured using magnitude squared coherence.
    """
    coh = cc_fd(data_1, data_2)
    
    return (bins, np.square(np.abs(coh)))

def cc_fd(data_1, data_2):
    """
    Calculate complex coherency from frequency domain data.
    
    Parameters
    ----------
    data_1 : np.ndarray or list, len(n_samples)
             First dataset from the complex frequency domain; vector of samples.
    data_2 : np.ndarray or list, len(n_samples)
             Second dataset from the complex frequency domain; vector of samples.
    
    Returns
    -------
    list
        Connectivity between data_1 and data_2 measured using complex coherence.
    """
    s_xx = np.conjugate(data_1) * data_1 * 2
    s_yy = np.conjugate(data_2) * data_2 * 2
    s_xy = np.conjugate(data_1) * data_2 * 2

    s_xx = np.mean(s_xx, axis = 0)
    s_yy = np.mean(s_yy, axis = 0)
    s_xy = np.mean(s_xy, axis = 0)

    return s_xy / np.sqrt(s_xx * s_yy)


def dac_cc(data, bins, fmin, fmax, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.4):
    """
    Calculate the directional absolute coherence from complex coherency.
    
    As the coherence is similar to the Fouier Transform of the Pearson correlation coefficient,
    the magnitude informs of the strength of the correlation and whereas the sign
    of the imaginary part informs on the direction.
    
    Important design decision: 
    - In case data_2 happens before data_1, the sign of the psi (used to gain directional information) is defined to be positive.
    - In case data_1 happens before data_2, the sign of the psi (used to gain directional information)  is defined to be negative.
    
    The sign of the imaginary part of the coherence is a sine with a frequency of f = 1 in [-180°, 180°].
    Naturally, there are two roots of this sine, one at 0° and another at -180°/180°.
    Around these root phase shifts, the calculated sign is proportionally more sensetive to noise in the signal.
    Therefore, in case of phase shifts from [-thresh°, +thresh°] the amplitude is corrected to 0.
    Furthermore, any same_frequency_coupling with a phase shift of ~0° is (mostly) indistingusihable from volume conduction effects.
    
    Parameters
    ----------
    data : list
           List of outer window complex coherency estimates calculated from two signals.
    bins : list
           Frequency bins of the complex coherency data.
    fmin : float
           Minimum frequency of the frequency range on which coherency gets evaluated.
    fmax : float
           Maximum frequency of the frequency range on which coherency gets evaluated.
    return_signed_conn : boolean
                         Flag whether the absolute coherence should be multiplied with [-1, 1] for directional information
    minimal_angle_thresh : float
                           The minimal angle (phase shift) to evaluate in this analysis.
                           Any angle smaller than the angle defined by minimal_angle_thresh is
                           considered volume conduction and therefore replace with np.nan.
    volume_conductance_ratio : float
                               Defines the ratio of below threshold connectivity values to identify volume conductance.
    
    Returns
    -------
    float
        Connectivity between data_1 and data_2 measured using the directionalized absolute coherence.
    """
    return _get_dac(data, bins, fmin, fmax, return_signed_conn, minimal_angle_thresh, volume_conductance_ratio)

def wpli_cc(s_xy):
    """
    Calculate the weighted phase lag index between two signals.
    
    Parameters
    ----------
    s_xy : list
           List of complex coherency estimates calculated from two signals.
    
    Returns
    -------
    float
        Connectivity estimated based on the provided complex coherency measured using the weighted lag slope index.
    """
    return _get_wpli(s_xy)

def psi_cc(data, bins, f_min, f_max, f_step_sz = 1):
    """
    Calculate the phase slope index between two signals.
    
    Parameters
    ----------
    data : list
           List of outer window complex coherency estimates calculated from two signals.
    bins : list
           Frequency bins of the sample data.
    f_min : float
            Minimum frequence for the evaluated interval.
    f_max : float
            Maximum frequence for the evaluated interval.
    f_step_sz : float
                Frequency step size in the evaluated interval.
    
    Returns
    -------
    float
        Connectivity estimated based on the provided complex coherency measured using the phase slope index.
    """
    return _get_psi(data, bins, f_min, f_max, f_step_sz)

def ic_cc(data):
    """
    Calculate the imaginary coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    Parameters
    ----------
    data : list
           Complex coherency values; vector of samples.
    
    Returns
    -------
    list
        Connectivity estimated based on the provided complex coherency measured via the imaginary coherence.
    """
    return np.imag(data)
    
def msc_cc(data):
    """
    Calculate the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    Parameters
    ----------
    data : list
           Complex coherency values; vector of samples.
    
    Returns
    -------
    list
        Connectivity estimated based on the provided complex coherency measured via the magnitude squared coherence.
    """
    return np.square(np.abs(data))



def _get_dac(coh, bins, fmin, fmax, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
    """
    Calculate directed absolute coherency from complex coherency.
    
    Parameters
    ----------
    coh : list
          Complex coherency.
    bins : list
           Frequency bins of the complex coherency data.
    fmin : float
           Minimum frequency of the frequency range on which coherency gets evaluated.
    fmax : float
           Maximum frequency of the frequency range on which coherency gets evaluated.
    return_signed_conn : boolean
                         Whether to add directional information and mask volume conductance.
    minimal_angle_thresh : float
                           Minimal phase shift angle to not be considered volume conductance.
    volume_conductance_ratio : float
                               Defines the ratio of below threshold connectivity values to identify volume conductance.
    
    Returns
    -------
    float
        Connectivity score
    """
    coh = np.asarray(coh)
    
    f_min_idx = np.argmin(np.abs(bins - fmin))
    f_max_idx = np.argmin(np.abs(bins - fmax))
    
    psi = 0
    psi_cnt = 0
    vol_cond_cnt = 0
    for freq_idx in range(f_min_idx, f_max_idx):
        if (np.abs(np.imag(coh[freq_idx])) < minimal_angle_thresh / 90):
            vol_cond_cnt += 1
        psi += np.conjugate(coh[freq_idx]) * coh[freq_idx + 1]
        psi_cnt += 1
        
    # In case of volume conductance
    if (vol_cond_cnt >= (len(range(f_min_idx, f_max_idx)) * volume_conductance_ratio)):
        return np.nan
        
    if (return_signed_conn):
        return np.sign(np.imag(psi)) * np.mean(np.square(np.abs(coh))[f_min_idx:f_max_idx])
    else:
        return np.mean(np.square(np.abs(coh))[f_min_idx:f_max_idx])

def _get_psi(data, bins, f_min, f_max, f_step_sz = 1.):
    """
    Calculate the phase slope index (psi) from a list of complex coherency data.
    
    Parameters
    ----------
    data : list or np.ndarray
           List of complex coherency data.
    bins : list or np.ndarray
           Frequency bins of the complex coherency data.
    f_min : float
            Minimum frequency of interest.
    f_max : float
            Maximum frequency of interest.
    f_step_sz : float
                Frequency step size.
    
    Returns
    -------
    float
        Returns the sfc measured as psi computed from data.
    """
    f_min_idx = np.argmin(np.abs(bins - f_min))
    f_max_idx = np.argmin(np.abs(bins - f_max))
    
    psi = np.zeros((len(data)), dtype = np.complex64)
    for (psi_idx, comp_coh) in enumerate(data):
        for freq_idx in range(f_min_idx, f_max_idx, 1):
            psi[psi_idx] += np.conjugate(comp_coh[freq_idx]) * comp_coh[freq_idx + f_step_sz]
        psi[psi_idx] = np.imag(psi[psi_idx])
    psi = np.asarray(psi.real, dtype = np.float32)
    
    if (len(data) > 1):
        var = 0
        for idx in range(len(data)):
            var += np.var(np.concatenate((psi[:idx], psi[(idx + 1):])))
        var /= len(data)
        
        return np.mean(psi) / (np.sqrt(var) * 2)
    else:
        return psi[0]
    
def _get_wpli(s_xy):
    """
    Calculate the weighted phase lag index (wPLI) from a list of complex coherency data.
    
    Parameters
    ----------
    s_xy : list or np.ndarray
           Complex coherency data.
    
    Returns
    -------
    list
        Returns the sfc measured as wPLI computed from data.
    """
    divident = np.sum(np.abs(np.imag(s_xy)) * np.sign(np.imag(s_xy)), axis = 0)
    divisor = np.sum(np.abs(np.imag(s_xy)), axis = 0)
    divisor[divisor == 0] = np.nan  # Zeros in the divisor are replaced with np.nan avoid divide by zero warnings/errors
    
    return divident / divisor    


def _segment_data(data, nperseg, pad_type = "zero"):
    """
    Chop data into segments.
    
    Parameters
    ----------
    data : list or np.ndarray
           Input data; single vector of samples.
    nperseg : int
              Length of individual segments.
    pad_type : str
               Type of applied padding.
    
    Returns
    -------
    list or np.ndarray
        Segmented data.
        
    Raises
    ------
    NotImplementedError
        If an invalid segmention method was selected.
    """
    seg_cnt = int(len(data) / nperseg)
    pad_width = nperseg - (len(data) - (seg_cnt * nperseg))
    
    if (pad_width != 0):
        if (pad_type == "zero"):
            s_data = np.pad(data, (0, pad_width), "constant", constant_values = 0)
        else:
            raise NotImplementedError("Error, only supports zero padding")
        seg_cnt += 1
        
        return np.reshape(s_data, (seg_cnt, nperseg))[:int(len(data) / nperseg), :]
    else:
        return data

def _calc_FFT(data, fs, nfft, window = "hanning"):
    """
    Calculate fft from data.
    
    Parameters
    ----------
    data : list or np.ndarray
           Input data; single vector of samples.
    fs : float
         Sampling frequency.
    nfft : int
           FFT window size.
    window : str
             Window type applied during fft.
    
    Returns
    -------
    tuple of (list, list)
        - bins : list
                 bins of the complex fft.
        - f_data : list
                   Frequency values of the complex fft.
    """
    m_data = data - np.repeat(np.expand_dims(np.mean(data, axis = 1), axis = 1), data.shape[1], axis = 1)

    if (window == "hanning" or window == "hann"):
        win = np.hanning(data.shape[1])
    else:
        win = np.concatenate((scipy.signal.get_window(window, data.shape[1] - 1, fftbins = True), [0]))
    w_data = m_data * win

    if (np.complex128 is data.dtype or np.complex256 is data.dtype or np.complex64 is data.dtype):
        f_data = np.fft.fft(w_data, n = nfft, axis = 1); f_data = f_data[:, :int(f_data.shape[1] / 2 + 1)]
    else:
        f_data = np.fft.rfft(w_data, n = nfft, axis = 1)

    bins = np.arange(0, f_data.shape[1], 1) * fs / nfft

    return (bins, f_data)

