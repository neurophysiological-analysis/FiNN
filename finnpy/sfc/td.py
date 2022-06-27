"""
Created on Dec 29, 2020

This module provides different functions to estimate sfc between two signals from the time domain.

@author: voodoocode

"""

import numpy as np
import finnpy.sfc._misc as misc
import finnpy.sfc.fd as fd
import finnpy.sfc._wpli as calc_wpli
import finnpy.sfc._psi as calc_psi
import finnpy.sfc._dac as calc_dac

def run_dac(data_1, data_2, fmin, fmax, fs, nperseg, nfft, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
    """
    Calculates the directional absolute coherence between two signals. Assumes data_1 and data_2 to be from time domain.
    
    As the coherence is similar to the Fouier Transform of the Pearson correlation coefficient, the magnitude informs of the strength of the correlation and whereas the sign of the imaginary part informs on the direction.
    
    Important design decision:
    - In case data_2 happens before data_1, the sign of the psi (used to gain directional information) is defined to be positive.
    - In case data_1 happens before data_2, the sign of the psi (used to gain directional information)  is defined to be negative.
    
    The sign of the imaginary part of the coherence is a sine with a frequency of f = 1 in [-180°, 180°]. Naturally, there are two roots of this sine, one at 0° and another at -180°/180°. Around these root phase shifts, the calculated sign is proportionally more sensetive to noise in the signal. Therefore, in case of phase shifts from [-thresh°, +thresh°] the amplitude is corrected to 0. Furthermore, any same_frequency_coupling with a phase shift of ~0° is (mostly) indistingusihable from volume conduction effects.
    
    :param data_1: First dataset from time domain; vector of samples.
    :param data_2: Second dataset from time domain; vector of samples.
    :param fmin: Minimum frequency of the frequency range on which coherency gets evaluated.
    :param fmax: Maximum frequency of the frequency range on which coherency gets evaluated.
    :param fs: Sampling frequency.
    :param nperseg: Size of individual segments in fft.
    :param nfft: fft window size.
    :param return_signed_conn: Flag whether the absolute coherence should be multiplied with [-1, 1] for directional information.
    :param minimal_angle_thresh: The minimal angle (phase shift) to evaluate in this analysis. Any angle smaller than the angle defined by minimal_angle_thresh is considered volume conduction and therefore replace with np.nan. 
    :param volume_conductance_ratio: Defines the ratio of below threshold connectivity values to identify volume conductance.
    
    :return: (bins, conn) - Frequency bins and corresponding same_frequency_coupling values.
    """
        
    (bins, coh) = run_cc(data_1, data_2, nperseg, "zero", fs, nfft, "hann")
    
    return (bins, calc_dac.run(coh, bins, fmin, fmax, return_signed_conn, minimal_angle_thresh, volume_conductance_ratio))
    
def run_wpli(data1, data2, fs, nperseg, nfft, window = "hann", pad_type = "zero"):
    """
    This function calculates the weighted phase lag index between two signals. 
    
    :param data1: First dataset from time domain; vector of samples.
    :param data2: Second dataset from time domain; vector of samples.
    :param fs: Sampling frequency.
    :param nperseg: Number of data points per segment.
    :param nfft: FFT window size.
    :param window: FFT window type. Supported window types are listed at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.
    :param pad_type: Padding type, currently only "zero" padding is supported.
    
    :return: (bins, conn) - Frequency bins and corresponding weighted phase lag index values.
    """
        
    s_xy = list()
    for block_start in np.arange(0, np.min([len(data1), len(data2)]) - nperseg, nperseg):
        loc_data1 = data1[block_start:(block_start + nperseg)]
        loc_data2 = data2[block_start:(block_start + nperseg)]
        
        seg_data_1 = misc._segment_data(loc_data1, nperseg, pad_type)
        seg_data_2 = misc._segment_data(loc_data2, nperseg, pad_type)
    
        (bins, f_data_1) = misc._calc_FFT(seg_data_1, fs, nfft, window)
        (_,    f_data_2) = misc._calc_FFT(seg_data_2, fs, nfft, window)
    
        s_xy.append((np.conjugate(f_data_1[0, :]) * f_data_2[0, :] * 2))

    s_xy = np.asarray(s_xy)
    
    return (bins, calc_wpli.run(s_xy))

def run_psi(data_1, data_2, nperseg_outer, fs, nperseg_inner, nfft, window, pad_type, f_min, f_max, f_step_sz = 1, normalize = True):
    """
    Calculates the phase slope index between two signals. Assumes data_1 and data_2 to be from time domain.
  
    :param data_1: First dataset from time domain; vector of samples.
    :param data_2: Second dataset from time domain; vector of samples.
    :param nperseg_outer: Outer window size. If normalize = False, this parameter is not used
    :param fs: Sampling frequency.
    :param nperseg_inner: Inner window size.
    :param nfft: fft window size.
    :param window: FFT window type. Supported window types are listed at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.
    :param pad_type: Padding type, currently only "zero" padding is supported.
    :param f_min: Minimum frequence for the evaluated interval.
    :param f_max: Maximum frequence for the evaluated interval.
    :param f_step_sz: Frequency step size in the evaluated interval.
    :param normalize: Determines whether to normalize by dividing through the variance
    
    :return: Connectivity between data1 and data 2 measured using the phase slope index.
    """
      
    if (normalize == True):
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
    Calculates the imaginary coherency between two signals. Assumes data_1 and data_2 to be from time domain.
  
    :param data_1: First dataset from time domain; vector of samples.
    :param data_2: Second dataset from time domain; vector of samples.
    :param fs: Sampling frequency.
    :param nperseg: Size of individual segments in fft.
    :param nfft: fft window size.
    
    :return: (bins, conn) - Frequency bins and corresponding imaginary coherency values.
    """
        
    (bins, coh) = run_cc(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, np.imag(coh))
    
def run_msc(data_1, data_2, fs, nperseg, nfft):
    """
    Calculates the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from time domain.
  
    :param data_1: First dataset from time domain; vector of samples.
    :param data_2: Second dataset from time domain; vector of samples.
    :param fs: Sampling frequency.
    :param nperseg: Size of individual segments in fft.
    :param nfft: fft window size.
    
    :return: (bins, conn) - Frequency bins and corresponding magnitude squared coherence values.
    """
    
    (bins, coh) = run_cc(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, np.square(np.abs(coh)))
    
def run_cc(data_1, data_2, nperseg, pad_type, fs, nfft, window):
    """
    Calculate complex coherency from time domain data.
    
    :param data_1: data set X from time domain.
    :param data_2: data set Y from time domain.
    :param nperseg: number of samples in a fft segment.
    :param pad_type: padding type to be applied.
    :param fs: Sampling frequency.
    :param nfft: Size of fft window.
    :param window: FFT window type. Supported window types are listed at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html.
    
    :return: Complex coherency.
    """

    seg_data_1 = misc._segment_data(data_1, nperseg, pad_type)
    seg_data_2 = misc._segment_data(data_2, nperseg, pad_type)
    
    seg_data_1 = seg_data_1[:seg_data_2.shape[0], :]
    seg_data_2 = seg_data_2[:seg_data_1.shape[0], :]

    (bins, f_data_1) = misc._calc_FFT(seg_data_1, fs, nfft, window)
    (_,    f_data_2) = misc._calc_FFT(seg_data_2, fs, nfft, window)

    return (bins, fd.run_cc(f_data_1, f_data_2))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
