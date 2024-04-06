'''
Created on Dec 29, 2020

This module provides different functions to estimate sfc between two signals from data presented in the frequency domain.

:author: voodoocode
'''

import numpy as np
import finnpy.sfc._psi as calc_psi
import finnpy.sfc._wpli as calc_wpli
import finnpy.sfc._dac as calc_dac

def run_dac(data_1, data_2, bins, fmin, fmax, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
    """
     Calculates the directional absolute coherence between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
    
    As the coherence is similar to the Fouier Transform of the Pearson correlation coefficient, the magnitude informs of the strength of the correlation and whereas the sign of the imaginary part informs on the direction.
    
    Important design decision: 
    - In case data_2 happens before data_1, the sign of the psi (used to gain directional information) is defined to be positive.
    - In case data_1 happens before data_2, the sign of the psi (used to gain directional information)  is defined to be negative.
    
    The sign of the imaginary part of the coherence is a sine with a frequency of f = 1 in [-180°, 180°]. Naturally, there are two roots of this sine, one at 0° and another at -180°/180°. Around these root phase shifts, the calculated sign is proportionally more sensetive to noise in the signal. Therefore, in case of phase shifts from [-thresh°, +thresh°] the amplitude is corrected to 0. Furthermore, any same_frequency_coupling with a phase shift of ~0° is (mostly) indistingusihable from volume conduction effects.
    
    :param data_1: First dataset from the complex frequency domain; vector of samples.
    :param data_2: Second dataset from the complex frequency domain; vector of samples.
    :param fmin: Minimum frequency of the frequency range on which coherency gets evaluated.
    :param fmax: Maximum frequency of the frequency range on which coherency gets evaluated.
    :param return_signed_conn: Flag whether the absolute coherence should be multiplied with [-1, 1] for directional information
    :param minimal_angle_thresh: The minimal angle (phase shift) to evaluate in this analysis. Any angle smaller than the angle defined by minimal_angle_thresh is considered volume conduction and therefore replace with np.nan.
    :param volume_conductance_ratio: Defines the ratio of below threshold connectivity values to identify volume conductance.
    
    :return: Connectivity between data_1 and data_2 measured using the directionalized absolute coherence.
    """
    
    coh = run_cc(data_1, data_2)
    
    return calc_dac.run(coh, bins, fmin, fmax, return_signed_conn, minimal_angle_thresh, volume_conductance_ratio)

def run_wpli(data_1, data_2):
    """
    This function calculates the weighted phase lag index between two signals. 
    
    :param data_1: First dataset from the frequency domain domain; vector of frequency domain samples.
    :param data_2: Second dataset from the frequency domain; vector of frequency domain samples.
    
    :return: Connectivity between data_1 and data_2 measured using the weighted phase lag index.
    """
    s_xy = list()
    for block_idx in np.arange(len(data_1)):
        s_xy.append((np.conjugate(data_1[block_idx]) * data_2[block_idx] * 2))

    s_xy = np.asarray(s_xy)
    
    return calc_wpli.run(s_xy)

def run_psi(data_1, data_2, bins, f_min, f_max, f_step_sz = 1):
    """
    Calculates the phase slope index between two signals. Assumes data_1 and data_2 to be from time domain.
  
    :param data1: First dataset from the frequency domain domain; vector of frequency domain samples.
    :param data2: Second dataset from the frequency domain; vector of frequency domain samples.
    :param bins: Frequency bins of the sample data.
    :param f_min: Minimum frequence for the evaluated interval.
    :param f_max: Maximum frequence for the evaluated interval.
    :param f_step_sz: Frequency step size in the evaluated interval.
    
    :return: Connectivity between data_1 and data_2 measured using the phase slope index.
    """
    
    data_coh = list()
    for (outer_window_idx, _) in enumerate(data_1):
        data_coh.append(run_cc(data_1[outer_window_idx], data_2[outer_window_idx]))
    
    return calc_psi.run(data_coh, bins, f_min, f_max, f_step_sz)

def run_ic(data_1, data_2):
    """
    Calculates the imaginary coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    :param data_1: First dataset from the complex frequency domain; vector of samples.
    :param data_2: Second dataset from the complex frequency domain; vector of samples.
    
    :return: Connectivity between data_1 and data_2 measured via imaginary coherence.
    """
    
    return np.imag(run_cc(data_1, data_2))

def run_msc(data_1, data_2, bins):
    """
    Calculates the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    :param data_1: First dataset from the complex frequency domain; vector of samples.
    :param data_2: Second dataset from the complex frequency domain; vector of samples.
    
    :return: Connectivity between data_1 and data_2 measured via imaginary coherence.
    """
    
    coh = run_cc(data_1, data_2)
    
    return (bins, np.square(np.abs(coh)))

def run_cc(data_1, data_2):
    """
    Calculate complex coherency from frequency domain data.
    
    :param f_data_1: data set X from the complex frequency domain.
    :param f_data_2: data set Y from the compley frequency domain.
    
    :return: Complex coherency
    """
    
    s_xx = np.conjugate(data_1) * data_1 * 2
    s_yy = np.conjugate(data_2) * data_2 * 2
    s_xy = np.conjugate(data_1) * data_2 * 2

    s_xx = np.mean(s_xx, axis = 0)
    s_yy = np.mean(s_yy, axis = 0)
    s_xy = np.mean(s_xy, axis = 0)

    return s_xy/np.sqrt(s_xx*s_yy)



