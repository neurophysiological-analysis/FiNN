'''
Created on Dec 29, 2020

This module provides different functions to estimate sfc between two signals from data presented in the coherency domain.

:author: voodoocode
'''

import numpy as np
import finnpy.sfc._misc as misc
import finnpy.sfc._wpli as calc_wpli
import finnpy.sfc._psi as calc_psi
import finnpy.sfc._dac as calc_dac

def run_dac(data, bins, fmin, fmax, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
    """
    Calculates the directional absolute coherence from complex coherency.
    
    As the coherence is similar to the Fouier Transform of the Pearson correlation coefficient, the magnitude informs of the strength of the correlation and whereas the sign of the imaginary part informs on the direction.
    
    Important design decision: 
    - In case data_2 happens before data_1, the sign of the psi (used to gain directional information) is defined to be positive.
    - In case data_1 happens before data_2, the sign of the psi (used to gain directional information)  is defined to be negative.
    
    The sign of the imaginary part of the coherence is a sine with a frequency of f = 1 in [-180°, 180°]. Naturally, there are two roots of this sine, one at 0° and another at -180°/180°. Around these root phase shifts, the calculated sign is proportionally more sensetive to noise in the signal. Therefore, in case of phase shifts from [-thresh°, +thresh°] the amplitude is corrected to 0. Furthermore, any same_frequency_coupling with a phase shift of ~0° is (mostly) indistingusihable from volume conduction effects.
    
    :param data: List of outer window complex coherency estimates calculated from two signals.
    :param bins: Frequency bins of the complex coherency data.
    :param fmin: Minimum frequency of the frequency range on which coherency gets evaluated.
    :param fmax: Maximum frequency of the frequency range on which coherency gets evaluated.
    :param return_signed_conn: Flag whether the absolute coherence should be multiplied with [-1, 1] for directional information
    :param minimal_angle_thresh: The minimal angle (phase shift) to evaluate in this analysis. Any angle smaller than the angle defined by minimal_angle_thresh is considered volume conduction and therefore replace with np.nan.
    :param volume_conductance_ratio: Defines the ratio of below threshold connectivity values to identify volume conductance.
    
    :return: Connectivity between data_1 and data_2 measured using the directionalized absolute coherence.
    """
        
    return calc_dac.run(data, bins, fmin, fmax, return_signed_conn, minimal_angle_thresh, volume_conductance_ratio)

def run_wpli(s_xy):
    """
    This function calculates the weighted phase lag index between two signals. 
    
    :param s_xy: List of complex coherency estimates calculated from two signals.
    
    :return: Connectivity estimated based on the provided complex coherency measured using the weighted lag slope index.
    """
    return calc_wpli.run(s_xy)

def run_psi(data, bins, f_min, f_max, f_step_sz = 1):
    """
    This function calculates the phase slope index between two signals. 
    
    :param data: List of outer window complex coherency estimates calculated from two signals.
    :param bins: Frequency bins of the sample data.
    :param f_min: Minimum frequence for the evaluated interval.
    :param f_max: Maximum frequence for the evaluated interval.
    :param f_step_sz: Frequency step size in the evaluated interval.
    
    :return: Connectivity estimated based on the provided complex coherency measured using the phase slope index.
    """
    return calc_psi.run(data, bins, f_min, f_max, f_step_sz)

def run_ic(data):
    """
    Calculates the imaginary coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    :param data: Complex coherency values; vector of samples.
    
    :return: Connectivity estimated based on the provided complex coherency measured via the imaginary coherence.
    """
    
    return np.imag(data)
    
def run_msc(data):
    """
    Calculates the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    :param data: Complex coherency values; vector of samples.
    
    :return: Connectivity estimated based on the provided complex coherency measured via the magnitude squared coherence.
    """

    return np.square(np.abs(data))
