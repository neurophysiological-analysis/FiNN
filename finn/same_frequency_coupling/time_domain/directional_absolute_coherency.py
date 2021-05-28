'''
Created on Jun 12, 2018

@author: voodoocode
'''

import numpy as np

import finn.same_frequency_coupling.time_domain.complex_coherency as td_cc
import finn.same_frequency_coupling.__calc_directionalized_absolute_coherence as calc
    
def run(data_1, data_2, fmin, fmax, fs, nperseg, nfft, return_signed_conn = True, minimal_angle_thresh = 2):
    """
    Calculates the directional absolute coherence between two signals. Assumes data_1 and data_2 to be from time domain.
    
    As the coherence is similar to the Fouier Transform of the Pearson correlation coefficient,
    the magnitude informs of the strength of the correlation and whereas the sign of the imaginary part informs on the direction.
    
    Important design decision: 
    - In case data_2 happens before data_1, the sign of the psi (used to gain directional information) is defined to be positive.
    - In case data_1 happens before data_2, the sign of the psi (used to gain directional information)  is defined to be negative.
    
    The sign of the imaginary part of the coherence is a sine with a frequency of f = 1 in [-180°, 180°].
    Naturally, there are two roots of this sine, one at 0° and another at -180°/180°. Around these
    root phase shifts, the calculated sign is proportionally more sensetive to noise in the signal. 
    Therefore, in case of phase shifts from [-thresh°, +thresh°] the amplitude is corrected to 0. 
    Furthermore, any same_frequency_coupling with a phase shift of ~0° is (mostly) indistingusihable from volume
    conduction effects.
    
    @param data_1: First dataset from time domain; vector of samples
    @param data_2: Second dataset from time domain; vector of samples
    @param fmin: Minimum frequency of the frequency range on which coherency gets evaluated.
    @param fmax: Maximum frequency of the frequency range on which coherency gets evaluated.
    @param fs: Sampling frequency
    @param nperseg: Size of individual segments in fft
    @param nfft: fft window size
    @param return_signed_conn: Flag whether the absolute coherence should be multiplied with [-1, 1] for directional information
    @param minimal_angle_thresh: The minimal angle (phase shift) to evaluate in this analysis. Any angle smaller than the angle defined by 
    minimal_angle_thresh is considered volume conduction and therefore replace with np.nan.
    
    @return (bins, conn) - Frequency bins and corresponding same_frequency_coupling values
    """
    
    (bins, coh) = td_cc.run(data_1, data_2, nperseg, "zero", fs, nfft, "hanning")
    
    return (bins, calc.run(coh, bins, fmin, fmax, return_signed_conn, minimal_angle_thresh))



