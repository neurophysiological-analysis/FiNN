'''
Created on Dec 29, 2020

This module implements the phase slope index.

@author: voodoocode
'''

import numpy as np

def run(data, bins, f_min, f_max, f_step_sz = 1):
    """
    
    Calculates the phase slope index (psi) from a list of complex coherency data.
    
    :param data: List of complex coherency data.
    :param bins: Frequency bins of the complex coherency data.
    :param f_min: Minimum frequency of interest.
    :param f_max: Maximum frequency of interest.
    :param f_step_size: Frequency step size.
    
    :return: Returns the sfc measured as psi computed from data.
    
    """
    
    f_min_idx = np.argmin(np.abs(bins - f_min))
    f_max_idx = np.argmin(np.abs(bins - f_max))
    
    psi = np.zeros((len(data)), dtype = np.complex64)
    for (psi_idx, comp_coh) in enumerate(data):
        for freq_idx in range(f_min_idx, f_max_idx, 1):
            psi[psi_idx] += np.conjugate(comp_coh[freq_idx])*comp_coh[freq_idx + f_step_sz]
        psi[psi_idx] = np.imag(psi[psi_idx])
    psi = np.asarray(psi.real, dtype = np.float32)
    
    if (len(data) > 1):
        var = 0
        for idx in range(len(data)):
            var += np.var(np.concatenate((psi[:idx], psi[(idx +1):])))
        var /= len(data)
        
        return np.mean(psi)/(np.sqrt(var) * 2)
    else:
        return psi[0]






