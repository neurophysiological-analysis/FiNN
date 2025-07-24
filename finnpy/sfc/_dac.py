"""
Created on Dec 29, 2020.

This module implements the DAC metric.

@author: voodoocode
"""

import numpy as np

def run(coh, bins, fmin, fmax, return_signed_conn = True, minimal_angle_thresh = 10, volume_conductance_ratio = 0.3):
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
