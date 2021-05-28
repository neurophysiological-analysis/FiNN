'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np
import finn.same_frequency_coupling.frequency_domain.complex_coherency as fd_cc

def run(data_1, data_2):
    """
    Calculates the imaginary coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    @param data_1: First dataset from the complex frequency domain; vector of samples
    @param data_2: Second dataset from the complex frequency domain; vector of samples
    @param return_signed_conn: Flag whether the absolute coherence should be multiplied with [-1, 1] for directional information
    """
    
    return np.imag(fd_cc.run(data_1, data_2))








