'''
Created on Jun 11, 2020

@author: voodoocode
'''

import numpy as np

def run(data):
    """
    Calculates the magnitude squared coherency between two signals. Assumes data_1 and data_2 to be from the complex frequency domain.
        
    @param data_1: Complex coherency values; vector of samples
    @param magnitude squared coherency
    """

    return np.square(np.abs(data))








