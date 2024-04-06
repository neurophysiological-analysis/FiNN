'''
Created on Dec 29, 2020

This module implements the weighted phase lag index.

@author: voodoocode
'''

import numpy as np

def run(s_xy):
    """
    
    Calculates the weighted phase lag index (wPLI) from a list of complex coherency data.
    
    :param s_xy: complex coherency data.
    
    :return: Returns the sfc measured as wPLI computed from data.
    
    """
    
    divident = np.sum(np.abs(np.imag(s_xy)) * np.sign(np.imag(s_xy)), axis = 0)
    divisor = np.sum(np.abs(np.imag(s_xy)), axis = 0)
    divisor[divisor == 0] = np.nan #Zeros in the divisor are replaced with np.nan avoid divide by zero warnings/errors
    
    return divident/divisor
