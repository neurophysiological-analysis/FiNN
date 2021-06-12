'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np

def run(s_xy):
    
    divident = np.sum(np.abs(np.imag(s_xy)) * np.sign(np.imag(s_xy)), axis = 0)
    divisor = np.sum(np.abs(np.imag(s_xy)), axis = 0)
    divisor[divisor == 0] = np.nan #Zeros in the divisor are replaced with np.nan avoid divide by zero warnings/errors
    
    return divident/divisor
