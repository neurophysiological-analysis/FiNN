'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np

def run(s_xy):
    return np.sum(np.abs(np.imag(s_xy)) * np.sign(np.imag(s_xy)), axis = 0)/np.sum(np.abs(np.imag(s_xy)), axis = 0)
