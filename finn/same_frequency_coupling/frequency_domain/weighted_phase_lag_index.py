'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np
import finn.same_frequency_coupling.__misc as misc
import finn.same_frequency_coupling.__calc_weighted_phase_lag_index as calc

def run(data_X, data_Y):
    s_xy = list()
    for block_idx in np.arange(len(data_X)):
        s_xy.append((np.conjugate(data_X[block_idx]) * data_Y[block_idx] * 2))

    s_xy = np.asarray(s_xy)
    
    return calc.run(s_xy)