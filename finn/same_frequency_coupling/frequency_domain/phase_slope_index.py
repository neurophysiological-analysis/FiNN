'''
Created on Dec 29, 2020

@author: voodoocode
'''

import finn.same_frequency_coupling.__calc_phase_slope_index as calc
import finn.same_frequency_coupling.frequency_domain.complex_coherency as fd_cc

def run(data_X, data_Y, bins, f_min, f_max, f_step_sz = 1):
    
    data_coh = list()
    for (outer_window_idx, _) in enumerate(data_X):
        data_coh.append(fd_cc.run(data_X[outer_window_idx], data_Y[outer_window_idx]))
    
    return calc.run(data_coh, bins, f_min, f_max, f_step_sz)








