'''
Created on Dec 29, 2020

@author: voodoocode
'''

import finn.same_frequency_coupling.__calc_phase_slope_index as calc

def run(data, bins, f_min, f_max, f_step_sz = 1):
    return calc.run(data, bins, f_min, f_max, f_step_sz)