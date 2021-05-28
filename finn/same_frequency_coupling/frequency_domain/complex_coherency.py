'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np

def run(data_X, data_Y):
#def get_coherence_from_frequency_data(f_data_X, f_data_Y):
    """
    Calculate complex coherency from fft domain data.
    
    @param f_data_X: data set X from the complex frequency domain.
    @param f_data_Y: data set Y from the compley frequency domain.
    
    @return: Complex coherency
    """
    
    s_xx = np.conjugate(data_X) * data_X * 2
    s_yy = np.conjugate(data_Y) * data_Y * 2
    s_xy = np.conjugate(data_X) * data_Y * 2

    s_xx = np.mean(s_xx, axis = 0)
    s_yy = np.mean(s_yy, axis = 0)
    s_xy = np.mean(s_xy, axis = 0)

    return s_xy/np.sqrt(s_xx*s_yy)



