'''
Created on Jun 2, 2020

:author: voodoocode
'''

import numpy as np

def run(data):
    """
    Applies common average re-referencing.
    
    :param data: Channels x samples.
    
    :return: Common average re-referenced data.
    """
    I = np.identity(n = len(data))
    v = np.ones((len(data))) / len(data)
    
    projection_matrix = np.subtract(I, np.dot(v, np.transpose(v)))
    
    return np.dot(projection_matrix, data)
