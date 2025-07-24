"""
Created on Jun 2, 2020.

:author: voodoocode
"""

import numpy as np

def run(data):
    """
    Apply common average re-referencing.
    
    Parameters
    ----------
    data : np.ndarray, shape(ch_cnt, samp_cnt)
           Data.

    Returns
    -------
    np.ndarray, shape(ch_cnt, samp_cnt)
        Common average re-referenced data.
    """
    one = np.identity(n = len(data))
    v = np.ones((len(data))) / len(data)
    
    projection_matrix = np.subtract(one, np.dot(v, np.transpose(v)))
    
    return np.dot(projection_matrix, data)
