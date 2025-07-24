"""
Created on Feb 2, 2023.

@author: voodoocode
"""

import pyexcel
import numpy as np

def run(path):
    """
    Get the first sheed of a pyexcel readable file.
    
    Parameters
    ----------
    path : str
           Path to the table file.
           
    Returns
    -------
    np.ndarray
        Read data.
    """
    data = pyexcel.get_sheet(file_name = path)
    return np.asarray(data)
