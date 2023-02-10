'''
Created on Feb 2, 2023

@author: voodoocode
'''

import pyexcel
import numpy as np

def run(path):
    data = pyexcel.get_sheet(file_name = path)
    return np.asarray(data)

