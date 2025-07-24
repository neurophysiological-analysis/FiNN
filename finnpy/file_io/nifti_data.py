'''
Created on Feb 2, 2023.

@author: voodoocode
'''

import nibabel

def run(path):
    """
    Load nifti data.
    
    Parameters
    ----------
    path : str
           Path to the *.nifti file.
    """
    return nibabel.load(path)

