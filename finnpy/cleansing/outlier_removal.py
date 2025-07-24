"""
Created on Jun 12, 2018.

This module implements code to identify (and remove) outliers in a data-set.

:author: voodoocode
"""

import scipy.stats
import numpy as np


def run(data, ref, max_std_dist = 2., min_samp_cnt = 5, axis = 0):
    """
    Remove any sample more distant from the mean than max_std_dist standard deviations. Terminates if either all samples are within the threshold or if the minimal sample count defined by min_samp_cnt is reached.
   
    Parameters
    ----------
    data : np.ndarray, shape(samples, )
           To be cleaned data.
    ref: np.ndarray, shape(samples, )
         Reference data to clean the data variable by. In most cases equal to data.
    max_std_dist : float
                 Threshold for outlier detection. Number of standard deviations permissible.
    min_samp_cnt : int
                   Minimal viable sample count. Terminates if reduced below this number or the current iteration would reduce below this number.
    axis: int
          Axis on which to evaluate the data object. 
    
    Returns
    -------
    np.ndarray(samples - filtered_samples,)
        Filtered array without outlier more different than n standard deviations.
    """
    data = np.asarray(data)
    
    if (data.shape[axis] < min_samp_cnt):
        return data
    
    while (True):
        zVals = scipy.stats.zscore(ref)
        
        if ((np.abs(zVals) >= max_std_dist).any()):
            if (len(zVals) - len(np.argwhere(np.abs(zVals) >= max_std_dist).squeeze(1)) <= min_samp_cnt):
                return data
            badPts = np.argwhere(np.abs(zVals) >= max_std_dist).squeeze(1)
            ref = np.delete(ref, badPts)
            data = np.delete(data, badPts, axis = axis)
        else:
            break
    
    return data
  
    
    
