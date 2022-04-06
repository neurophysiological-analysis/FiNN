'''
Created on Jun 12, 2018

This module implements code to identify (and remove) outliers in a data-set.

:author: voodoocode
'''

import scipy.stats
import numpy as np


def run(data, ref, max_std_dist = 2, min_samp_cnt = 5, axis = 0):
    """
    Removes any sample more distant from the mean than max_std_dist standard deviations. Terminates if either all samples are within the threshold or if the minimal sample count defined by min_samp_cnt is reached.
    
    :param data: Input data. 
    :param ref: Single dimensional reference data.
    :param maxStdDist: Threshold for outlier detection. Number of standard deviations permissible.
    :param min_samp_cnt: Minimal viable sample count. Terminates if reduced below this number or the current iteration would reduce below this number.
    :param axis: Axis on which to evaluate the data object 
    
    :return: Filtered array without outlier more different than n standard deviations.
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
  
    
    
