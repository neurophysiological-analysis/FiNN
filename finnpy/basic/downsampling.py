"""
Created on Jun 2, 2020.

:author: voodoocode
"""

import numpy as np
import scipy.signal
import gc

def run(data, src_freq, tgt_freq):
    """
    Downsample a signal (data) from src_freq to tgt_freq.
    
    Parameters
    ----------
    data : np.ndarray, shape(duration * src_freq,)
           Data to be downsampled.
    src_freq : int 
               The original frequency of the signal.
    tgt_freq: int
              The new frequency of the signal.
              
    Returns
    -------
    Downsampled data : np.ndarray, shape(duration * tgt_freq,)
    """
    paddedData = np.zeros((_nextPowerTwo(len(data))))
    paddedData[0:len(data)] = data
    
    padTgtSampNum = int(float(len(paddedData)) / float(src_freq) * float(tgt_freq))
    tgtSampNum = int(float(len(data)) / float(src_freq) * float(tgt_freq))
    paddedData = scipy.signal.resample(paddedData, padTgtSampNum)
    
    data = paddedData[0:tgtSampNum]
    paddedData = None
    gc.collect()
    
    return data

def _nextPowerTwo(value):
    """
    Return the next power of two.
    
    Parameters
    ----------
    value: int
           The value to start the search.
           
    Returns
    -------
    int
        A power of two >= value.
    
    Source: https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
    """
    return 1 << (value - 1).bit_length()    
