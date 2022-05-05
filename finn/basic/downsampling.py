'''
Created on Jun 2, 2020

:author: voodoocode
'''

import numpy as np
import scipy.signal
import gc

def run(data, src_freq, tgt_freq):
    """
    Downsamples a signal (data) from src_freq to tgt_freq.
    
    :param data: The data to be downsampled. 
    :param src_freq: The original frequency of the signal.
    :param tgt_freq: The new frequency of the signal.
    :return: Downsampled data
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
    Returns the next power of two.
    
    :param value: The value to start the search.
    
    :return: A power of two >= value.
    
    Source: https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
    """
    
    
    return 1<<(value-1).bit_length()    
