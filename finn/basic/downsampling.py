'''
Created on Jun 2, 2020

@author: voodoocode
'''

import numpy as np
import scipy.signal
import gc

def run(data, srcFreq, tgtFreq):
    
    paddedData = np.zeros((nextPowerTwo(len(data))))
    paddedData[0:len(data)] = data
    
    padTgtSampNum = int(float(len(paddedData)) / float(srcFreq) * float(tgtFreq))
    tgtSampNum = int(float(len(data)) / float(srcFreq) * float(tgtFreq))
    paddedData = scipy.signal.resample(paddedData, padTgtSampNum)
    
    data = paddedData[0:tgtSampNum]
    paddedData = None
    gc.collect()
    
    return data

def nextPowerTwo(value):
    """
    Returns the 'next' power of two.
    
    @param value: The value to start the search.
    
    @return: A power of two >= value
    
    Source: https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
    """
    
    
    return 1<<(value-1).bit_length()    