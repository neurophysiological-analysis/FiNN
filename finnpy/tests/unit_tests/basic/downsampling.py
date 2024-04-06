'''
Created on Jun 2, 2020

@author: voodoocode
'''

import unittest
import numpy as np
import random

import finnpy.basic.downsampling as ds        

class test_downsampling(unittest.TestCase):
    def test_downsampling(self):
               
        #Configure sample data
        channel_count = 1
        frequency = [random.randint(10, 25) for _ in range(channel_count)]
        data_range = np.arange(0, 10000000)
        frequency_sampling = 10000000
        frequency_downsampled = 10000
        
        #Generate some sample data
        raw_data = [None for _ in range(channel_count)]
        for idx in range(channel_count):
            genuine_signal = np.sin(2 * np.pi * frequency[idx] * data_range / frequency_sampling)

            raw_data[idx] = genuine_signal
        raw_data = np.asarray(raw_data)
            
        ds_data = ds.run(raw_data[0], frequency_sampling, frequency_downsampled)
        
        reference_data = (raw_data[0, np.arange(0, len(data_range), frequency_sampling/frequency_downsampled, dtype = int)])[100:-100]
        test_ds_data = ds_data[100:-100]
        
        assert((np.abs(test_ds_data - reference_data) < 0.1).all())
        
        
