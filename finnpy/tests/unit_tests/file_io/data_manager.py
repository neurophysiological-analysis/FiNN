'''
Created on Jun 3, 2020

@author: voodoocode
'''

import random
import numpy as np
import unittest

import finnpy.file_io.data_manager as dm
import shutil

class test_data_manager(unittest.TestCase):
    def test_data_manager(self):
        #Configure sample data
        channel_count = 32
        frequency = [random.randint(5, 50) for _ in range(channel_count)]
        data_range = np.arange(0, 10000)
        frequency_sampling = 200
        
        #Generate some sample data
        epoch_count = 10
        state_count = 2
        
        raw_data = [[[None for _ in range(channel_count)] for _ in range(epoch_count)] for _ in range(state_count)]
        for (state_idx, _) in enumerate(range(state_count)):
            for (epoch_idx, _) in enumerate(range(epoch_count)):
                for ch_idx in range(channel_count):
                    genuine_signal = np.sin(2 * np.pi * frequency[ch_idx] * data_range / frequency_sampling)
                        
                    raw_data[state_idx][epoch_idx][ch_idx] = genuine_signal
        
        #Save data
        dm.save(raw_data, ".", max_depth = 2, legacy_mode = True, legacy_params = {"var_name" : "test_file", "ending" : ".pkl"})
        
        #Load data
        loaded_data = dm.load("test_file", legacy_mode = True)
        shutil.rmtree("./test_file", ignore_errors=True)
        
        assert((np.asarray(raw_data) == np.asarray(loaded_data)).all())
        
        #Save data
        dm.save(raw_data, "test_file", max_depth = 2)
        
        #Load data
        loaded_data = dm.load("test_file")
        
        assert((np.asarray(raw_data) == np.asarray(loaded_data)).all())
        
        shutil.rmtree("./test_file", ignore_errors=True)
