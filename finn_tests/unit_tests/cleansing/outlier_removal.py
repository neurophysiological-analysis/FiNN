'''
Created on Jun 3, 2020

@author: voodoocode
'''

import numpy as np
import random
import scipy.stats

import finn.artifact_rejection.outlier_removal as om
import unittest

class test_outlier_removal(unittest.TestCase):
    def test_outlier_removal(self):
        #Configure sample data
        channel_count = 32
        data_range = 100
        
        #Configure niose
        noise_count = int(data_range * 0.05)
        
        #Generate sample data
        raw_data = [None for _ in range(channel_count)]
        for ch_idx in range(channel_count):
            raw_data[ch_idx] = np.random.normal(0, 2, data_range)
            for noise_idx in [random.randint(0, data_range - 1) for _ in range(noise_count)]:
                raw_data[ch_idx][noise_idx] = np.random.randint(1, 10)
        
        filtered_data = [None for _ in range(channel_count)]
        for ch_idx in range(channel_count):
            filtered_data[ch_idx] = om.run(raw_data[ch_idx], raw_data[ch_idx], maxStdDist = 2, minSampCnt = 0)
        
        assert(np.max(np.abs(scipy.stats.zscore(filtered_data[0]))) < 2)

