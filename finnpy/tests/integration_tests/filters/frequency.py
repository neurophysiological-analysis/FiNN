'''
Created on May 5, 2022

@author: voodoocode
'''

import unittest

import numpy as np
import random

import finnpy.filters.frequency as ff

import scipy.signal

random.seed(0)
np.random.seed(0)

class test_filters(unittest.TestCase):

    def test_filters(self):
        #Configure sample data
        channel_count = 1
        frequency = [random.randint(5, 50) for _ in range(channel_count)]
        data_range = np.arange(0, 10000)
        frequency_sampling = 200
        
        #Configure noise data
        frequency_noise = 50
        shared_noise_strength = 10
        random_noise_strength = 1
        
        #Generate some sample data
        raw_data = [None for _ in range(channel_count)]
        for ch_idx in range(channel_count):
            genuine_signal = np.sin(2 * np.pi * frequency[ch_idx] * data_range / frequency_sampling)
            shared_noise_signal = np.sin(2 * np.pi * frequency_noise * data_range / frequency_sampling) * shared_noise_strength
            random_noise_signal = np.random.random(len(data_range)) * random_noise_strength
            
            raw_data[ch_idx] = genuine_signal + shared_noise_signal + random_noise_signal
        raw_data = np.asarray(raw_data)
        
        #Filter data - butter
        filtered_butter_data = [None for _ in range(channel_count)]    
        for ch_idx in range(channel_count):
            filtered_butter_data[ch_idx] = ff.butter(raw_data[ch_idx], 1, 40, frequency_sampling, order = 7, zero_phase = True)
        filtered_butter_data = np.asarray(filtered_butter_data)
        
        #Filter data - fir
        filtered_fir_data = [None for _ in range(channel_count)]    
        for ch_idx in range(channel_count):
            filtered_fir_data[ch_idx] = ff.fir(raw_data[ch_idx], 52, 48, 0.1, frequency_sampling, ripple_pass_band = 1e-5, stop_band_suppression = 1e-7, fft_win_sz = frequency_sampling, pad_type = "zero")
        filtered_fir_data = np.asarray(filtered_fir_data)
        
        raw_psd = scipy.signal.welch(raw_data[0, :], fs = frequency_sampling, window = "hann", nperseg = frequency_sampling,
                                     noverlap = frequency_sampling//2, nfft = frequency_sampling, detrend = None)[1]
        bwf_psd = scipy.signal.welch(filtered_butter_data[0, :], fs = frequency_sampling, window = "hann", nperseg = frequency_sampling,
                                     noverlap = frequency_sampling//2, nfft = frequency_sampling, detrend = None)[1]
        fir_psd = scipy.signal.welch(filtered_fir_data[0, :], fs = frequency_sampling, window = "hann", nperseg = frequency_sampling,
                                     noverlap = frequency_sampling//2, nfft = frequency_sampling, detrend = None)[1]
        
        assert(raw_psd[50] > bwf_psd[50])
        assert(raw_psd[50] > fir_psd[50])
        assert(bwf_psd[50] > fir_psd[50])

if __name__ == '__main__':
    unittest.main()


