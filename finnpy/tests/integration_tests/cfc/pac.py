'''
Created on May 5, 2022

@author: voodoocode
'''

import numpy as np
import unittest

import finnpy.cfc.pac as pac

np.random.seed(0)

class test_pac(unittest.TestCase):
    
    ref_dmi_score = 0.940
    ref_plv_score = 0.633
    ref_mvl_score = 0.249
    ref_mi_score = 0.121
    
    def generate_high_frequency_signal(self, n, frequency_sampling, frequency_within_bursts, random_noise_strength,
                                       offset, burst_count, burst_length,
                                       sinusoidal = True):
        signal = np.random.normal(0, 1, n) * random_noise_strength
    
        if (sinusoidal == False):
            for burst_start in np.arange(offset, n, n/burst_count):
                burst_end = burst_start + (burst_length/2)
                signal[int(burst_start):int(burst_end)] =  np.sin(2 * np.pi * frequency_within_bursts * np.arange(0, (int(burst_end) - int(burst_start))) / frequency_sampling)
        else:
            signal += (np.sin(2 * np.pi * 200 * np.arange(len(signal)) / 1000)) * (np.sin(2 * np.pi * 10 * np.arange(len(signal)) / 1000) + 1)/2
    
        return signal
    
    def test_pac(self):
        data_range = np.arange(0,  500)
       
        frequency_sampling = 1000
        frequency_between_bursts = 10
        tgt_frequency_between_bursts = 10
        frequency_within_bursts = 200
        high_freq_frame_offset = 0
        
        random_noise_strength = 0.00
            
        #Generate sample data
        burst_length = frequency_sampling / tgt_frequency_between_bursts
        burst_count = len(data_range) / frequency_sampling * tgt_frequency_between_bursts
        
        high_freq_signal = self.generate_high_frequency_signal(len(data_range), frequency_sampling, frequency_within_bursts, random_noise_strength,
                                                               high_freq_frame_offset, burst_count, burst_length, True)
        low_freq_signal = np.sin(2 * np.pi * frequency_between_bursts * data_range / frequency_sampling)
        
        dmi_score = pac.run_dmi(low_freq_signal, high_freq_signal, phase_window_half_size = 4, phase_step_width = 2)[0]
        plv_score = pac.run_plv(low_freq_signal, high_freq_signal)
        mvl_score = pac.run_mvl(low_freq_signal, high_freq_signal)
        mi_score = pac.run_mi(low_freq_signal, high_freq_signal)
        
        #print(dmi_score, plv_score, mvl_score, mi_score)
        
        assert(np.abs(dmi_score - self.ref_dmi_score) < 0.01)
        assert(np.abs(plv_score - self.ref_plv_score) < 0.01)
        assert(np.abs(mvl_score - self.ref_mvl_score) < 0.01)
        assert(np.abs(mi_score - self.ref_mi_score) < 0.01)
        
        

if __name__ == '__main__':
    unittest.main()
