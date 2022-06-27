'''
Created on May 5, 2022

@author: voodoocode
'''

import unittest

import numpy as np
np.random.seed(0)

import finnpy.data.paths as paths

import finnpy.sfc.td as sfc_td
import finnpy.sfc.fd as sfc_fd
import finnpy.sfc._misc as sfc_misc

import finnpy.sfc.cd as sfc_cohd

class test_sfc(unittest.TestCase):
    def test_sfc(self):
        data = np.load(paths.fct_sfc_data)
        frequency_sampling = 5500
        frequency_peak = 30
        
        noise_weight = 0.2
        
        phase_shift = 153
        
        nperseg = frequency_sampling
        nfft = frequency_sampling
        
        #Generate data
        offset = int(np.ceil(frequency_sampling/frequency_peak))
        loc_data = data[offset:]
        signal_1 = np.zeros((loc_data).shape)
        signal_1 += loc_data
        signal_1 += np.random.random(len(loc_data)) * noise_weight
        
        loc_offset = offset - int(np.ceil(frequency_sampling/frequency_peak * phase_shift/360))
        loc_data = data[(loc_offset):]
        signal_2 = np.zeros(loc_data.shape)
        signal_2 += loc_data
        signal_2 += np.random.random(len(loc_data)) * noise_weight
        
        (bins, cc_td) = self.calc_from_time_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft)
        cc_fd = self.calc_from_frequency_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft)
        
        assert((cc_fd == cc_td).all())
        
    def calc_from_time_domain(self, signal_1, signal_2, frequency_sampling, nperseg, nfft, window = "hann", pad_type = "zero"):
        return sfc_td.run_cc(signal_1, signal_2, nperseg, pad_type, frequency_sampling, nfft, window)
    
    def calc_from_frequency_domain(self, signal_1, signal_2, frequency_sampling, nperseg, nfft, window = "hann", pad_type = "zero"):
        seg_data_X = sfc_misc._segment_data(signal_1, nperseg, pad_type)
        seg_data_Y = sfc_misc._segment_data(signal_2, nperseg, pad_type)
    
        (bins, fd_signal_1) = sfc_misc._calc_FFT(seg_data_X, frequency_sampling, nfft)
        (_,    fd_signal_2) = sfc_misc._calc_FFT(seg_data_Y, frequency_sampling, nfft)
        
        return sfc_fd.run_cc(fd_signal_1, fd_signal_2)
            
if __name__ == '__main__':
    unittest.main()



