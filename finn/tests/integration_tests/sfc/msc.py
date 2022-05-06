'''
Created on May 5, 2022

@author: voodoocode
'''

import unittest

import numpy as np
np.random.seed(0)

import finn.data.paths as paths

import finn.sfc.td as sfc_td
import finn.sfc.fd as sfc_fd
import finn.sfc._misc as sfc_misc

import finn.sfc.cd as sfc_cohd

class test_sfc(unittest.TestCase):
    def test_sfc(self):
        data = np.load(paths.fct_sfc_data)
        frequency_sampling = 5500
        frequency_peak = 30
        
        noise_weight = 0.2
        
        phase_min = -90
        phase_max = 270
        phase_step = 4
        
        frequency_target = 30
        nperseg = frequency_sampling
        nfft = frequency_sampling
        
        #Generate data
        offset = int(np.ceil(frequency_sampling/frequency_peak))
        loc_data = data[offset:]
        signal_1 = np.zeros((loc_data).shape)
        signal_1 += loc_data
        signal_1 += np.random.random(len(loc_data)) * noise_weight

        for phase_shift in np.arange(phase_min, phase_max, phase_step):
            loc_offset = offset - int(np.ceil(frequency_sampling/frequency_peak * phase_shift/360))
            loc_data = data[(loc_offset):]
            signal_2 = np.zeros(loc_data.shape)
            signal_2 += loc_data
            signal_2 += np.random.random(len(loc_data)) * noise_weight
            
            conn_value_td = self.calc_from_time_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft, frequency_target)
            conn_value_fd = self.calc_from_frequency_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft, frequency_target)
            conn_value_coh = self.calc_from_coherency_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft, frequency_target)
            
            if (np.isnan(conn_value_td) == False and np.isnan(conn_value_fd) == False and np.isnan(conn_value_coh) == False):
                assert(conn_value_td == conn_value_fd and conn_value_td == conn_value_coh)
        
        
    def calc_from_time_domain(self, signal_1, signal_2, frequency_sampling, nperseg, nfft, frequency_target):
        return sfc_td.run_msc(signal_1, signal_2, frequency_sampling, nperseg, nfft)[1][frequency_target]
    
    def calc_from_frequency_domain(self, signal_1, signal_2, frequency_sampling, nperseg, nfft, frequency_target):
        seg_data_X = sfc_misc._segment_data(signal_1, nperseg, pad_type = "zero")
        seg_data_Y = sfc_misc._segment_data(signal_2, nperseg, pad_type = "zero")
    
        (bins, fd_signal_1) = sfc_misc._calc_FFT(seg_data_X, frequency_sampling, nfft, window = "hanning")
        (_,    fd_signal_2) = sfc_misc._calc_FFT(seg_data_Y, frequency_sampling, nfft, window = "hanning")
        
        return sfc_fd.run_msc(fd_signal_1, fd_signal_2, bins)[1][[np.argmin(np.abs(bins - frequency_target))]]
            
    def calc_from_coherency_domain(self, signal_1, signal_2, frequency_sampling, nperseg, nfft, frequency_target):
        (bins, coh) = sfc_td.run_cc(signal_1, signal_2, nperseg, "zero", frequency_sampling, nfft, "hanning")
        
        return sfc_cohd.run_msc(coh)[np.argmin(np.abs(bins - frequency_target))]
            
if __name__ == '__main__':
    unittest.main()



