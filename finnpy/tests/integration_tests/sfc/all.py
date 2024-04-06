'''
Created on May 5, 2022

@author: voodoocode
'''

'''
Created on May 5, 2022

@author: voodoocode
'''

import unittest
import numpy as np

import finnpy.sfc.cd as cohd
import finnpy.sfc._misc as misc
import finnpy.sfc.td as td

import finnpy.data.paths as paths

np.random.seed(0)

class test_sfc_all(unittest.TestCase):

    def mag_sq_coh(self, data, _0, _1, frequency_tgt, _2, _3):
        return cohd.run_msc(data)[frequency_tgt]
    def img_coh(self, data, _0, _1, frequency_tgt, _2, _3):
        return cohd.run_ic(data)[frequency_tgt]
    def dac_coh(self, data, _0, bins, frequency_tgt, _1, _2, freq_range = 5, min_phase_diff = 10, volume_conductance_ratio = 0.4):
        dac_range = cohd.run_dac(data, bins, frequency_tgt - freq_range, frequency_tgt + freq_range + 1,
                                 return_signed_conn = True, minimal_angle_thresh = min_phase_diff, 
                                 volume_conductance_ratio = volume_conductance_ratio)
        return (0 if (np.isnan(dac_range)) else dac_range)
    def psi_coh(self, _0, data, bins, frequency_tgt, _1, _2, freq_range = 5):
        return cohd.run_psi(data, bins, frequency_tgt - freq_range, frequency_tgt + freq_range + 1)
    def wpli_coh(self, _0, _1, _2, frequency_tgt, data1, data2):
        win_sz = 5500
        
        s_xy = list()
        for block_start in np.arange(0, np.min([len(data1), len(data2)]) - win_sz, win_sz):
            loc_data1 = data1[block_start:(block_start + win_sz)]
            loc_data2 = data2[block_start:(block_start + win_sz)]
            
            seg_data_X = misc._segment_data(loc_data1, win_sz, "zero")
            seg_data_Y = misc._segment_data(loc_data2, win_sz, "zero")
        
            (_, f_data_X) = misc._calc_FFT(seg_data_X, 5500, win_sz, "hann")
            (_,    f_data_Y) = misc._calc_FFT(seg_data_Y, 5500, win_sz, "hann")
        
            s_xy.append((np.conjugate(f_data_X[0, :]) * f_data_Y[0, :] * 2))
    
        s_xy = np.asarray(s_xy)
        
        return cohd.run_wpli(s_xy)[frequency_tgt]

    def single_channel_shift(self, noise_weight,
                             phase_min, phase_max, phase_step,
                             pad_type, window,
                             methods = [mag_sq_coh, img_coh, dac_coh, psi_coh],
                             frequency_peak = 30, path = "", frequency_sampling = 5500):
        
        if (path[-5] == "0"):
            data = np.load(path)[3, :] #Channel with 'strongest' beta activity @30Hz
        elif(path[-5] == "1"):
            data = np.load(path)[3, :] #Channel with 'strongest' beta activity @20Hz
        elif(path[-5] == "2"):
            data = np.load(path)[1, :] #Channel with 'strongest' beta activity @20Hz
        else:
            raise AssertionError
        offset = int(np.ceil(frequency_sampling/frequency_peak))
        
        #overwriting paramters with sampling frequency from loaded data
        nperseg = frequency_sampling
        fs = frequency_sampling
        nfft = frequency_sampling
        
        #Data container
        features = list()
        for _ in methods:
            features.append(list())
        
        #Generate data
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
            
            (bins, comp_coh) = td.run_cc(signal_1, signal_2, nperseg, pad_type, fs, nfft, window)
            
            signal_1_step_sz = len(signal_1)/10
            signal_2_step_sz = len(signal_2)/10
            comp_coh2 = [td.run_cc(signal_1[int(idx * signal_1_step_sz):int((idx + 1) * signal_1_step_sz)],
                                   signal_2[int(idx * signal_2_step_sz):int((idx + 1) * signal_2_step_sz)],
                                   nperseg, pad_type, fs, nfft, window)[1] for idx in range(10)]
            
            for (method_idx, method) in enumerate(methods):
                features[method_idx].append(method(comp_coh, comp_coh2, bins, frequency_peak, signal_1, signal_2))
        return features

    def test_sfc_all(self):
        #Signal configuration
        window = "hann"
        pad_type = "zero"
        frequency_tgt_shift = np.concatenate((np.arange(-8, 8, 1/10), [8])); sigma = 25
        signal_amplitude_scaling = 10000
        signal_amplitde_helper = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.arange(-len(frequency_tgt_shift)/2, len(frequency_tgt_shift)/2+1, 1/signal_amplitude_scaling) - 0)**2 / (2 * sigma**2))
        signal_amplitude = [signal_amplitde_helper[int(loc_signal_amplitude_idx * signal_amplitude_scaling + signal_amplitude_scaling/2)] for (loc_signal_amplitude_idx, _) in enumerate(frequency_tgt_shift)]
        signal_amplitude = np.asarray(signal_amplitude)
        signal_amplitude *= sigma
        
        noise_weight = 0.2
        
        #Phase range
        phase_min = -270
        phase_max = 270
        phase_step = 2
            
        #Select methods
        methods = [self.mag_sq_coh, self.img_coh, self.wpli_coh, self.psi_coh, self.dac_coh]
            
        #demo file
        demo_file = paths.per_sfc_data_0
        frequency_tgt = 30
        results = self.single_channel_shift(noise_weight, phase_min, phase_max, phase_step, pad_type, window,
                                            methods, frequency_peak = frequency_tgt, path = demo_file)
        
        assert(np.min(results[0]) > 0.9) #check MSC
        assert(np.abs(np.mean(results[1])) < 0.01) #check IC
        assert((np.asarray(results[2][-10:]) == 1).all() and (np.asarray(results[2][:10]) == -1).all()) #check wPLI
        assert((np.asarray(results[3][-10:]) < -0.94).all() and (np.asarray(results[3][:10]) > 0.94).all()) #check PSI
        assert((np.asarray(results[4][-10:]) < -0.98).all() and (np.asarray(results[4][:10]) > 0.99).all()) #check DAC
        
if __name__ == '__main__':
    unittest.main()














