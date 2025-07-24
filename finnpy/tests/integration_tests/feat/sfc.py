'''
Created on May 5, 2022

@author: voodoocode
'''

import unittest
import numpy as np

import finnpy.feat.sfc as sfc  # @UnresolvedImport
import finnpy.demo.functionality.sfc.gen_demo_data as gen_demo_data  # @UnresolvedImport

np.random.seed(0)

class test_sfc_all(unittest.TestCase):

    def mag_sq_coh(self, data, _0, _1, frequency_tgt, _2, _3):
        return sfc.msc_cc(data)[frequency_tgt]
    def img_coh(self, data, _0, _1, frequency_tgt, _2, _3):
        return sfc.ic_cc(data)[frequency_tgt]
    def dac_coh(self, data, _0, bins, frequency_tgt, _1, _2, freq_range = 5, min_phase_diff = 10, volume_conductance_ratio = 0.4):
        dac_range = sfc.dac_cc(data, bins, frequency_tgt - freq_range, frequency_tgt + freq_range + 1,
                                 return_signed_conn = True, minimal_angle_thresh = min_phase_diff, 
                                 volume_conductance_ratio = volume_conductance_ratio)
        return (0 if (np.isnan(dac_range)) else dac_range)
    def psi_coh(self, _0, data, bins, frequency_tgt, _1, _2, freq_range = 5):
        return sfc.psi_cc(data, bins, frequency_tgt - freq_range, frequency_tgt + freq_range + 1)
    def wpli_coh(self, _0, _1, _2, frequency_tgt, data1, data2):
        win_sz = 5500
        
        s_xy = list()
        for block_start in np.arange(0, np.min([len(data1), len(data2)]) - win_sz, win_sz):
            loc_data1 = data1[block_start:(block_start + win_sz)]
            loc_data2 = data2[block_start:(block_start + win_sz)]
            
            seg_data_X = sfc._segment_data(loc_data1, win_sz, "zero")  # pylint: disable=protected-access
            seg_data_Y = sfc._segment_data(loc_data2, win_sz, "zero")  # pylint: disable=protected-access
        
            (_, f_data_X) = sfc._calc_FFT(seg_data_X, 5500, win_sz, "hann")  # pylint: disable=protected-access
            (_,    f_data_Y) = sfc._calc_FFT(seg_data_Y, 5500, win_sz, "hann")  # pylint: disable=protected-access
        
            s_xy.append((np.conjugate(f_data_X[0, :]) * f_data_Y[0, :] * 2))
    
        s_xy = np.asarray(s_xy)
        
        return sfc.wpli_cc(s_xy)[frequency_tgt]

    def single_channel_shift(self, noise_weight,
                             phase_min, phase_max, phase_step,
                             pad_type, window,
                             methods = [mag_sq_coh, img_coh, dac_coh, psi_coh],
                             frequency_peak = None, data = None, frequency_sampling = 3000):
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
            
            (bins, comp_coh) = sfc.cc_td(signal_1, signal_2, nperseg, pad_type, fs, nfft, window)
            
            signal_1_step_sz = len(signal_1)/10
            signal_2_step_sz = len(signal_2)/10
            comp_coh2 = [sfc.cc_td(signal_1[int(idx * signal_1_step_sz):int((idx + 1) * signal_1_step_sz)],
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
        minimum_frequency = 13
        maximum_frequency = 27
        
        frequency_sampling = 3000
        time_s = 120
        offset_s = 1
        signal_length_samples = int(frequency_sampling * (time_s + offset_s * 2)) 
        data = gen_demo_data.gen_wn_signal(minimum_frequency, maximum_frequency, frequency_sampling, signal_length_samples)
        frequency_peak = (maximum_frequency + minimum_frequency)/2
        frequency_tgt = 20
        results = self.single_channel_shift(noise_weight, phase_min, phase_max, phase_step, pad_type, window,
                                            methods, frequency_peak = frequency_tgt, data = data)
        
        assert(np.min(results[0]) > 0.9) #check MSC
        assert(np.max(results[1]) > 0.9) #check IC
        assert(np.max(results[2]) > 0.9) #check wPLI
        assert(np.max(results[3]) > 0.9) #check PSI
        assert((np.max(np.asarray(results[4])[np.argwhere(np.asarray(results[4]) < 0).squeeze(1)]) < -0.9) and
               (np.min(np.asarray(results[4])[np.argwhere(np.asarray(results[4]) > 0).squeeze(1)]) > 0.9)) #check DAC
        
if __name__ == '__main__':
    unittest.main()














