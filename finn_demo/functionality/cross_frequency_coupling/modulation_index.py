'''
Created on Jan 5, 2021

@author: voodoocode
'''

import numpy as np

import finn.cross_frequency_coupling.modulation_index as mi

def generate_high_frequency_signal(n, frequency_sampling, frequency_within_bursts, random_noise_strength, 
                                   offset, burst_count, burst_length):
    signal = np.random.normal(0, 1, n) * random_noise_strength

    for burst_start in np.arange(offset, n, n/burst_count):
        burst_end = burst_start + (burst_length/2)
        signal[int(burst_start):int(burst_end)] =  np.sin(2 * np.pi * frequency_within_bursts * np.arange(0, (int(burst_end) - int(burst_start))) / frequency_sampling)

    return signal

def main():
    #Configure sample data
    data_range = np.arange(0, 10000)
    frequency_sampling = 1000
    frequencies_between_bursts = [2, 5, 10, 15, 50]
    tgt_frequency_between_bursts = 10
    frequency_within_bursts = 200
    high_freq_frame_offsets = [0, 20, 40]
    
    #Configure noise data
    random_noise_strength = 0.3
    
    #Generate sample data
    burst_length = frequency_sampling / tgt_frequency_between_bursts
    burst_count = len(data_range) / frequency_sampling * tgt_frequency_between_bursts
    
    high_freq_signals = [generate_high_frequency_signal(len(data_range), frequency_sampling, frequency_within_bursts,
                                                        random_noise_strength, high_freq_frame_offset, burst_count, burst_length) for high_freq_frame_offset in high_freq_frame_offsets]
    low_freq_signals = [np.sin(2 * np.pi * frequency_between_bursts * data_range / frequency_sampling) for frequency_between_bursts in frequencies_between_bursts]
    
    scores = np.zeros((len(high_freq_signals), len(low_freq_signals)));
    for (high_freq_idx, high_freq_signal) in enumerate(high_freq_signals):
        for (low_freq_idx, low_freq_signal) in enumerate(low_freq_signals):
            scores[high_freq_idx, low_freq_idx] = mi.run(low_freq_signal, high_freq_signal, frequency_window_half_size = 20, frequency_step_width = 5)
    print(scores)

main()






