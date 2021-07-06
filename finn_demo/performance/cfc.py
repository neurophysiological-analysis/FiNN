'''
Created on Aug 7, 2020

@author: voodoocode
'''

import numpy as np

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import finn.cfc.pac as pac

import csv

def generate_high_frequency_signal(n, frequency_sampling, frequency_within_bursts, random_noise_strength, 
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

def draw_figure(scores, best_fits, amplitude_signals, frequencies_between_bursts, tgt_frequency_between_bursts, high_freq_frame_offsets):
    (_, axes) = plt.subplots(len(high_freq_frame_offsets), len(frequencies_between_bursts))
    
    axes = np.reshape(axes, -1)
    
    for (ax_idx, _) in enumerate(amplitude_signals):
        axes[ax_idx].plot(np.arange(0, 1, 1/len(amplitude_signals[ax_idx])), amplitude_signals[ax_idx], label = "original data")
        axes[ax_idx].plot(np.arange(0, 1, 1/len(amplitude_signals[ax_idx])), best_fits[ax_idx], label = "fitted curve")
        axes[ax_idx].set_title("DMI| PLV| MVL| MI  \n%2.2f|%2.2f|%2.2f|%2.2f" % (scores[ax_idx][0], scores[ax_idx][1], scores[ax_idx][2], scores[ax_idx][3]))

def main(write_csv = False, visualize = True):

    for repetition in range(5):
        data_range = np.arange(0, 30000) #previously noise strength = 0.5
        #data_range = np.arange(0, 500)
        #data_range = np.arange(0, 300000)
        
        #data_range = np.arange(0, 500)
        #data_range = np.arange(0, 250)
        data_range = np.arange(0,  500)
        #data_range = np.arange(0,  600)
        #data_range = np.arange(0,  700)
        #data_range = np.arange(0,  800)
        #data_range = np.arange(0,  900)
        #data_range = np.arange(0, 1000)
        
        #data_range = np.arange(0, 300000)
        
        frequency_sampling = 1000
        frequencies_between_bursts = [2, 5, 10, 15, 20, 25, 30]
        #frequencies_between_bursts = [10, 20]
        tgt_frequency_between_bursts = 10
        frequency_within_bursts = 200
        #high_freq_frame_offsets = [0, 20, 40]
        high_freq_frame_offsets = [0]
        #high_freq_frame_offsets = [0]
        
        #Configure noise data
        #random_noise_strength = 0 #0.25 #0.25 #0.5 #1# 1.50#1.5
        #random_noise_strength =  0.50#1.5
        #random_noise_strength =  0.33
        
        random_noise_strength = 0.00
        #random_noise_strength = 0.25
        #random_noise_strength = 0.50
        #random_noise_strength = 1.00
        #random_noise_strength = 1.50
    
        #Generate sample data
        burst_length = frequency_sampling / tgt_frequency_between_bursts
        burst_count = len(data_range) / frequency_sampling * tgt_frequency_between_bursts
        
        for demo_data_type in [True]:
        
            high_freq_signals = [generate_high_frequency_signal(len(data_range), frequency_sampling, frequency_within_bursts, random_noise_strength,
                                                                high_freq_frame_offset, burst_count, burst_length, demo_data_type) for high_freq_frame_offset in high_freq_frame_offsets]
            low_freq_signals = [np.sin(2 * np.pi * frequency_between_bursts * data_range / frequency_sampling) for frequency_between_bursts in frequencies_between_bursts]
            
            scores = list(); best_fits = list(); amplitude_signals = list()
            for high_freq_signal in high_freq_signals:
                for low_freq_signal in low_freq_signals:
                    tmp = pac.run_dmi(low_freq_signal, high_freq_signal, phase_window_half_size = 4, phase_step_width = 2)
                    d_mi_score = tmp[0]
                    plv_score = pac.run_plv(low_freq_signal, high_freq_signal)
                    mvl_score = pac.run_mvl(low_freq_signal, high_freq_signal)
                    mi_score = pac.run_mi(low_freq_signal, high_freq_signal) * 100
                    scores.append([d_mi_score, plv_score, mvl_score, mi_score]); best_fits.append(tmp[1]); amplitude_signals.append(tmp[2])
            
            #visualization
            draw_figure(scores, best_fits, amplitude_signals, frequencies_between_bursts, tgt_frequency_between_bursts, high_freq_frame_offsets)
        
            if (write_csv == True):
                file = csv.writer(open("figure_" + str(len(data_range)) + "_" + str(random_noise_strength).replace(".", "") + "_" + str(repetition) + ".csv", "w"), delimiter = ",")
                
                for idx0 in range(0, 7):
                    for idx1 in range(0, 1):
                        file.writerow(scores[idx0 + idx1 * 7])
        
            plt.legend()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        if (visualize == True):
            plt.show(block = True)

main()


