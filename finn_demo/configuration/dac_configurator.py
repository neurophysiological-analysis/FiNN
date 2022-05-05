'''
Created on Dec 29, 2021

@author: voodoocode
'''

import numpy as np

import finn.sfc.td as td

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import finn.misc.timed_pool as tp

import finn_demo.demo_data.demo_data_paths as demo_paths

################################
# CONFIGURE ACCORDING TO SETUP #
################################

frequency_sampling = 5500
frequency_min = 25
frequency_max = 35
frequency_bin_sz = 0.5

thread_cnt = 1 #Number of threads used for the simulation

input_data = None

##################################
# CONFIGURE ACCORDING TO RESULTS #
##################################

minimal_angle_thresh = 10
volume_conductance_ratio = 0.3

#############################
# CODE BELOW, DO NOT MODIFY #
#############################

def dac(data_1, data_2, freq_bin_factor):
    """
    Calculates dac scores in the scope of this configurator.
    
    :param data_1: First demo data channel to be analyzed.
    :param data_2: Second data channel to be analyzed. 
    :param freq_bin_factor: Factor inverse to the frequency bin width.
    
    :return: DAC scores
    """
    
    return td.run_dac(data_1, data_2, frequency_min, frequency_max, frequency_sampling, 
                    int(frequency_sampling*freq_bin_factor), int(frequency_sampling*freq_bin_factor),
                    True, minimal_angle_thresh, volume_conductance_ratio)[1]

def main(data = None):
    """
    Call this function to execute the configurator.
    
    :param data: Expects a numpy array of format 2 x samples or 1 x samples to be used as input data. In case no input data is provided a generic input data is used.
    """   
    #Phase range
    phase_min = -270
    phase_max = 270
    phase_step = 2

    run(data, phase_min, phase_max, phase_step)
            
    plt.show(block = True)

def _run_inner(data1, data2, offset, phase_shift, freq_bin_factor):
    
    """
    Parallelization-friendly inner loop of the DAC call.
    
    :param data_1: First demo data channel to be analyzed.
    :param data_2: Second data channel to be analyzed. 
    :param offset: Initial offset of the first data vector to enable temporal shifts (of the 2nd) into both directions.
    :param phase_shift: Phase shift of this specific sample.
    :param freq_bin_factor: Factor inverse to the frequency bin width.
    
    :return: DAC scores
    """
    
    loc_offset = offset - int(np.ceil(frequency_sampling/(np.mean([frequency_min, frequency_max])) * phase_shift/360))
    loc_data = data2[(loc_offset):]
    data2_padded = np.zeros(loc_data.shape)
    data2_padded += loc_data
    
    return dac(data1, data2_padded, freq_bin_factor)

def run(data, phase_min, phase_max, phase_step):
    """
    Parallelization-friendly outer loop of the DAC call.
    
    :param data: Expects a numpy array of format 2 x samples or 1 x samples to be used as input data. In case no input data is provided a generic input data is used.
    :param phase_min: Minimal phase shift for the simulation.
    :param phase_max: Maximum phase shift for the simulation.
    :param phase_step: Phase shift step width for the simulation.
    
    :return: DAC scores
    """
    
    freq_bin_factor = 1/frequency_bin_sz
    
    if (data is None):
        path = demo_paths.per_sfc_data_0
        data1 = np.load(path)[2, :]
        data2 = np.load(path)[2, :]
    else:
        data1 = np.copy(data)
        data2 = np.copy(data)
        
    #Data container
    offset = int(np.ceil(frequency_sampling/np.mean([frequency_min, frequency_max])))
    data1 = data1[(offset):]
    
    dac_scores = tp.run(thread_cnt, _run_inner, [(data1, data2, offset, phase_shift,
                                                   freq_bin_factor) for phase_shift in np.arange(phase_min, phase_max, phase_step)], max_time = None, delete_data = False)
    dac_scores = np.asarray(dac_scores)
    
    #Draw figure
    ref_shape = np.ones(len(np.arange(phase_min, phase_max, phase_step)));
    ref_shape[np.argwhere(np.arange(phase_min, phase_max, phase_step) > 0).squeeze()] = -1;
    ref_shape[np.argwhere(np.abs(np.arange(phase_min, phase_max, phase_step)) < minimal_angle_thresh).squeeze()] = 0;

    plt.scatter(np.arange(phase_min, phase_max, phase_step), ref_shape, color = "black", label = "idealized reference", marker = "s")
    plt.scatter(np.arange(phase_min, phase_max, phase_step), dac_scores, color = "blue", label = "DAC scores")
    plt.scatter(np.arange(phase_min, phase_max, phase_step)[np.argwhere(np.isnan(dac_scores)).squeeze()], np.zeros(np.argwhere(np.isnan(dac_scores)).squeeze().shape), color = "red", 
                label = "volume conductance")
    
    plt.legend()

#main(input_data)


