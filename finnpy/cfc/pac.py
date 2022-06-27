'''
Created on Dec 29, 2020

This module implements different phase amplitude coupling metrics. 

@author: voodoocode
'''

import numpy as np
import scipy.signal
import scipy.stats
import lmfit

def run_dmi(low_freq_data, high_freq_data,
         phase_window_half_size = 10, phase_step_width = 1,
         max_model_fit_iterations = 200):
    """
    Calculates the direct modulation index between a low frequency signal and a high frequency signal. Instead of the original modulation index based on entropy, this modulation index estimate is based on a sinusoidal fit. 
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data. Must have the same length as low_freq_data.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth, but also potentially increased PAC estimates.
    :param phase_step_width: Step width of the phase window used for calculation of frequency/phase histogram.
    :param max_model_fit_iterations: Maximum number of iterations applied during sine fitting.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """
    
    if (len(low_freq_data) != len(high_freq_data)):
        raise AssertionError("Both signals must have the same length")
    
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, phase_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, phase_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < phase_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(high_freq_data[phase_indices])) # No need for a hilbert transform
    
    amplitude_signal -= np.nanpercentile(amplitude_signal,25)
    amplitude_signal /= np.nanpercentile(amplitude_signal,75)

    amplitude_signal = amplitude_signal * 2 - 1
    amplitude_signal*= 0.70710676
    
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 1, min = 0.95, max = 1.05, vary = True)
    model = lmfit.Model(_sine, nan_policy = "omit")
    result = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)),
                       params = params, max_nfev = max_model_fit_iterations)

    if (np.isnan(amplitude_signal).any() == True):
        amplitude_signal = np.where(np.isnan(amplitude_signal) == False)[0]

    error = np.sum(np.square(result.best_fit - amplitude_signal))/len(amplitude_signal)
    
    error = 1 if (error > 1) else error #Capping the error

    score = 1 - error
    score = 0 if (score < 0) else score

    return (score, result.best_fit, amplitude_signal)

def _sine(x, phase, amp):
    """
    Internal method. Used in run_dmi to estimate the direct modulation index. The amount of PAC is quantified via a sine fit. This sine is defined by the following paramters:
    
    :param x: Samples
    :param phase: Phase shift of the sine.
    :param amp: Amplitude of the sine.
    
    :return: Returns the fitted sine at the locations indicated by x.
    """
    freq = 1
    fs = 1
    return amp * (np.sin(2 * np.pi * freq * (x - ((phase + 180)/360)) / fs))
    
def run_plv(low_freq_data, high_freq_data):
    """
    Calculates the phase lock value between a low frequency signal and a high frequency signal.
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data.
    
    :return: Amount of phase amplitude coupling measured using phase lock value.
    """
    phase_low_freq_signal = np.angle(scipy.signal.hilbert(low_freq_data))
    amplitude_signal = np.abs(scipy.signal.hilbert(high_freq_data))
    phase_amplitude_high_freq_signal = np.angle(scipy.signal.hilbert(amplitude_signal))
    
    phase_signal = phase_low_freq_signal - phase_amplitude_high_freq_signal
    
    phase_signal = np.exp(complex(0, 1) * phase_signal)
    
    score = np.abs(np.average(phase_signal))
    
    return score

def run_mi(low_freq_data, high_freq_data, phase_window_half_size = 10, phase_step_width = 20):
    """
    Calculates the modulation index between a low frequency signal and a high frequency signal.
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth/increased PAC estimates.
    :param phase_step_width: Step width/shift of the phase window used for calculation of frequency/phase histogram.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, phase_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, phase_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < phase_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(high_freq_data[phase_indices])) # No need for a hilbert transform
        
    amplitude_signal = np.concatenate((amplitude_signal, amplitude_signal))
    amplitude_signal /= np.nansum(amplitude_signal)
    
    len_signal = len(amplitude_signal)
    uniform_signal = np.random.uniform(np.nanmin(amplitude_signal), np.nanmax(amplitude_signal), len_signal)
    
    score = (scipy.stats.entropy(amplitude_signal, uniform_signal, len_signal))/np.log(len_signal)

    return score

def run_mvl(low_freq_data, high_freq_data):
    """
    Calculates the mean vector length between a low frequency signal and a high frequency signal.
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data.
    
    :return: Amount of phase amplitude coupling measured using the mean vector length.
    """


    phase_low_freq_signal = np.angle(scipy.signal.hilbert(low_freq_data))
    amplitude_high_freq_signal = np.abs(scipy.signal.hilbert(high_freq_data))
    
    vectors = amplitude_high_freq_signal * np.exp(complex(0, 1) * phase_low_freq_signal)
    
    score = np.abs(np.average(vectors))
    
    return score
