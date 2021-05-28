'''
Created on Jun 2, 2020

@author: voodoocode
'''

import numpy as np
import scipy.signal

import lmfit

def __sine(x, phase, amp):
    freq = 1
    fs = 1
    return amp * (np.sin(2 * np.pi * freq * (x - phase) / fs))

def run(low_freq_data, high_freq_data,
         frequency_window_half_size = 10, frequency_step_width = 1,
         max_model_fit_iterations = 200):
    
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, frequency_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, frequency_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < frequency_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(high_freq_data[phase_indices]))
    
    amplitude_signal -= np.nanpercentile(amplitude_signal,25)
    amplitude_signal /= np.nanpercentile(amplitude_signal,75)

    amplitude_signal = amplitude_signal * 2 - 1
    amplitude_signal*= 0.70710676
    
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 1, min = 0.95, max = 1.05, vary = True)
    model = lmfit.Model(__sine, nan_policy = "omit")
    result = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)), params = params, fit_kws = {"maxfev" : max_model_fit_iterations})

    if (np.isnan(amplitude_signal).any() == True):
        amplitude_signal = np.where(np.isnan(amplitude_signal) == False)[0]

    error = np.sum(np.square(result.best_fit - amplitude_signal))/len(amplitude_signal)
    
    error = 1 if (error > 1) else error #Capping the error

    score = 1 - error
    score = 0 if (score < 0) else score

    return (score, result.best_fit, amplitude_signal)





