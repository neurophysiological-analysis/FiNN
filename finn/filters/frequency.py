'''
Created on Jun 11, 2018

Provides different methods for frequency filtering of provided signals.

:author: voodoocode
'''

import scipy.signal
import scipy.fftpack
import scipy.fft
import math
import numpy as np

def butter(data, f_low, f_high, fs, order, zero_phase = True):
    """
    Creates a butterworth filter and applies it to the input data.
    
    :param data: Single vector of input data    
    :param f_low: Start of filters's pass band.
    :param f_high: End of the filters's pass band.
    :param fs: Sampling frequency
    :param order: Order of the filters.
    :param zero_phase: If 'true' the filters is applied forwards and backwards resulting in zero phase distortion.
        
    :return: Filtered data.
    """
    
    nyq = 0.5 * fs
        
    #https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.butter.html
    filter_param = scipy.signal.butter(N = order, Wn = (f_low / nyq , f_high / nyq), btype = 'bandpass', analog = False, output = 'ba')
    filter_func = _apply_zero_phase if (zero_phase == True) else _apply_nonzero_phase

    return filter_func(filter_param[0], filter_param[1], data)
        
def _apply_zero_phase(b, a, data): 
    """
    Applies the filters twice resulting in zero phase distortion
    
    :param b: Numerator of the filters's coefficient vector
    :param a: Denominator of the filters's coefficient vector
    :param data: The data array
    
    :return: The filtered data array
    """
    
    return scipy.signal.filtfilt(b, a, data)
            
def _apply_nonzero_phase(b, a, data):
    """
    Applies scipy's lfilter function.
    
    :param b: Numerator of the filters's coefficient vector
    :param a: Denominator of the filters's coefficient vector
    :param data: The data array
    
    :return: The filtered data array
    """    
     
    return scipy.signal.lfilter(b, a, data)

def fir(data, f_low, f_high, trans_width, fs, ripple_pass_band = 10e-5, stop_band_suppression = 10e-7, fft_win_sz = 2, pad_type = "zero", 
        mode = "fast"):
    """
    Constructs an FIR filter and applies it to data.
    
    f_low = None and f_high = value creates a lowpass filter
    f_low = value and f_high = None creates a highpass filter
    f_low = value and f_high = value creates a bandstop or bandpass filter.
    In case f_low < f_high, a bandpass is created; if f_low > f_high a bandstop filter is created.
    
    This filter is always a zero-phase filter.
    
    :param data: Data to be filtered. Single vector of data.
    :param f_low: Start of the pass band.
    :param f_high: End of the pass band.
    :param trans_width: Width of the transition band.
    :param fs: Sampling frequency.
    :param ripple_pass_band: Suppression of the ripple in the pass band.
    :param stop_band_suppression: Suppression of the stop band.
    :param fft_win_sz: Size of the fft window.
    :param pad_type: Type of padding. Supported types are: *zero* and *mirror*
    :param mode: Defines whether the FIR filter is executed in *fast* or *precise* mode (32 vs 64 bit floats)
    
    :return: Filtered data.
    """
    
    
    nyq = fs / 2
    
    N = _estimate_FIR_coeff_num(fs, trans_width, ripple_pass_band, stop_band_suppression)
        
    if (f_low is not None and f_high is None): # Is high pass
        (freq, gain) = _calc_FIR_freq_and_gain_left_sided(nyq, f_low, trans_width)
        if ((N % 2) == 0):
            N += 1
    if (f_low is None and f_high is not None): # Is low pass
        (freq, gain) = _calc_FIR_freq_and_gain_right_sided(nyq, f_high, trans_width)
        if ((N % 2) == 1):
            N += 1
    if (f_low is not None and f_high is not None): # Is band pass or band stop
        if (f_low > f_high): # Is band stop
            (freq, gain) = _calc_FIR_freq_and_gain_two_sided(nyq, f_high, f_low, trans_width)
            gain = np.logical_not(gain).astype(int)
            if ((N % 2) == 0):
                N += 1
        else: # Is band pass
            (freq, gain) = _calc_FIR_freq_and_gain_two_sided(nyq, f_low, f_high, trans_width)
            if ((N % 2) == 1):
                N += 1
    
    coeffs = scipy.signal.firwin2(numtaps = N, freq = freq, gain = gain, window = 'hamming') # Guarantees a linear phase
    if (mode == "fast"):
        data = np.asarray(data, dtype = np.float32)
        coeffs = np.asarray(coeffs, dtype = np.float32)
    res_data = _overlap_add(data, coeffs, fs, trans_width, fft_win_sz, pad_type) # Removes shift introduced by the filter
                    
    return res_data

def _estimate_FIR_coeff_num(fs, trans_width, ripple_pass_band = 10e-5, stop_band_suppression = 10e-7):
    """
    Estimates the FIR filters coefficients base on the given parameters. Note, the sharper the 'edges' of the filters are, 
    the wider becomes the filters
    
    :param fs: Samling frequency
    :param trans_width: Width of the transition band
    :param ripple_pass_band: Factor for suppression of the ripple in the pass band
    :param stop_band_suppression: Factor for suppression of the ripple in the stop band
    
    :return: FIR filters coefficients
    
    Source of the algorithm is: 
    Digital Processing of Signals – Theory and Practice, Bellanger 1984
    https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need/31077
    """
    
    return int(2/3 * math.log10(1/(10 * ripple_pass_band * stop_band_suppression)) * fs / trans_width)

def _calc_FIR_freq_and_gain_two_sided(nyq, f_low, f_high, trans_width):
    """
    Calculates the two-sided frequency window and the gain for a FIR filters 
    
    :param nyq: Nyquist frequency of the signal
    :param f_low: Low pass frequency of the filters
    :param f_high: High pass frequency of the filters
    :param trans_width: Width of the transition band
    
    :return: (Frequency window, Gain window) for the FIR filters
    """
    
    freq = [max(f_low - trans_width, 0), f_low, f_high, min(f_high + trans_width, nyq)]
    gain = [0, 1, 1, 0]
    
    if (freq[0] != 0):
        freq = [0] + freq 
        gain = [0] + gain
        
    if (freq[-1] != nyq):
        freq = freq + [nyq]
        gain = gain + [0]
    
    freq = np.asarray(freq, dtype = float)
    freq /= nyq
    
    return (freq, gain)

def _calc_FIR_freq_and_gain_left_sided(nyq, f_low, trans_width):
    """
    Calculates the left-sided frequency window and the gain for a FIR filters 
    
    :param nyq: Nyquist frequency of the signal
    :param f_low: Low pass frequency of the filters
    :param f_high: High pass frequency of the filters
    :param trans_width: Width of the transition band
    
    :return: (Frequency window, Gain window) for the FIR filters
    """
    
    freq = [0, f_low, min(f_low + trans_width, nyq), nyq]
    gain = [0, 0, 1, 1]
        
    freq = np.asarray(freq, dtype = float)
    freq /= nyq
    
    return (freq, gain)

def _calc_FIR_freq_and_gain_right_sided(nyq, f_high, trans_width):
    """
    Calculates the right-sided frequency window and the gain for a FIR filters 
    
    :param nyq: Nyquist frequency of the signal
    :param f_low: Low pass frequency of the filters
    :param f_high: High pass frequency of the filters
    :param trans_width: Width of the transition band
    
    :return: (Frequency window, Gain window) for the FIR filters
    """
    
    freq = [0, f_high, min(f_high + trans_width, nyq), nyq]
    gain = [1, 1, 0, 0]
    
    freq = np.asarray(freq, dtype = float)
    freq /= nyq
    
    return (freq, gain)

def _overlap_add(data, coeffs, fs, trans_width, fft_win_sz, pad_type):
    """
    Implements the overlap add approach to speed up filter application.
    
    :param data: Data to be filtered. Single vector of data.
    :param coeffs: Filter coefficients.
    :param fs: Sampling frequency.
    :param trans_width: Width of the transition band.
    :param fft_win_sz: Size of the fft window.
    :param pad_type: Type of padding. Supported types are: *zero* and *mirror*
    
    :return: Filtered data
    """
    
    coeff_length = len(coeffs)
    data_length = len(data)
    
    #pad_width = max(min(coeff_length, data_length) - 1, 0)
    pad_width = coeff_length
    
    #Compensate for filters introduced shift
    shift = math.floor((coeff_length - 1) / 2) + pad_width
        
    fft_win_sz = _overlap_add_sanity_check(fs = fs, trans_width = trans_width, data_length = data_length, filter_length = coeff_length,
                                       shift = shift, fft_win_sz = fft_win_sz)
    
    #filter_length has to be at least the length of a single fft window
    coeffsFFT = scipy.fftpack.fft(np.concatenate([coeffs, np.zeros(fft_win_sz - coeff_length, dtype = coeffs.dtype)]))
        
    
    (seg_length, seg_cnt) = _estimate_overlap_add_segments(data_length = data_length, pad_edge_width = pad_width,
                                                      filter_length = coeff_length, fft_win_sz = fft_win_sz)

    padded_data = _pad_overlap_add_data(data, pad_width, pad_type)
    
    result_data = np.zeros((len(padded_data)), dtype = coeffs.dtype)

    coeffsFFT = coeffsFFT[:int(len(coeffsFFT)/2 + 1)]
    
    for segIdx in range(0, seg_cnt):
        
        #Position of the current segment in the input data
        win_start = segIdx * seg_length
        win_end = win_start + seg_length
        
        #Pad segment size to fft_win_sz using zeros
        seg_data = padded_data[win_start:win_end]
        #seg_data = np.concatenate([seg_data, np.zeros(fft_win_sz - len(seg_data))])
        seg_data = np.pad(seg_data, ((0, fft_win_sz - len(seg_data))), "constant", constant_values = 0)
        
        #Apply fft, filters and ifft
        seg_result_data = scipy.fft.irfft((coeffsFFT * scipy.fft.rfft(seg_data, fft_win_sz)), fft_win_sz)
        
        filtStart = max(0, win_start - shift)    # Permanently shift 'n' bytes to the left as the padded info is unnecessary
                                                # Idea: if "start" < "shift", fall back to zero. This is a shortened segment.
                                                #      else provide area for a 'complete' segment
        filtEnd = min(max(0, win_start - shift + fft_win_sz), len(padded_data))
        
        tmpStart = max(0, shift - win_start)     # Copy only bytes 'after' the shift has been compensated.
                                                # Idea: if "start" < "shift", Skip first 'n' bytes
                                                #       else copy all bytes as out of shift zone
        tmpEnd = tmpStart + filtEnd - filtStart
        
        result_data[filtStart:filtEnd] += seg_result_data[tmpStart:tmpEnd] # Overlapping of the data is not a problem, since it is zero padded.
                                                                        # The overlapping part will be all/mostly zeros and therefore
                                                                        # is negligible in regard to distortions. Nevertheless it is important
                                                                        # regarding a smooth continuation of the signal in contrast to an
                                                                        # abrupt stop.
    
    res_data = result_data[:len(padded_data) - 2 * pad_width].astype(data.dtype)    # Remember to remove the padded data from the new data.
                                                                                # The left pad should already be removed, but the 
                                                                                # "padded_data" reference still 'counts' for two pads.

    return res_data

def _overlap_add_sanity_check(fs, trans_width, data_length, filter_length, shift, fft_win_sz = 2):
    """
    Checks whether designed filters can be applied to the data
    
    :param fs: Sampling frequency.
    :param trans_width: Width of the transition band.
    :param data_length: Length of the data in samples.
    :param filter_length: Length of the filters in samples.
    :param shift: Size of the filters induced shift.
    :param fft_win_sz: Size of the fft window. 
    
    :return: Valid size of the fft window.
    """
    minfft_win_sz = max(filter_length, shift) #FFT window has to be at least as large as the filters including the shifted amount
    maxfft_win_sz = data_length #FFT window cannot be longer than the entire data
    
    while (fft_win_sz < minfft_win_sz):
        fft_win_sz *= 2
        
    if (fft_win_sz < minfft_win_sz):
        print("Error, there is no power of two between the min fft win size %i and the max fft win size %i." % (minfft_win_sz, maxfft_win_sz))
        print("Either add more data or easen the restrictions on the filters.")
        
    fftBinWidth = (fs / (fft_win_sz / 2))
    if (fftBinWidth > trans_width):
        print("Warning, fft bin size is only %f, but the desired transition width is %f" % (fftBinWidth, trans_width))
        print("Either add more data or increase the transition width.")        
    return fft_win_sz

def _estimate_overlap_add_segments(data_length, pad_edge_width, filter_length, fft_win_sz):
    """
    Estimates the size and the number of segments used in the overlap add approach.
    
    :param data_length: Lenght of the data in samples.
    :param pad_edge_width: Length of the padding on the left and right side of the data.
    :param filter_length: Length of the filters in samples.
    :param fft_win_sz: Size of the fft window.
    
    :return: (Segment size, segment number) for overlap adding.

    Formular for optimal filters length, not the best source but a source at least:
    https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method

    """
    
    seg_length = fft_win_sz - filter_length + 1
    seg_cnt = int(np.ceil((data_length + pad_edge_width + pad_edge_width) / seg_length))
    
    return (seg_length, seg_cnt)

def _pad_overlap_add_data(data, pad_width, pad_type = "zero"):
    """
    Pads the individual segments of the overlap add function. 
    
    :param data: Input data.
    :param pad_width: Size of the padding.
    
    :return: Padded data. 
    """

    if (pad_type == "zero"):
        return np.pad(data, pad_width, "constant", constant_values = 0)
    elif(pad_type == "mirror"):
        return np.pad(data, pad_width, "reflect")















