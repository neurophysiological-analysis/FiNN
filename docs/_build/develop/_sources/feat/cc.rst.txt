
.. _cc_module:

Complex coherency
=================

This function may be called for data in the time domain or the frequency domain.

.. currentmodule:: feat.sfc
.. autofunction:: cc_td

.. currentmodule:: feat.sfc
.. autofunction:: cc_fd

The following code example shows how to calculate complex coherency, a precursor to several connectivity metrics. 

.. code-block:: python
    
   import numpy as np

   import finnpy.feat.sfc as sfc  # @UnresolvedImport
   import finnpy.demo.functionality.sfc.gen_demo_data as gen_demo_data  # @UnresolvedImport

   def main():
      minimum_frequency = 13
      maximum_frequency = 27
      
      frequency_sampling = 3000
      time_s = 120
      offset_s = 1
      signal_length_samples = int(frequency_sampling * (time_s + offset_s * 2)) 
      data = gen_demo_data.gen_wn_signal(minimum_frequency, maximum_frequency, frequency_sampling, signal_length_samples)
      frequency_peak = (maximum_frequency + minimum_frequency)/2
      
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
      
      (bins, cc_td) = calc_from_time_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft)  # @UnusedVariable
      cc_fd = calc_from_frequency_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft)
      
      if ((cc_fd == cc_td).all() == False):
         print("Error")
      
   def calc_from_time_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft, window = "hann", pad_type = "zero"):
      return sfc.cc_td(signal_1, signal_2, nperseg, pad_type, frequency_sampling, nfft, window)

   def calc_from_frequency_domain(signal_1, signal_2, frequency_sampling, nperseg, nfft, pad_type = "zero"):
      seg_data_X = sfc._segment_data(signal_1, nperseg, pad_type)  # pylint: disable=protected-access
      seg_data_Y = sfc._segment_data(signal_2, nperseg, pad_type)  # pylint: disable=protected-access

      (bins, fd_signal_1) = sfc._calc_FFT(seg_data_X, frequency_sampling, nfft)  # pylint: disable=protected-access @UnusedVariable, unknown-option-value
      (_,    fd_signal_2) = sfc._calc_FFT(seg_data_Y, frequency_sampling, nfft)  # pylint: disable=protected-access
      
      return sfc.cc_fd(fd_signal_1, fd_signal_2)
      
      
   main()




