
Phase Lock Value
=======================


.. currentmodule:: cfc.pac
.. autofunction:: run_plv


The following example shows how to apply the plv to estimate PAC.

.. code-block:: python
   
   import numpy as np

   import finn.cfc.pac as pac

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
       random_noise_strength = 1
       
       #Generate sample data
       burst_length = frequency_sampling / tgt_frequency_between_bursts
       burst_count = len(data_range) / frequency_sampling * tgt_frequency_between_bursts
       
       high_freq_signals = [generate_high_frequency_signal(len(data_range), frequency_sampling, frequency_within_bursts,
                                                           random_noise_strength, high_freq_frame_offset, burst_count, burst_length) for high_freq_frame_offset in high_freq_frame_offsets]
       low_freq_signals = [np.sin(2 * np.pi * frequency_between_bursts * data_range / frequency_sampling) for frequency_between_bursts in frequencies_between_bursts]
       
       scores = np.zeros((len(high_freq_signals), len(low_freq_signals)));
       for (high_freq_idx, high_freq_signal) in enumerate(high_freq_signals):
           for (low_freq_idx, low_freq_signal) in enumerate(low_freq_signals):
               scores[high_freq_idx, low_freq_idx] = pac.run_plv(low_freq_signal, high_freq_signal)
               
       print("target frequency: ", tgt_frequency_between_bursts)
       for x in range(len(frequencies_between_bursts)):
           print("%.3f" % (frequencies_between_bursts[x]), end = "\t")
       print("")
       for y in range(len(high_freq_frame_offsets)):
           for x in range(len(frequencies_between_bursts)):
               print("%.3f" % (scores[y][x],), end = "\t")
           print("")

   main()

Using the direct modulation index, PAC between the low frequency signal (2/5/10/15/50Hz) signal and the high frequency signal (10Hz amplitude modulation). 

======  ======  ======  ======  ======
2Hz     5Hz     10Hz    15Hz    50Hz
======  ======  ======  ======  ======
2.000   5.000   10.000  15.000  50.000
0.004   0.007   0.036    0.005   0.009
0.008   0.003   0.024    0.005   0.010
0.003   0.004   0.029    0.006   0.003
======  ======  ======  ======  ======


