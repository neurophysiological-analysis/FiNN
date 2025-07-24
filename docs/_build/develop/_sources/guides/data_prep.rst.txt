
.. _data_prep_label:

General
=======

While data preparation is a highly paradigm-specific endeavor, this section provides a general outline which may be followed to preprocess most data.

1. Faulty signal detection
2. Re-referencing
3. Artifact removal
4. Re-sampling

Faulty signal detection
-----------------------

A common sight in any electrophysiological analysis is the (temporary) loss of one or multiple channels during recording. Such occurrences must be identified and removed from the analysis in a first step. While automated tools exist to detect faulty channels and epochs, these often assume that the channel/epoch is faulty *on average*. Hence, if an artifact is only sporadically or momentarily present, such occurrences will likely not get picked up. Therefore, manual screening is highly recommended. There are two strategies to handle faulty channels/epochs,

1. to remove any faulty channel/epoch
<or>
2. to attempt to restore the uncorrupted signal.

While the optimal strategy depends on the specific application case, restoring the uncorrupted signal from its corrupted state is much more complex and may easily, if not done carefully, introduce *artificial effects* which may later on be miss-characterized as authentic effects.

Re-referencing
--------------

Most signals (except MEG) are recorded with a reference to a specific site. Hence, they do not measure activity at a specific location, but rather the difference in activity between two sites. A common way to generate reference-free recordings from those recorded against a reference is common average re-referencing. FiNNpy provides functionality to achieve this task,

.. code-block::
   
   import finnpy.basic.car

   finnpy.basic.car.run(data)

| *data* being the input data.

While common average re-referencing removes the shared component across signals, there may be unintended interactions with source reconstruction. Namely, in case a sensor-noise covariance matrix is to be computed for source reconstruction (see source reconstruction :ref:`src_rec_main_label`), this matrix *must* be computed *before* applying common average re-referencing to avoid a rank-deficient sensor noise covariance, ultimately denying proper source reconstruction.

Artifact removal
----------------

Some artifacts are easier to identify in the frequency domain whereas others are easier to identify in the time domain. For frequency domain artifacts (such as line noise) FiNNpy provides an FIR filter which may be configured as a stop-band filter, removing line-noise

.. code-block::
   
   import finnpy.filters.frequency as ff

   ff.fir(data, f_max, f_min, f_res, fs)

| *data* being the input data
| *f_max* the upper band of the stop-band
| *f_min* the lower band of the stop-band
| *f_res* being the frequency resolution (i.e. 0.1 Hz steps)
| *fs* being the sampling frequency

Time domain artifacts are much more difficult to handle and will require specific approaches for the problem at hand.


Re-sampling
-----------

To avoid aliasing (the erroneous occurrence of high-frequency effects are lower frequencies), electrophysiological signals should be recorded at high frequencies. However, this may significantly impact data evaluation efforts as processing highly resolved data is very slow. Hence, it is often a good choice to downsample data to a frequency range of interest (e.g. up to 400 Hz for most electrophysiological signals, more for single unit activity) to decrease the amount of to-be processed data. To avoid aliasing, this requires the application of a low-pass filter. However, to avoid aliasing from downsampling, the Nyquist–Shannon theorem needs to be heeded. Particularly, if the target frequency range is <400 Hz, a low-pass filter needs to be applied at less than half the new sampling frequency, <200 Hz (e.g. 195 Hz). In FiNNpy, this may be realized as follows

.. code-block::
   
   import finnpy.filters.frequency as ff
   import finnpy.basic.downsampling as ds

   filt_data = ff.fir(data, f_min, f_max, f_res, fs)
   ds_data = ds.run(filt_data, fs, new_fs)

| *data* being the input data
| *f_min* the lower band of the stop-band
| *f_max* the upper band of the stop-band
| *f_res* being the frequency resolution (i.e. 0.1 Hz steps)
| *fs* being the sampling frequency

Of note, by not reversing the arguments *f_min* and *f_max*, the filter is applied as a pass-band filter. Hence, any activity between f_min (defaults to None) and f_max (195 in this example) is permitted whereas any other activity is attenuated. 
