
.. _src_rec_sensors_label:

Sensor position dependent factors
=================================

Having extracted patient's anatomy in section :ref:`_src_rec_anatomy_label`, the following steps compute the co-variability between individual M/EEG channels and the co-registration between M/EEG sensor position and anatomy. Unlike anatomy, these *may* have to be recomputed for a number of scenarios, including refreshing of EEG gel, EEG electrode cap replacements, patient change, moving significantly within the MEG, changing body position in the MEG. 

Sensor noise covariance
-----------------------

The sensor noise covariance describes how strongly a signal change in one channel is reflected in another channel (assuming a linear relationship. This relationship is critical to ultimately identify those channels that contribute to a change of the observed signals.

The following code-block outlines how to compute the sensor noise covariance.
    
.. code-block::
	
  import finnpy.src_rec.sen_cov
  finnpy.src_rec.sen_cov.run(sensor_data, fs, cov_path, signal_type,
							 valid_channels, ch_names, ch_types,
							 method = None, float_sz = 64, epoch_sz_s = 0.2, method_params = None, 
							 fast_eigendecomp_path = "../FinnPy_speedups/Release/FinnPy_speedups.so", overwrite = False):

*sensor_data* is a numpy array of shape samples x channel_count.
*fs* is the sampling frequency
*cov_path* Path used to save temporary files. Needs to be covariance specific.
*signal_type* is be either 'EEG' or 'MEG'.
*ch_types* is a list identifying channel types. 
*method* is the method employed to calculate the sensor noise covariance, defaults to 'empirically'. Others are 'shrinkage' and 'factor_analysis'.
*float_sz* is the accuracy of the computation. At least 64 bit, recommended are 256 bits (to reduce compounding rounding errors).
*method_params* are parameters passed on to the 'shrinkage' and 'factor analysis' methods.
*fast_eigendecomp_path* is the path to the custom written eigen decomposition. Likewise allows for arbitrary precision, but is much faster than pure Python.
The libraries code is available `here <https://github.com/neurophysiological-analysis/FiNN>`_ and needs to be compiled locally (see :ref:`ext_eigendecom`). 
*overwrite* Boolean flag whether to load preliminary results from the cov_path directory.

M/EEG co-registration
---------------------

The M/EEG co-registration describes the spatial transformation between M/EEG sensors and the patient's anatomy.

The following code-block outlines how to compute the M/EEG co-registration.

.. code-block::

  coreg = finnpy.src_rec.coreg.run(subj_name, anatomy_path, signal_type, use_nasion = True, rec_info = None)

*subj_name* is the name of the subject.
*anatomy_path* is the path to the anatomy folder created by freesurfer. This path should have a sub-folder with the subject's name.
*signal_type* is be either 'EEG' or 'MEG'.
*use_nasion* is a boolean that flags whether to use the nasion in the co-registration. In case the nose is cut from a subject's MRI, it is advised to be 'False', 'True' otherwise.
*rec_info* must either be the employed EEG setup ('1005' or '1020') or the path of the \*.fif file.

Next steps
----------

This concludes the sensor position focused section. The inverse model is calculated in the next section :ref:`src_rec_downstream_label`.
