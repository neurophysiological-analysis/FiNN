
.. _src_rec_device_label:

MEG recording agnostic model computation
========================================

This part of the guide explains how the device specific components of the model are computed. 

Herein, the individual commands of this section will be explained in a step-by-step fashion.
    
.. code-block::
	
  import finnpy.src_rec.sen_cov
  finnpy.src_rec.sen_cov.run(sen_cov_src_path, sen_cov_tgt_path)

Currently, the sole device specific component is the calculation of the sensor noise covariance. 
*sen_cov_src_path* points to an empty room recording from which the sensor noise covariance is to be calculated.
*sen_cov_tgt_path* points towards the location where the computed sensor noise covariance is to be stored. 
Initially, free surfer paths need to be configured for subsequent FreeSurfer calls.

This concludes the device specific part of the MEG source reconstruction procedure. The next step (subject specific part) is explained in :ref:`src_rec_subject_label`.




