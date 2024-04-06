
.. _src_rec_apply_label:

Model application
=================

This part of the guide explains how the inverse model and the fsaverage source space morphing is applied. For previous steps, see :ref:`src_rec_device_label`, :ref:`src_rec_subject_label`, :ref:`src_rec_recording_label`. 

Herein, the individual commands of this section will be explained in a step-by-step fashion. 

The first step is to employ the previously created inverse model (see :ref:`src_rec_recording_label`) to transform the data from sensor to source space.
    
.. code-block::

  src_data = finnpy.src_rec.inv_mdl.apply(sen_data, inv_mdl)

Afterwards, data is moved from subject specific into fs average space.

.. code-block::
  
  fsavg_src_data = finnpy.src_rec.subj_to_vsavg.apply(subj_to_fsavg_mdl, src_data)

Finally, the in fs-average space defined Desikan-Killiany is employed to consolidate individiual source space channels into regions. 

.. code-block::
  
  (clust_src_data, chs, ch_names) = finnpy.src_rec.avg_src_reg.run(fsavg_src_data, subj_to_fsavg_mdl, fs_path)

This concludes the source construction pipeline. Potential pitfalls during source reconstruction are discussed in :ref:`src_rec_pitfalls_label`. 




