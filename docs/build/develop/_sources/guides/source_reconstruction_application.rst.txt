
.. _source_reconstruction_application_label:

Model computation
=================

This part of the guide explains how the inverse model and the fsaverage source space morphing is applied. For previous steps, see :ref:`source_reconstruction_general_label`, :ref:`source_reconstruction_specific_label`. 

Herein, the individual commands of this section will be explained in a step-by-step fashion.
    
.. code-block::

  import finnpy.source_reconstruction.inverse_model as finnpy_sr_inv
  import finnpy.source_reconstruction.utils as finnpy_sr_utils
  import finnpy.source_reconstruction.source_region_model as finnpy_sr_srm

  src_data = finnpy_sr_inv.apply_inverse_model(sen_data,
  					       inv_trans,
  					       noise_norm)
  fs_avg_src_data = finnpy_sr_utils.apply_mri_subj_to_fs_avg_trans_mat(fs_avg_trans_mat,
  								       src_data)
  (morphed_epoch_data,
   morphed_epoch_channels,
   morphed_region_names) = finnpy_sr_srm.avg_source_regions(fs_avg_src_data,
   							    src_fs_avg_valid_lh_vert,
   							    src_fs_avg_valid_rh_vert,
   							    octa_model_vert,
   							    octa_model_faces,
   							    fs_path)
  morphed_epoch_data = np.asarray(morphed_epoch_data)

The first step applies the inverse model to sensor space data, transforming the data into source space.

.. code-block::

  src_data = finnpy_sr_inv.apply_inverse_model(sen_data,
  					       inv_trans,
  					       noise_norm)
       							 
Having transformed the data from sensor to source space, the data has been shifted into a subject specific variant of source space. This naturally limits comparbility between individual subjects. To establish comparability, the data is subsequently transformed from subject specific source space into fsaverage source space. 

.. code-block::

  fs_avg_src_data = finnpy_sr_utils.apply_mri_subj_to_fs_avg_trans_mat(fs_avg_trans_mat,
  								       src_data)

Finally, as the source reconstruction results in a unsustainably high number of channels, individual channels are clustered according to the Desikan-Killiany atlas.

.. code-block::

  (morphed_epoch_data,
   morphed_epoch_channels,
   morphed_region_names) = finnpy_sr_srm.avg_source_regions(fs_avg_src_data,
   							    src_fs_avg_valid_lh_vert,
   							    src_fs_avg_valid_rh_vert,
   							    octa_model_vert,
   							    octa_model_faces,
   							    fs_path)
  morphed_epoch_data = np.asarray(morphed_epoch_data)

After this step, the data has been transformed from sensor space into fsaverage source space with cortical regions clustered according to the Desikan-Killiany atlas.




