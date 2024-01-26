
.. _source_reconstruction_pitfalls_label:

Likely pitfalls
===============

This guide explains how to deal with likely pitfalls in source reconstruction for MEG using FiNNPy.

MEG and MRI coregistration
--------------------------
The coregistration between MEG and MRI space has been left unchecked. This step must be manually verified. This may be done as follows:

.. code-block::
       
       finnpy.source_reconstruction.coregistration_meg_mri.plot_coregistration(rigid_mri_to_meg_trans, rec_meta_info, meg_pts, fs_subj_path)
       
Producing the following output:

.. image:: img/MEG_source_reconstruction_coreg.png
   :alt: MEG and MRI coregistration example
   :width: 400
   :align: center


Sensor noise covariance
-----------------------
The sensor noise covariance is faulty. This may be investigated by adding a power spike to a sensor-space channel and investigate where it is projected.

Alternatively, multiple files for a sensor noise covariance may be acquired and their similarity compared.

Improper skull model
--------------------
The skull model was improperly extracted. This step must be manually verified.  This may be done as follows:

.. code-block::
       
       (ws_in_skull_vert, ws_in_skull_faces, 
        ws_out_skull_vert, ws_out_skull_faces,
        ws_out_skin_vect, ws_out_skin_faces) = finnpy.source_reconstruction.bem_model.read_skull_and_skin_models(fs_subj_path, subj_name + rec_folder)
       
       finnpy.source_reconstruction.bem_model.plot_skull_and_skin_models(ws_in_skull_vert, ws_in_skull_faces,
                                                                         ws_out_skull_vert, ws_out_skull_faces,
                                                                         ws_out_skin_vect, ws_out_skin_faces,
                                                                         fs_subj_path)
       
Producing the following output:

.. image:: img/MEG_source_reconstruction_ws.png
   :alt: Skull model example
   :width: 500
   :align: center



