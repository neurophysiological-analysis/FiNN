
.. _source_reconstruction_label:

Source reconstruction
=====================

This guide explains how to apply source reconstruction for MEG using FiNNPy.

.. image:: ../images/MEG_source_reconstruction_simplified.png
   :scale: 100 %
   :alt: Graphic presentation of the relationship between skull & cortical model
   :align: center


Skull model processing
----------------------

The skull model is a geometrical description of the skull surface. For MEG analyses, a single layer model suffices (option for EEG source reconstruction to be added soon). The skull model is derived from T1 scans and extracted using the watershed algorithm of FreeSurfer. Its density is reduced to increase computeability and mathematical stability. Herein, it is employed to project MEG data from sensor space onto the skull surface.

Cortex model processing
-----------------------
The cortical model is a geometric description of the cortical structure. The cortical model may be directly extracted (using FreeSurfer) from T1 scans. Akin to the skull model, its density is reduced to increase computeability and mathematical stability. Herein, it will be employed to MEG activity from the skull surface onto the cortex proper.

Skull and cortex model fusion
-----------------------------
Skull and cortical models are fused to create the forward model. While the skull model describes the transition from sensor space to skull space, the cortex model may be employed to transition further into cortical space.

Application
-----------

An application example of source reconstruction for MEG is provided below.


.. code-block::

       rec_meta_info = mne.io.read_info(data_path)
       
       print("Setting up freesurfer paths")
       finnpy.source_reconstruction.utils.init_fs_paths(loc_fs_path)
           
       print("Calibrating anatomy")
       coreg_rotors = calibrate_anatomy(loc_fs_path, fs_subj_path, subj_name + rec_folder, rec_meta_info, visualize_coregistration)
       rigid_mri_to_meg_trans = finnpy.source_reconstruction.coregistration_meg_mri.get_rigid_transform(coreg_rotors)
       rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
          
       print("Getting skull and skin models")
       (in_skull_vert, in_skull_faces,
        out_skull_vert, out_skull_faces,
        out_skin_vect, out_skin_faces) = get_skin_skull_model(fs_subj_path, subj_name + rec_folder, visualize_skull_skin_plots)
       del out_skull_vert; del out_skull_faces; del out_skin_vect; del out_skin_faces
        
       print("Setting up source model")
       (lh_white_vert, lh_white_faces,
        rh_white_vert, rh_white_faces,
        lh_white_valid_vert, rh_white_valid_vert,
        octa_model_vert, octa_model_faces) = finnpy.source_reconstruction.source_mesh_model.create_source_mesh_model(fs_subj_path)
          
       print("Calculating BEM model")
       (in_skull_reduced_vert, in_skull_faces, 
        in_skull_faces_area, in_skull_faces_normal, 
        bem_solution) = finnpy.source_reconstruction.bem_model.calc_bem_model_linear_basis(in_skull_vert, in_skull_faces)
       
       print("Calculate forward solution")
       (fwd_sol,
        lh_white_valid_vert, rh_white_valid_vert) = finnpy.source_reconstruction.forward_model.calc_forward_model(lh_white_vert, rh_white_vert, rigid_meg_to_mri_trans, rigid_mri_to_meg_trans, rec_meta_info, in_skull_reduced_vert, in_skull_faces, in_skull_faces_normal, in_skull_faces_area, bem_solution, lh_white_valid_vert, rh_white_valid_vert)
       
       print("Optimize forward model orientation")
       reset_fwd_sol = finnpy.source_reconstruction.forward_model.optimize_fwd_model(lh_white_vert, lh_white_faces, lh_white_valid_vert, rh_white_vert, rh_white_faces, rh_white_valid_vert, fwd_sol, rigid_mri_to_meg_trans)
          
       print("Calculate sensor covariance")
       (sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names) = finnpy.source_reconstruction.sensor_covariance.get_sensor_covariance(file_path = SENSOR_COV_PATH, cov_path = loc_fs_path + "cov_data/", overwrite = overwrite_sensor_cov)
       
       print("Calculate inverse model")
       (inv_trans, noise_norm) = finnpy.source_reconstruction.inverse_model.calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names, reset_fwd_sol, rec_meta_info)
       
       print("Calculate transformation to fs-average")
       (fs_avg_trans_mat, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert) = finnpy.source_reconstruction.utils.get_mri_subj_to_fs_avg_trans_mat(lh_white_valid_vert, rh_white_valid_vert, octa_model_vert, fs_subj_path, loc_fs_path + "fsaverage/", overwrite = overwrite_mri_trans)

Common errors
-------------

MEG and MRI coregistration
^^^^^^^^^^^^^^^^^^^^^^^^^^
The coregistration between MEG and MRI space has been left unchecked. This step must be manually verified

Sensor noise covariance
^^^^^^^^^^^^^^^^^^^^^^^
The sensor noise covariance is faulty. This may be investigated by adding a power spike to a sensor-space channel and investigate where it is projected.

Improper skull model
^^^^^^^^^^^^^^^^^^^^
The skull model was improperly extracted. This step must be manually verified




