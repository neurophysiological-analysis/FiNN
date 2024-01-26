
.. _source_reconstruction_specific_label:

MEG recording specific model computation
========================================

This part of the guide explains how the MEG recording specific model is computed. After hereon, the model is specified towards a distinct MEG recording (MEG markers). As such, if there are multiple sessions for a single subject, the output of the previous step (:ref:`source_reconstruction_general_label`) may be duplicated to be adapted towards multiple MEG sessions to avoid recomputing.

Herein, the individual commands of this section will be explained in a step-by-step fashion.
    
.. code-block::

  import finnpy.source_reconstruction.utils as finnpy_sr_utils
  import finnpy.source_reconstruction.coregistration_meg_mri as finnpy_sr_coreg
  import finnpy.source_reconstruction.mri_anatomy as finnpy_sr_mri_anat
  import finnpy.source_reconstruction.bem_model as finnpy_sr_bem
  import finnpy.source_reconstruction.source_mesh_model as finnpy_sr_smm
  import finnpy.source_reconstruction.forward_model as finnpy_sr_fwd
  import finnpy.source_reconstruction.inverse_model as finnpy_sr_inv
  
  finnpy_sr_utils.init_fs_paths(anatomy_path, fs_path) #Optional, please only executed if not execuded previously
	
  rec_meta_info = mne.io.read_info(data_path)
  (coreg_rotors, meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name,anatomy_path,
  						       rec_meta_info,
  						       registration_scale_type = "free")
  finnpy_sr_mri_anat.scale_anatomy(anatomy_path, subj_name, coreg_rotors[6:9])
  (coreg_rotors, meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name, anatomy_path,
  						       rec_meta_info,
  						       registration_scale_type = "restricted")

  if (visualize_coregistration):
    rigid_mri_to_meg_trans = finnpy_sr_coreg.get_rigid_transform(coreg_rotors)
    finnpy_sr_coreg.plot_coregistration(rigid_mri_to_meg_trans, rec_meta_info,
    					meg_pts, anatomy_path, subj_name)

  finnpy_sr_bem.calc_skull_and_skin_models(anatomy_path, subj_name,
  					   overwrite = overwrite_ws_extract)
  (ws_in_skull_vert, ws_in_skull_faces, 
   ws_out_skull_vert, ws_out_skull_faces,
   ws_out_skin_vect, ws_out_skin_faces) = finnpy_sr_bem.read_skull_and_skin_models(anatomy_path, subj_name)

  if (visualize_skull_skin_plots):
    finnpy_sr_bem.plot_skull_and_skin_models(ws_in_skull_vert, ws_in_skull_faces,
    		                                                      ws_out_skull_vert, ws_out_skull_faces,
    		                                                      ws_out_skin_vect, ws_out_skin_faces,
    		                                                      anatomy_path, subj_name)
  del ws_out_skull_vert; del ws_out_skull_faces; del ws_out_skin_vect; del ws_out_skin_faces

  (in_skull_reduced_vert, in_skull_faces, 
   in_skull_faces_area, in_skull_faces_normal, 
   bem_solution) = finnpy_sr_bem.calc_bem_model_linear_basis(ws_in_skull_vert, ws_in_skull_faces)

  (lh_white_vert, lh_white_faces,
   rh_white_vert, rh_white_faces,
   lh_sphere_vert,
   rh_sphere_vert) = finnpy_sr_utils.read_cortical_models(anatomy_path, subj_name)
  (octa_model_vert, octa_model_faces) = finnpy_sr_smm.create_source_mesh_model()
  (lh_white_valid_vert, rh_white_valid_vert) = finnpy_sr_smm.match_source_mesh_model(lh_sphere_vert, rh_sphere_vert, octa_model_vert)

  rec_meta_info = mne.io.read_info(data_path)
  finnpy_sr_utils.init_fs_paths(anatomy_path, fs_path)

  rigid_mri_to_meg_trans = finnpy_sr_coreg.get_rigid_transform(coreg_rotors)
  rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
  (fwd_sol,
  lh_white_valid_vert, rh_white_valid_vert) = finnpy_sr_fwd.calc_forward_model(lh_white_vert, rh_white_vert,
		                                                                                                  rigid_meg_to_mri_trans, rigid_mri_to_meg_trans,
		                                                                                                  rec_meta_info, in_skull_reduced_vert, in_skull_faces,
		                                                                                                  in_skull_faces_normal, in_skull_faces_area,
		                                                                                                  bem_solution, lh_white_valid_vert, rh_white_valid_vert)
  optimized_fwd_sol = finnpy_sr_fwd.optimize_fwd_model(lh_white_vert, lh_white_faces, lh_white_valid_vert,
  										  rh_white_vert, rh_white_faces, rh_white_valid_vert,
  										  fwd_sol, rigid_mri_to_meg_trans)
  (inv_trans, noise_norm) = finnpy_sr_inv.calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names, optimized_fwd_sol, rec_meta_info)



  (fs_avg_trans_mat, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert) = finnpy_sr_utils.get_mri_subj_to_fs_avg_trans_mat(lh_white_valid_vert, rh_white_valid_vert, octa_model_vert,
																		     anatomy_path, subj_name, fs_path, overwrite = overwrite_mri_trans)

If not done previously, FreeSurfer needs to be initialized.

.. code-block::

	finnpy_sr_utils.init_fs_paths(anatomy_path,
       							 fs_path)
       							 
See :ref:`source_reconstruction_general_label` for details. The next step is to compute a coregistration between the MEG recording and the anatomical scan.

.. code-block::

	rec_meta_info = mne.io.read_info(data_path)
	(coreg_rotors,
	 meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name,
	 									   anatomy_path,
	 									   rec_meta_info,
	 									   registration_scale_type = "free")
	finnpy_sr_mri_anat.scale_anatomy(anatomy_path,
							       subj_name,
							       coreg_rotors[6:9])
	(coreg_rotors,
	 meg_pts) = finnpy_sr_coreg.calc_coreg(subj_name,
	 									   anatomy_path,
	 									   rec_meta_info,
	 									   registration_scale_type = "restricted")

Meta information is read from the MEG recording (such as MEG-fiducial positions). These are to be aligned with the subject's anatomy through a three step process. Initially, the subject's anatomy is scaled towards the scale of the MEG-fiducials. However, this rescaling of the subject's anatomy necessitates a recomputation of the coregistration. (Background insight: As the coregistration is an iterative optimization, it's reapplication after rescaling provides a different output from the initial coregistration).
		
.. code-block::

	if (visualize_coregistration):
		rigid_mri_to_meg_trans = finnpy_sr_coreg.get_rigid_transform(coreg_rotors)
		finnpy_sr_coreg.plot_coregistration(rigid_mri_to_meg_trans,
											rec_meta_info,
											meg_pts,
											anatomy_path,
											subj_name)
											
Finally, after the coregistration is recomputed, it may be visualized as indicated herein. This allows for visual (manual) confirmation successful coregistration, this is expanded upon in :ref:`source_reconstruction_pitfalls_label`. (Warning: Under mayavi 4.8.2 & vtk 9.3, the call to points3d doesn't work properly. As such, if a TraitError is raised during rendering, please downgrade vtk to 9.2.6 and/or mayavi to an earlier version.)

The next step is to employ the watershed algorithm of FreeSurfer to extract the *inner skull*, *outer skull* and *outer skin* models. As this example details a MEG reconstruction, only the inner skull model is needed. In the first step, the watershed algorithm is employed (provided by FreeSurfer). While this algorithm provides reasonably good results from good quality T1 images, output quality may very if the image is either over or underexposed. However, this may be compensated by adjusting the *preflood_height* parameter (default = 25). 
To verify extraction accuracy, FiNN offers a visualization tool which may be used to inspect the results of the algorithm and adjust the *preflood_height* as needed. Finally, some of the components are not needed for MEG reconstruction (only EEG), they are deleted afterwards. Visual confirmation of correct watershed extraction is highly recommended.

.. code-block::

       finnpy_sr_bem.calc_skull_and_skin_models(anatomy_path, subj_name, overwrite = overwrite_ws_extract)
	(ws_in_skull_vert, ws_in_skull_faces, 
	 ws_out_skull_vert, ws_out_skull_faces,
	 ws_out_skin_vect, ws_out_skin_faces) = finnpy_sr_bem.read_skull_and_skin_models(anatomy_path, subj_name)

Likewise, successful extraction of surfaces with the watershed algorithm should be visually (manually) confirmed.

.. code-block::

	if (visualize_skull_skin_plots):
		finnpy_sr_bem.plot_skull_and_skin_models(ws_in_skull_vert, ws_in_skull_faces,
				                                                  ws_out_skull_vert, ws_out_skull_faces,
				                                                  ws_out_skin_vect, ws_out_skin_faces,
				                                                  anatomy_path, subj_name)

.. code-block::

	del ws_out_skull_vert; del ws_out_skull_faces; del ws_out_skin_vect; del ws_out_skin_faces

Finally, since this model is focused on MEG reconstruction, extracted surfaces not needed for source reconstruction are removed from memory (not hdd).

Using the skin and skull models, the BEM (boundary elements model) is computed.

.. code-block::

       (in_skull_reduced_vert, in_skull_faces, 
        in_skull_faces_area, in_skull_faces_normal, 
        bem_solution) = finnpy_sr_bem.calc_bem_model_linear_basis(in_skull_vert, in_skull_faces)

Afterwards, the anatomical section of this analysis is to reduce the number of vertices in the FreeSurfer extracted cortical models.
       
.. code-block::

       (lh_white_vert, lh_white_faces,
       rh_white_vert, rh_white_faces,
       lh_sphere_vert,
       rh_sphere_vert) = finnpy_sr_utils.read_surface_model(fs_subj_path)
       (octa_model_vert, octa_model_faces) = finnpy_sr_smm.create_source_mesh_model()
       (lh_white_valid_vert, rh_white_valid_vert) = finnpy_sr_smm.match_source_mesh_model(lh_sphere_vert, rh_sphere_vert, octa_model_vert)

The next step is to compute the forward model and constrain it.

.. code-block::

	rigid_mri_to_meg_trans = finnpy_sr_coreg.get_rigid_transform(coreg_rotors)
	rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
	(fwd_sol,
	lh_white_valid_vert, rh_white_valid_vert) = finnpy_sr_fwd.calc_forward_model(lh_white_vert, rh_white_vert,
		                                                                                                  rigid_meg_to_mri_trans, rigid_mri_to_meg_trans,
		                                                                                                  rec_meta_info, in_skull_reduced_vert, in_skull_faces,
		                                                                                                  in_skull_faces_normal, in_skull_faces_area,
		                                                                                                  bem_solution, lh_white_valid_vert, rh_white_valid_vert)
	optimized_fwd_sol = finnpy_sr_fwd.optimize_fwd_model(lh_white_vert, lh_white_faces, lh_white_valid_vert,
											  rh_white_vert, rh_white_faces, rh_white_valid_vert,
											  fwd_sol, rigid_mri_to_meg_trans)

The penultimate step is to inverse the forward model and compute the noise normalization vector. These may be used to transform sensor space data into subject specific source space.

.. code-block::

	(inv_trans, noise_norm) = finnpy_sr_inv.calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names, optimized_fwd_sol, rec_meta_info)

To enable cross subject comparability, the data needs to be morphed from subject specific space into a common source space, such as the one defined by fsaverage. As such, electrophysiological features of individual subjects become directly comparable within source space.

.. code-block::

	(fs_avg_trans_mat, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert) = finnpy_sr_utils.get_mri_subj_to_fs_avg_trans_mat(lh_white_valid_vert, rh_white_valid_vert, octa_model_vert,
																		     anatomy_path, subj_name, fs_path, overwrite = overwrite_mri_trans)

Having compute the inverse solution and the transformation into fsaverage source space concludes the MEG recording specific part of the source-reconstruction. The application of these matrices is explained in :ref:`source_reconstruction_application_label`.




