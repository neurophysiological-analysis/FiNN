
.. _src_rec_recording_label:

MEG recording specific model computation
========================================

This part of the guide explains how the recording specific part of the MEG source reconstruction is computed. After hereon, the model is specified towards a distinct MEG recording (MEG markers/fiducials). As such, if there are multiple sessions for a single subject, the output of the previous step (:ref:`src_rec_subject_label`) may be used for multiple MEG sessions to avoid recomputing.

Important note if the following code is to be externally parallelized: Numpy auto-parallelizes specific functions. This behavior must be deactivated, if external parallelization is to be applied. This may be achieved with the following code-block:

.. code-block::

  import threadpoolctl
  threadpoolctl.threadpool_limits(limits=1, user_api='blas')

Herein, the individual commands of this section will be explained in a step-by-step fashion. The full pipeline looks as follows:
    
.. code-block::
  
  import mne
  import finnpy.src_rec.coreg
  import finnpy.src_rec.skull_skin_mdls
  import finnpy.src_rec.bem_mdl
  import finnpy.src_rec.cort_mdl
  import finnpy.src_rec.fwd_mdl
  import finnpy.src_rec.sen_cov
  import finnpy.src_rec.inv_mdl
  import finnpy.src_rec.subj_to_fsavg

  rec_meta_info = mne.io.read_info(data_path)
  coreg = finnpy.src_rec.coreg.run(subj_name, anatomy_path, rec_meta_info)
  
  skull_skin_mdls = finnpy.src_rec.skull_skin_mdls.read(anatomy_path, subj_name, "MEG", coreg)
  bem_mdl = finnpy.src_rec.bem_mdl.run(fs_path, skull_skin_mdls.in_skull_vert, skull_skin_mdls.in_skull_faces)
  cort_mdl = finnpy.src_rec.cort_mdl.get(anatomy_path, subj_name, coreg, bem_mdl.vert)

  fwd_mdl = finnpy.src_rec.fwd_mdl.compute(cort_mdl, coreg, rec_meta_info, bem_model)
  fwd_mdl = finnpy.src_rec.fwd_mdl.restrict(cort_mdl, fwd_sol, coreg)
  sen_cov = finnpy.src_rec.sen_cov.load(cov_path)
  inv_mdl = finnpy.src_rec.inv_mdl.compute(sen_cov, fwd_mdl, rec_meta_info)
  
  subj_to_fsavg_mdl = finnpy.src_rec.subj_to_fsavg.compute(cort_mdl, anatomy_path, subj_name, fs_path)

If not done previously, FreeSurfer needs to be initialized.

.. code-block::

  rec_meta_info = mne.io.read_info(data_path)
  coreg = finnpy.src_rec.coreg.run(subj_name, anatomy_path, rec_meta_info)
       							 
The first step is to compute a coregistration between the MEG and the MRI recordings. *subj_name* indicates the name of a specific subject while the *anatomy_path* points towards the anatomy of *all* subjects. Finally, *rec_meta_info* is derived via mne.io.read_info from a specific MEG recording, querying MEG registration points (fiducials).

.. code-block::

  skull_skin_mdls = finnpy.src_rec.skull_skin_mdls.read(anatomy_path, subj_name, "MEG", coreg)
  bem_mdl = finnpy.src_rec.bem_mdl.run(fs_path, skull_skin_mdls.in_skull_vert, skull_skin_mdls.in_skull_faces)
  cort_mdl = finnpy.src_rec.cort_mdl.get(anatomy_path, subj_name, coreg, bem_mdl.vert)

Afterwards, the previously extracted skull and skin models are loaded and scaled as identified through the coregistration. While the skull and skin models present the informational bases for the construction of BEM models, cortical models are read (and scaled) from freesurfer extracted files. 

.. code-block::

  fwd_mdl = finnpy.src_rec.fwd_mdl.compute(cort_mdl, coreg, rec_meta_info, bem_model)
  fwd_mdl = finnpy.src_rec.fwd_mdl.restrict(cort_mdl, fwd_sol, coreg)
  
Importantly, coregistration results should be manually verified: 

.. code-block::

  rec_meta_info = mne.io.read_info(data_path)
  meg_ref_pts = finnpy.src_rec.coreg.load_meg_ref_pts(rec_meta_info)
  (coreg, bad_hsp_pts) = finnpy.src_rec.coreg.run(subj_name, anatomy_path, rec_meta_info)
  finnpy.src_rec.coreg.plot_coregistration(coreg, meg_ref_pts, bad_hsp_pts, anatomy_path, subj_name)

BEM and cortical models are used in the computation of the forward model. This model describes how cortical data projects into sensor space (the inverse of the desired model). Afterwards, the MEG model may be restricted, limiting bipoles to an surface orthogonal orientation. This restriction effectively enforces a simplification of the model.

.. code-block::

  inv_mdl = finnpy.src_rec.inv_mdl.compute(sen_cov, fwd_mdl, rec_meta_info)

Finally, the model is inverted, describing how activity recorded in sensor space may have looked like in sensor space.

Additionally, one may want to compute a transformation from subject specific source space into fs-average source space to simplify group level analyses. As such, the corresponding model may be computed as follows.

.. code-block::

  subj_to_fsavg_mdl = finnpy.src_rec.subj_to_fsavg.compute(cort_mdl, anatomy_path, subj_name, fs_path)

This concludes the recording specific part of the MEG source reconstruction pipeline. The next step (the model's application) is explained in :ref:`src_rec_apply_label`.





