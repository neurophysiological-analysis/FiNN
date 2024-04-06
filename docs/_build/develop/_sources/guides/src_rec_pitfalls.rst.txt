
.. _src_rec_pitfalls_label:

Likely pitfalls
===============

This guide explains how to deal with likely pitfalls in source reconstruction for MEG using FiNNPy.

MEG and MRI coregistration
--------------------------
The coregistration between MEG and MRI space has been left unchecked. This step must be manually verified. This may be done as follows:

.. code-block::

  rec_meta_info = mne.io.read_info(data_path)
  meg_ref_pts = finnpy.src_rec.coreg.load_meg_ref_pts(rec_meta_info)
  (coreg, bad_hsp_pts) = finnpy.src_rec.coreg.run(subj_name, anatomy_path, rec_meta_info)
  finnpy.src_rec.coreg.plot_coregistration(coreg, meg_ref_pts, bad_hsp_pts, anatomy_path, subj_name)
       
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

  skull_skin_mdl = finnpy.src_rec.skull_skin_mdls.read(anatomy_path, subj_name, "MEG")
  finnpy.src_rec.skull_skin_mdls.plot(skull_skin_mdl, anatomy_path, subj_name)
       
Producing the following output:

.. image:: img/MEG_source_reconstruction_ws.png
   :alt: Skull model example
   :width: 500
   :align: center



