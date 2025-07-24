
.. _src_rec_fs_avg_proj_label:

Shared group space projections
==============================

This guide explains how to project data from an individual subjects space into a shared group space (i.e. fs-average space). 
This step is necessary if the data is to be analyzed on a group scale or projected onto an atlas.

Group space projections
-----------------------

The following code demonstrates how to project data into fs-average space. 

.. code-block::

  finnpy.src_rec.subj_to_fsavg.prepare(fs_path, anatomy_path, subj_name)
  subj_to_fsavg_mdl = finnpy.src_rec.subj_to_fsavg.compute(cort_mdl, anatomy_path, subj_name, fs_path, overwrite)
  src_fsavg_data = finnpy.src_rec.subj_to_fsavg.apply(subj_to_fsavg_mdl, src_data)
  (src_avg_data, morphed_channels, region_names) = finnpy.src_rec.avg_src_reg.run(src_fsavg_data, subj_to_fsavg_mdl, fs_path)
       
*fs_path* points towards the FreeSurfer installation. The FreeSurfer folder should contain the 'bin' folder, license.txt, and sources.sh (among other files/directories).
*anatomy_path* contains sub-folders with anatomy folders for all subjects. FreeSurfer will fail if a subjects folder already exists.
*subj_name* is the name of the subject.
*overwrite* Boolean flag whether to overwrite MRI maps. Defaults to False.
*src_data* Source space data. Currently in the source space of the respective subject.
*subj_to_fsavg_mdl* Preliminary result.
*src_fsavg_data* Data projected into fs-average space.
*src_avg_data* Data segmented according to the Desikan-Killiany atlas.
*morphed_channels* List matching fs-average channels with atlas regions.
*region_names* Names of the respective Desikan-Killiany atlas regions.

Projection into atlas space
---------------------------

After projecting the data from their individual subject's space into fs-average space for group analyses, the Desikan-Killiany atlas
may be employed to cluster regions, significantly decreasing channel count and also variability across subjects.

.. code-block::

  (src_avg_data, morphed_channels, region_names) = finnpy.src_rec.avg_src_reg.run(src_fsavg_data, subj_to_fsavg_mdl, fs_path)

