
.. _src_rec_subject_label:

MEG recording agnostic model computation
========================================

This part of the guide explains how the subject specific components of the MEG source reconstruction are computed. This model is not specified towards a distinct MEG recording (MEG markers).

Herein, the individual commands of this section will be explained in a step-by-step fashion. Importantly, this code block may easily be parallelized externally across individual subjects, for example employing finnpys timed pool (finnpy.misc.timed_pool). Thereby, processing time may be reduced from approx. 2-4 hours x subjects to 2-4 hours total.
    
.. code-block::
	
  import finnpy.src_rec.freesurfer
  import finnpy.src_rec.subj_to_fsavg

  finnpy.src_rec.freesurfer.init_fs_paths(anatomy_path, fs_path)
  if (t1_path is not None and os.path.exists(t1_path)):
    finnpy.src_rec.freesurfer.extract_mri_anatomy(anatomy_path, subj_name, t1_scan_file)
  else:
    finnpy.src_rec.freesurfer.copy_fsavg_anatomy(fs_path, anatomy_path, subj_name)
  finnpy.src_rec.freesurfer.extract_skull_skin(anatomy_path, subject_name, preflood_height = 25)
  finnpy.src_rec.freesurfer.calc_head_model(anatomy_path, subj_name)
  
  finnpy.src_rec.subj_to_fsavg.prepare(anatomy_path, subj_name)
	
Initially, free surfer paths need to be configured for subsequent FreeSurfer calls.

.. code-block::

  finnpy_sr_utils.init_fs_paths(fs_path, anatomy_path)
       
*fs_path* points towards the FreeSurfer installation. The FreeSurfer folder should contain the 'bin' folder, license.txt, and sources.sh (among other files/directories).
*anatomy_path* contains sub-folders with anatomy folders for all subjects. FreeSurfer will fail if a subjects folder already exists.

Important: The T1 scan to be used for the extract must be placed outside the anatomy folder, otherwise FreeSurfer will not a process a subject who already maintains a subfolder in the anatomy directory. 

As such, FiNN adheres to the following, FreeSurfer aligned, organizational structure:

::

    project_data
    └── anatomy_path
        ├── subject_a (must be created via FreeSurfer, *do not create this folder manually*)
        │   ├── bem (automatically generated at later steps)
        │   │   └── watershed (automatically generated at later steps)
        │   ├── mri (automatically generated at later steps)
        │   │   └── transforms (automatically generated at later steps)
        │   ├── proj (automatically generated at later steps)
        │   └── surf (automatically generated at later steps)
        └── subject_b

The above command may look as follows on a Windows machine.

.. code-block::

  finnpy_sr_utils.init_fs_paths(anatomy_path = "C:/.../anatomry/",
				   fs_path = "C:/.../FreeSurfer/version/")


Be aware that the *anatomy_path* variable in this initialization step must point towards the directory of *all* subjects, not a specific one.       
After this initial setup, FreeSurfer may be used to either extract anatomical information from subject specific MRI scans or copy data from fs-average, an averaged MRI scan provided through FreeSurfer.

.. code-block::

  if (t1_path is not None and os.path.exists(t1_path)):
    finnpy.src_rec.freesurfer.extract_mri_anatomy(anatomy_path, subj_name, t1_scan_file)
  else:
    finnpy.src_rec.freesurfer.copy_fsavg_anatomy(fs_path, anatomy_path, subj_name)


Let's take a closer look at this code example.

.. code-block::

  if (t1_path is not None and os.path.exists(t1_path)):
           

Initially, it is checked whether the *t1_path* variable is not None and whether the indicated directory does actually exist. If this is the case, the following line is executed:

.. code-block::

  finnpy.src_rec.freesurfer.extract_mri_anatomy(anatomy_path, subj_name, t1_scan_file, fiducials_file, fiducials_path)
    
*fiducials_file* and *fiducials_path* may be employed to include subject specific fiducials, otherwise these will be compute from *fsaverage*. Finally, the *overwrite* flag may be used to allow overwriting a subject's folder. As such, herein anatomy is extracted from a T1 scan. Be aware, this process may likely take several hours to run (reference: linux on a 5950X - 1.5 h - 4 h). Therefore, it is highly recommended to execute this command for several subjects in parallel.

.. code-block::

  finnpy.src_rec.freesurfer.copy_fsavg_anatomy(fs_path, anatomy_path, subj_name)
       
This line of code copies reference anatomy from *fsaverage* into the folder indicated by *subj_name*. *fsaverage* is provided by FreeSurfer and found in *FreeSurfer/version/subjects/* or *FreeSurfer/subjects/*

Either of the above steps will create the following folders, *bem*, *bem/watershed*, *mri*, *mri/transforms*, *surf*, and partially populate these.

The next step is to extract skull and skin models. These are extracted using freesurfers watershed algorithm. The preflood parameter controls how *strict* or *liberal* the extraction of these models is performed. 

.. code-block::

  finnpy.src_rec.freesurfer.extract_skull_skin(anatomy_path, subject_name, preflood_height = 25)

Important, watershed extraction results should be manually verified. As the watershed extraction results are used to construct the skull and skin models, their respective plotting routines are also found in this module: 

.. code-block::
  
  import finnpy.src_rec.skull_skin_mdls
  
  skull_skin_mdl = finnpy.src_rec.skull_skin_mdls.read(anatomy_path, subj_name, "MEG")
  finnpy.src_rec.skull_skin_mdls.plot(skull_skin_mdl, anatomy_path, subj_name)

Another subject specific step is the computation of a seghead models. This may be performed as indicated below:

.. code-block::

  finnpy.src_rec.freesurfer.calc_head_model(anatomy_path, subj_name)

Finally, if the data is to be transformed from a subject specific into fs-average space (if desired), the following line of code may be performed herein to speed up later computations.

.. code-block::

  finnpy.src_rec.subj_to_fsavg.prepare(anatomy_path, subj_name)

Importantly, if maintainable (memory-wise), subjects should be processed in parallel to minimize computation time. As such, the entire herein discussed part of the reconstruction pipeline may be easily parallelized on the outside, for example using finnpys timed pool for multithreading (finnpy.misc.timed_pool).

This concludes the MEG subject specific part of the source reconstruction pipeline. The next step (MEG recording specific) is explained in :ref:`src_rec_recording_label`.




