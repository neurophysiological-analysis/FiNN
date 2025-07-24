
.. _src_rec_anatomy_label:

Anatomy extraction
==================

This part of the guide explains how to extract anatomy information from a patient's T1 using via FiNNpy's FreeSurfer wrappings. 

General
-------	

When called via FiNNpy, FreeSurfer commands will be executed in dedicated processes with dedicated temporary working spaces. As such, anatomy from multiple subjects may be extracted concurrently. If it is unknown how many processes may be executed concurrently without overburdening the PC, it is recommended to consecutively start processes as memory consumption is mostly stable throughout the extraction process. Time-wise, the extraction will last between 2 and 4 hours. FreeSurfer employs the below detailed folder structure. Thereof, only the project_data (name customizable) and anatomy path (name customizable) are to be created by the user. Subject folders and their contents are created by FreeSurfer. Folders marked as 'automatically generated' must not be created by the user as FreeSurfer will not operate correctly in case these are present.

::

    project_data (user created)
    └── anatomy_path (user created)
        ├── subject_a (automatically generated)
        │   ├── bem (automatically generated)
        │   │   └── watershed (automatically generated)
        │   ├── mri (automatically generated)
        │   │   └── transforms (automatically generated)
        │   ├── proj (automatically generated)
        │   └── surf (automatically generated)
        └── subject_b (automatically generated)


Initialization
--------------

Initially, FreeSurfer paths need to be configured for subsequent FreeSurfer calls.

.. code-block::

  finnpy_sr_utils.init_fs_paths(fs_path, anatomy_path)
       
*fs_path* points towards the FreeSurfer installation. The FreeSurfer folder should contain the 'bin' folder, license.txt, and sources.sh (among other files/directories).
*anatomy_path* contains sub-folders with anatomy folders for all subjects. FreeSurfer will fail if a subjects folder already exists.
*fastsurfer_path* FreeSurfer may also be called via FastSurfer. In this case, the FastSurfer path must also be indicated.
*fastsurfer_python_path* Option to provide a python path to FastSurfer if the system-wide Python installation is not to be used (default in Linux).
*freesurfer_license_path* Path to the FreeSurfer license file. This can be freely obtained (as of 2025/06/04) on the `FreeSurfer website <https://freesurfer.net/>`_. 

The above command may look as follows on a Windows machine.

.. code-block::

  finnpy_sr_utils.init_fs_paths(anatomy_path = "C:/.../anatomry/",
								fs_path = "C:/.../FreeSurfer/version/")


Be aware that the *anatomy_path* variable in this initialization step must point towards the directory of *all* subjects, not a specific one. 

Surface extraction
------------------

FiNNpy may either employ FreeSurfer to extract anatomical information from subject specific MRI scans or to copy data from fs-average, an averaged MRI scan provided by FreeSurfer. 

.. code-block::

  if (t1_path is not None and os.path.exists(t1_path)):
    finnpy.src_rec.extract_anatomy.extract_mri(anatomy_path, subj_name, t1_scan_file, fiducials_file, fiducials_path, mode, overwrite)
  else:
    finnpy.src_rec.extract_anatomy.copy_fsavg(freesurfer_path, anatomy_path, subj_name, overwrite)

*anatomy_path* contains sub-folders with anatomy folders for all subjects. FreeSurfer will fail if a subjects folder already exists.
*subj_name* Name of the subject.
*t1_scan_file* Path to the subject's T1 scan file. It is important to note that the patient's T1 scan file must *not* be placed within the subject's folder inside the anatomy folder. The subject folder must be created by FreeSurfer and the T1 file placed elsewhere. It will be copied by FreeSurfer during the extraction and may be deleted thereafter. 
*fiducials_file* Name of the fiducials file. If none is present, default fiducials are morphed from fs-average, defaults to None.
*fiducials_path* Path to the fiducials file. If none is present, default fiducials are morphed from fs-average, defaults to None. 
*mode* Mode is either "FreeSurfer" (default) or "FastSurfer".
*overwrite* Flag whether to overwrite the files of a subject in case the respective subject's folder already exists. Defaults to False.

*freesurfer_path* points towards the FreeSurfer installation. The FreeSurfer folder should contain the 'bin' folder, license.txt, and sources.sh (among other files/directories).

Let's take a closer look at this code example.

.. code-block::

  if (t1_path is not None and os.path.exists(t1_path)):
           

Initially, it is checked whether the *t1_path* variable is not None and whether the indicated directory does actually exist. If this is the case, the following line is executed:

.. code-block::

  finnpy.src_rec.extract_anatomy.extract_mri(anatomy_path, subj_name, t1_scan_file, fiducials_file, fiducials_path)
    
*fiducials_file* and *fiducials_path* may be employed to include subject specific fiducials, otherwise these will be compute from *fsaverage*. Finally, the *overwrite* flag may be used to allow overwriting a subject's folder. As such, herein anatomy is extracted from a T1 scan. Be aware, this process may likely take several hours to run (reference: Linux on a 5950X - 1.5 h - 4 h). Therefore, it is highly recommended to execute this command for several subjects in parallel.

.. code-block::

  finnpy.src_rec.extract_anatomy.copy_fsavg(fs_path, anatomy_path, subj_name)
       
Alternatively, this line of code copies reference anatomy from *fsaverage* into the folder indicated by *subj_name*. *fsaverage* is provided by FreeSurfer and found in *FreeSurfer/version/subjects/* or *FreeSurfer/subjects/*

Either of the above steps will create the following folders, *bem*, *bem/watershed*, *mri*, *mri/transforms*, *surf*, and partially populate these.

Get skull and skin models
-------------------------

The next step is to extract skull and skin models using FiNNpy's wrapping of the FreeSurfer watershed algorithm.

.. code-block::

  finnpy.src_rec.extract_anatomy.get_skull_skin(anatomy_path, subject_name, preflood_height = 25)

*preflood_height* The preflood parameter controls how *strict* or *liberal* the extraction of these models is performed. 

It is important that the correct extraction is manually verified, as these are highly dependent on adjusting the preflood height. Results may be visualized as follows:

.. code-block::
  
  import finnpy.src_rec.skull_skin_mdls
  
  (vert, faces) = finnpy.src_rec.extract_anatomy.read_skin_skull(anatomy_path, subj_name, "MEG")
  finnpy.src_rec.extract_anatomy.plot_skin_skull(vert, faces, anatomy_path, subj_name, block)

*vert* and *faces* are lists of vertices/faces of the skin, outer skull, and inner skull models.
*block* indicates whether the program is to stop and display an interactive matplotlib image for visualization.


Get head surface model
----------------------

The next step is to compute the seghead models, these are required for M/EEG co-registration:

.. code-block::

  finnpy.src_rec.extract_anatomy.get_head_model(anatomy_path, subj_name)

Enabling group analyses
-----------------------

If features are to evaluated on the group level (or for distinct regions of an atlas), a transformation to fs-average (standard-template) needs to be compute for every subject, making all data resides within the same space. To enable these, the following FreeSurfer call is needed.

.. code-block::

  finnpy.src_rec.subj_to_fsavg.prepare(anatomy_path, subj_name)

Next steps
----------

Importantly, if maintainable (memory-wise), subjects should be processed in parallel to minimize computation time. As such, the entire herein discussed part of the reconstruction pipeline may be easily parallelized on the outside, for example using finnpys timed pool for multithreading (finnpy.misc.timed_pool).

This concludes MRI extraction segment of the source reconstruction pipeline. The next step (MEG recording specific) is explained in :ref:`src_rec_sensors_label`.
