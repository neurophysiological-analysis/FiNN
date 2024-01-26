
.. _source_reconstruction_general_label:

MEG recording agnostic model computation
========================================

This part of the guide explains how the MEG recording agnostic model is computed. This model is not specified towards a distinct MEG recording (MEG markers).

Herein, the individual commands of this section will be explained in a step-by-step fashion.
    
.. code-block::
	
	finnpy.source_reconstruction.utils.init_fs_paths(anatomy_path, fs_path)
    
	if (t1_path is None):
		finnpy.source_reconstruction.mri_anatomy.copy_fs_avg_anatomy(fs_path, anatomy_path, subj_name, overwrite = overwrite_fs_extract)
	else:
		finnpy.source_reconstruction.mri_anatomy.extract_anatomy_from_mri_using_fs(subj_name, t1_path, overwrite = overwrite_ws_extract)
	(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names) = finnpy.source_reconstruction.sensor_covariance.get_sensor_covariance(file_path = sensor_cov_path, cov_path = cov_path, overwrite = overwrite_sensor_cov)
	
Initially, free surfer paths need to be configured for subsequent FreeSurfer calls.

.. code-block::

	finnpy.source_reconstruction.utils.init_fs_paths(anatomy_path,
       							 fs_path)
       
*anatomy_path* contains sub-folders with anatomy folders for all subjects. FreeSurfer will fail if a subjects folder already exists.
*fs_path* points towards the FreeSurfer installation. The FreeSurfer folder should contain the 'bin' folder, license.txt, and sources.sh (among other files/directories).

Important: The T1 scan to be used for the extract must be placed outside the anatomy folder, otherwise FreeSurfer will think the patient already exists.

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

As such, the above command may look as follows on a Windows machine.

.. code-block::

	finnpy.source_reconstruction.utils.init_fs_paths(anatomy_path = "C:/.../anatomry/",
							 fs_path = "C:/.../FreeSurfer/version/")


Be aware that the *anatomy_path* variable in this initialization step must point towards the directory of *all* subjects, rather than a single one.       
After this initial setup, FreeSurfer may be used to either extract anatomical information from subject specific MRI scans or copy data from fs-average, an averaged MRI scan provided through FreeSurfer.

.. code-block::

	if (t1_path is None):
		finnpy.source_reconstruction.mri_anatomy.copy_fs_avg_anatomy(fs_path,
									     anatomy_path,
									     subj_name,
									     overwrite = overwrite_fs_extract)
	else:
		finnpy.source_reconstruction.mri_anatomy.extract_anatomy_from_mri_using_fs(subj_name,
											   t1_path,
											   overwrite = overwrite_fs_extract)


Let's take a closer look at this code example.

.. code-block::

	if (t1_path is None):
           

Inititially, it is checked whether the *t1_path* variable is None. If this is the case, the following line is executed:

.. code-block::

	finnpy.source_reconstruction.mri_anatomy.copy_fs_avg_anatomy(fs_path,
								     anatomy_path,
								     subj_name,
								     overwrite = overwrite_fs_extract)
           

This line of code copies reference anatomy from *fsaverage* into the folder indicated by *subj_name*. *fsaverage* is provided by FreeSurfer and found in *FreeSurfer/version/subjects/* or *FreeSurfer/subjects/*. However, if anatomy is available, it is extracted from T1 scans, pointed to by *t1_path*. As such, if a T1 scan file is provided, the following line of code is executed. Be aware, this function may likely take several hours to run (reference: linux on a 5950X - 1.5 h - 2 h).

.. code-block::

       finnpy.source_reconstruction.mri_anatomy.extract_anatomy_from_mri_using_fs(subj_name,
       										  t1_path,
       										  overwrite = overwrite_fs_extract)
       

*fiducials_file* and *fiducials_path* may be employed to include subject specific fiducials, otherwise these will be compute from *fsaverage*. Finally, the *overwrite* flag may be used to allow overwriting a subject's folder.

Hence, either of the above steps will create the following folders, *bem*, *bem/watershed*, *mri*, *mri/transforms*, *surf*, and partially populate these.

The final step in the MEG recording agnostic preparation is the computation of the sensor noise covariance. While it isn't needed until much later in the source reconstruction process, it is MEG recording agnostic.


.. code-block::

	(sensor_cov_eigen_val,
 	 sensor_cov_eigen_vec,
 	 sensor_cov_names) = finnpy.source_reconstruction.sensor_covariance.get_sensor_covariance(file_path = sensor_cov_path,
 	 											  cov_path = cov_path,
 	 											  overwrite = overwrite_sensor_cov)


This concludes the MEG recording agnostic part of the source reconstruction procedure. The next steps (MEG recording specific) are explained in :ref:`_source_reconstruction_specific_label`.




