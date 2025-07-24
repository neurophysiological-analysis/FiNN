
.. _api_label:

API
===

This section documents the functions offered by FiNNPy.

Feature/Signal processing
-------------------------

FiNNpy offers a wide range of tools for general signal processing.


The Basic package
^^^^^^^^^^^^^^^^^

This :ref:`package <basic_package>` implements basic signal processing functionality.

| Common average Re-reference data via the :ref:`car_module` module.
| Modify data's sampling frequency via the :ref:`downsample_module` module.


The Cleansing package
^^^^^^^^^^^^^^^^^^^^^

This :ref:`package <cleansing_package>` implements several tools to identify and remove bad samples.

| Identify bad (e.g. oversaturated) channels graphically via the :ref:`bad_ch_module` module.
| Recover lost channels by averaging neighboring channels in the :ref:`ch_rest_module` module.
| Automatically filter outlier and harmonize samples via the :ref:`orem_module` module.


The Features package
^^^^^^^^^^^^^^^^^^^^

FiNNpy offers methods for spectral power analysis, 

| Measure spectral power via Welch's method in the :ref:`spectral_power` module.

cross frequency connectivity (cfc) via multiple metrics, 

| Measure cfc as :ref:`dmi_module` (DMI).
| Measure cfc as :ref:`mi_module` (MI).
| Measure cfc as :ref:`mvl_module` (MVL).
| Measure cfc as :ref:`plv_module` (PLV).

and same frequency connectivity (sfc),

| Measure sfc as :ref:`dac_module` (DAC).
| Measure sfc as :ref:`psi_module` (PSI).
| Measure sfc as :ref:`wpli_module` (wPLI).
| Measure sfc as :ref:`icoh_module` (IC).
| Measure sfc as :ref:`msc_module` (MSC).
| Measure sfc as :ref:`cc_module` module measures coupling as the complex coherency, a precursor to several sfc metrics.



The Filters package
^^^^^^^^^^^^^^^^^^^

This :ref:`package <filters_package>` implements several frequency domain filters.

| Employ a highly configure FIR filter via the :ref:`fir_module` module. 
| Employ a configurable Butterworth filter via the :ref:`butter_module` module.



The Source Reconstruction
-------------------------

Data may be projected from sensor to source space via FiNNpy's :ref:`Source Reconstruction package <srcrec_package>`.

Initial steps
^^^^^^^^^^^^^

| Compute a sensor covariance matrix via the :ref:`src_rec_sencov_module` module.
| Coregister M/EEG and anatomical spaces in the :ref:`src_rec_coreg_module` module.
| Call FreeSurfer from within Python in the :ref:`src_rec_fs_module` module.


Pipeline proper
^^^^^^^^^^^^^^^

| Compute a BEM model in the :ref:`src_rec_bem_module` module.
| Load highly accurate cortical data via the :ref:`src_rec_cort_module` module.
| Calculate a forward (source -> sensor space) model in the :ref:`src_rec_fwd_module` module.
| Inverse the forward model (sensor -> source space) via the :ref:`src_rec_inv_module` module.


Optional steps
^^^^^^^^^^^^^^

| To process source reconstructed data on a group level, the data needs to be projected from individual subject into a group space (fs-average space). This can be done via the :ref:`src_rec_fsavgproj_module` module.
| Localize cortical regions as per the Desikan-Killiany Atlas in the :ref:`src_rec_clust_module` module.


Supplementary
^^^^^^^^^^^^^

| Compute spheres for the source reconstruction via the :ref:`src_rec_spheremdl_module` module.
| Several supplementary tools used internally in the source reconstruction are bundled in the :ref:`src_rec_utils_module` module.


Feature analysis
----------------

FiNNpy offers advanced feature analysis tools.

The Statistics package
^^^^^^^^^^^^^^^^^^^^^^

This :ref:`package <stat_package>` implements methods for the statistical analysis of data.

| Linear mixed models may be configured and deployed via the :ref:`glmm_module` module.



Presentation
------------

FiNNpy offers In-Python and Blender based visualization tools in the :ref:`visualization <vis_package>` package.

Topoplots
^^^^^^^^^

| Plot 2D topoplots (sensor space) via the :ref:`topoplot_module` module.
| Plot 3D topoplots (source space) via the :ref:`plt_src_rec_module` module.


Other
^^^^^

| Plot volumetric data/pointcloids using the :ref:`volumetric_module` module.
| Convert nifti files to \*.obj files for use in visualizations in the :ref:`atlas_module` module.



Quality of life
---------------

FiNNpy offers a wide range of quality-of-life tools.


The File IO package
^^^^^^^^^^^^^^^^^^^

This :ref:`package <fileio_package>` enables reading/saving variables from/to the harddrive.

| Load/Save any data structure using the :ref:`datamanager_module` module. Even large data sets may be safely stored/loaded as these can be split into multiple auto-assembling subsets.
| Load BrainVision data using the :ref:`brainvision_module` module.


The Misc package
^^^^^^^^^^^^^^^^

This :ref:`package <misc_package>` provides a wide range of tools.

| Employ a highly configurable multiprocessing loop to perform multiple computations in parallel via the :ref:`tp_module` module. The loop is designed to minimize the memory footprint, enabling a maximum of concurrent evaluations.


