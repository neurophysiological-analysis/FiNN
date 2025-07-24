
.. _guides_label:

Guides
======

Although FinnPy provides numerous tools for the analysis of electrophysiological data, the quality of the results derived using these tools equally depends on correct application and configuration of the former. This section contains guides tackling specific topics from the field of electrophysiological data analysis, providing guide to users of this and other frameworks.

Data preparation
----------------

Discussion on :ref:`data_prep_label` concepts of data preparation prior to feature extraction.

Feature generation
------------------

Same frequency coupling may be evaluated using directional absolute coherence, one metric of many provided in FiNNpy. To provide robustness against volume conductance, yet highly sensitive measurements, it is recommended to adjust the metric to the specific use-case. To assist in this effort, FiNNpy provides the :ref:`sfc_dac_configurator_label`.

Source reconstruction
---------------------

Explains how :ref:`src_rec_main_label` in the context of M/EEG data. 

Statistics
----------

:ref:`stats_general_label` overview on important statistic concepts relevant for any type of statistical analysis.

Explains how to use :ref:`stats_lmm_label` supported via FiNNpy to model and evaluate data relationships.

Visualization
-------------

To support visualizing the results of cortical analyses, FiNNpy provides functionality to produce :ref:`vis_topoplots_label` and for more complex visualizations the option to generate :ref:`In-Blender Cortical Plots`.

Additionally, FiNNpy may be used to visualize :ref:`vis_volumetric_label`, either within Python or within Blender. As structures from (sub-)cortical atlases may be imported, this provides an attractive opportunity to generate high quality visualizations for subcortical data. 

.. toctree::
   :hidden:
   
   guides/dac_configurator
   guides/src_rec_apply
   guides/src_rec_device
   guides/src_rec_recording
   guides/src_rec_subject
   guides/src_rec_install
   guides/src_rec_main
   guides/src_rec_pitfalls
   guides/src_rec_vis

   
   






