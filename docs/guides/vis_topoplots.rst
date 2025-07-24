
.. _vis_topoplots_label:

Source reconstruction
=====================

3D topoplots may be generated using pyvista. Depending on whether the visualization happens in subject space, fs-average space, or region-averaged fs-average space, corresponding methods are provided in the visualization module.

.. code-block::

   plot_subj_space(cort_mdl, color_data, signal_type = None, rec_meta_info = None, coreg = None, ch_names = None)

<or>

.. code-block::

   plot_fsavg_space(subj_to_fsavg_mdl, color_data, signal_type = None, rec_meta_info = None, coreg = None, ch_names = None)

<or>

.. code-block::

   plot_reg_avg(subj_to_fsavg_mdl, morphed_channels, color_data, signal_type = None, rec_meta_info = None, coreg = None, ch_names = None)

Of note, all of the above required intermediary byproducts (e.g. cort_mdl) are routinely produced during source reconstruction (see: :ref:`_src_rec_downstream_label`).

Alternatively, data may be exported into blender for advanced visualizations (see :ref:`vis_src_rec_blender_label`).
