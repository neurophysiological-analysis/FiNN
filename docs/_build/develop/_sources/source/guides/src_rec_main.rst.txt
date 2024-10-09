
.. _src_rec_main_label:

Model computation & application
===============================

This guide explains how to apply source reconstruction for MEG using FiNNPy.

.. image:: img/MEG_source_reconstruction_simplified.png
   :alt: Graphic presentation of the relationship between skull & cortical model
   :align: center


Skull model processing
----------------------
The skull model is a geometrical description of the skull surface. For MEG analyses, a single layer model suffices (option for EEG source reconstruction to be added soon). The skull model is derived from T1 scans and extracted using the watershed algorithm of FreeSurfer. Its density is reduced to increase computeability and mathematical stability. Herein, it is employed to project MEG data from sensor space onto the skull surface.

Cortex model processing
-----------------------
The cortical model is a geometric description of the cortical structure. The cortical model may be directly extracted (using FreeSurfer) from T1 scans. Akin to the skull model, its density is reduced to increase computeability and mathematical stability. Herein, it will be employed to MEG activity from the skull surface onto the cortex proper.

Skull and cortex model fusion
-----------------------------
Skull and cortical models are fused to create the forward model. While the skull model describes the transition from sensor space to skull space, the cortex model may be employed to transition further into cortical space.

Application
-----------

An application example of source reconstruction for MEG is provided below. Generally, source reconstruction may be divided into four steps, 
1. Device specific steps
2. Subject specific steps
3. Recording specific steps
4. Model application

The following sections will provide examples on how to install FreeSurfer

0. :ref:`src_rec_install_label`

and  apply Finnpy to execute these steps.

1. :ref:`src_rec_device_label`
2. :ref:`src_rec_subject_label`
3. :ref:`src_rec_recording_label`
4. :ref:`src_rec_apply_label`

Additionally, likely pitfalls in source reconstruction are discussed in :ref:`src_rec_pitfalls_label`.




