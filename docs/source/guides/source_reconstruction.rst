
.. _source_reconstruction_label:

Source reconstruction
=====================

This guide explains how to apply source reconstruction for MEG using FiNNPy.

.. image:: ../images/MEG_source_reconstruction_simplified.png
   :scale: 100 %
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

Model application
-----------------

Model application is outlined in the code below.


.. code-block::

       TEST

Common errors
-------------

MEG and MRI coregistration
^^^^^^^^^^^^^^^^^^^^^^^^^^
The coregistration between MEG and MRI space has been left unchecked. This step must be manually verified

Sensor noise covariance
^^^^^^^^^^^^^^^^^^^^^^^
The sensor noise covariance is faulty. This may be investigated by adding a power spike to a sensor-space channel and investigate where it is projected.

Improper skull model
^^^^^^^^^^^^^^^^^^^^
The skull model was improperly extracted. This step must be manually verified




