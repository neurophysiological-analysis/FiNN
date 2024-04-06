
.. _dac_configurator_label:

DAC configurator
================

A primary application case of FiNNPy is the estimation of neuronal network dynamics using same and cross frequency coupling. This guide is focused on calibrating the directionalized absolute coherence (DAC) metric which is employed for same frequency connectivity estimates. Generally speaking, the reliability of same frequency connectivity estimates depends on a number of factors. While some of these vary in between the different same frequency connectivity metrics, the reliability limiting factor *amount of available data per sample* applies to all. Furthermore, in the case of DAC, limitations from magnitude squared coherence (MSC), imaginary coherence (IC), and phase slope index (PSI) are inherited.

A general review of this topic is provided in this `video <https://youtu.be/sn7Xnt6zIiE>`_.

Jump to the :ref:`dac_configurator_application_label` section.

Theory
------

The following paragraphs explain the theoretical limitations of magnitude squared coherence (MSC), imaginary coherence (IC), and the phase slope index (PSI):

Limitations of the MSC
^^^^^^^^^^^^^^^^^^^^^^

As magnitude squared coherence (MSC) is calculated in the frequency domain from the cross spectra of two signals, its limitations are closely aligned with the limitations of power spectral density estimates. Although transforming a signal from the time into the frequency domain is, in an ideal world, a lossless transformation, careful consideration has to be given to the selection of the frequency bin width. This parameter can be controlled by the Fast Fourier Transform (FFT) window width. Exemplary, if a power spectral density (PSD) was to be estimated, too wide frequency bins would result in an overly smooth, smeared estimate whereas too small (padded) frequency bins would result in a pointy and highly erratic PSD estimate. Both are different to the expected result.

A similar logic applies to same frequency connectivity estimates using the MSC. If too large FFT windows (wide frequency bins) are chosen, the MSC estimate becomes too smooth whereas if too small (padded) FFT windows are chosen, the MSC estimate becomes erratic.

Limitations of the IC
^^^^^^^^^^^^^^^^^^^^^

Imaginary coherence (IC) is calculated similar to MSC, thus inheriting its limitations. Yet, as only part of the estimated cross spectral density is evaluated, an addition limitation has to be considered: 

1. The bounds of IC depend on the phase difference between the signal at the source and target sites. 

The implications of this limitation are twofold. On the one hand, IC is not increased by zero lag cross spectral coupling, commonly referred to as volume conduction, as the bounds for zero phase differences are [0, 0]. On the other hand, the bounds for any non zero phase lag cross spectral coupling is not [-1, 1], but on a continuous scale between [0, 0] and [-1, 1]. The bounds are widest for uneven integer multiple of 90° phase shifts. Hence, a low IC estimate can be caused by either a general lack of coupling between A and B, or a close to an even integer multiple of 180° phase shift. Yet, to determine the presence of volume conduction a threshold is needed under which any signal is considered volume conduction.

Limitations of the PSI
^^^^^^^^^^^^^^^^^^^^^^

Although sharing the estimate of cross spectra between signals with the MSC and IC, the phase slope index (PSI) estimates connectivity through another means. It evaluates whether the phase difference from a range of cross spectral estimates is monotonically increasing or decreasing. This way, the fact that most *natural* electrophysiological signals are smooth in the frequency domain and not of fixed, but much rather bounded frequency, is leveraged.

Hence, it requires not a single cross spectrum estimate, but multiple from slightly different frequencies. Yet, as multiple cross spectra estimates are required, the frequency bin width of the FFT is of increased importance. If the frequency bins are designed too small, the PSI estimates will be erratic; whereas if the bins are too wide, they are too few in number to reliably evaluate whether the signal appears first at the source or the target. 

.. _dac_configurator_application_label:

Application
-----------

Calibrating same frequency connectivity methods is vital if reliable results are to be generated. To this end, the DAC configurator can be used to verify a correct calibration. The DAC configurator allows to evaluate the following assertions: 

1. The data quantity to frequency resolution relationship is sufficient to estimate the connectivity **amplitude**.
2. The selected frequency bin width is sufficient to estimate the **sequence** of events (direction).
3. **Volume conductance** is successfully detected with minimal false positives.

The following parameters have to be configured to confirm the aforementioned assertions:

General parameters
^^^^^^^^^^^^^^^^^^

These parameters are determined by the setup and the to be investigated frequency band. These parameters should not be changed in the parameter tuning process::

    frequency_sampling = 5500
    frequency_min = 25
    frequency_max = 35

Data parameters
^^^^^^^^^^^^^^^

The DAC configurator can be used with provided data from FiNNPy using GIT-LFS or ones own data. Generally, it is recommended to use data acquired with the setup which is to be used in the investigation. If the provided data is to be used, this needs to be downloaded via GIT-LFS and the data needs to be set to *None*.::

    input_data = None

Specific parameters
^^^^^^^^^^^^^^^^^^^

These parameters are to be used to tailor the connectivity estimate to the data at hand to to verify the correctness of this configuration. While the minimal angle threshold and the volume conductance ratio determine whether volume conductance is present, the frequency bin sizes determines the frequency resolution used in the DAC estimate.::

    frequency_bin_sz = 0.5
    minimal_angle_thresh = 10
    volume_conductance_ratio = 0.3

The following rules describe how to tune the *specific parameters* for optimal connectivity estimates:

1. A higher frequency resolution will improve the estimate of the sequence of events (directionality) at the cost of decrease amplitude estimation reliability.
2. The minimal angle threshold [0°, 180°] determines whether the connectivity estimate of a single frequency bin is considered volume conductance or not.
3. The volume conductance ratio determines the ratio single frequency bins with volume conductance needed to flag the estimate as volume conductance.

Remarks
^^^^^^^

Generally, it is not possible to distinguish between low connectivity and volume conductance using methods based on cross spectral estimates since in both situations, the imaginary part of the cross spectra default to zero due to either inconsistent phase differences between the signals (low connectivity) or close to zero phase shift (volume conductance). Hence, if low connectivity estimates are to be performed, it is recommended to set the minimal angle threshold to a comparatively low level.

