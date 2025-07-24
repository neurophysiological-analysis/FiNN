
.. _install_label:

Installation
============

This guide explains how to install FinnPy automatically (via pip) or manually using the build package. FinnPy proper can be installed either via pip or manually build using the build package. Additionally, to use the statistics module offered by FinnPy, R needs to be installed and visible within the system path.

1. Automatic install
--------------------

FinnPy can be installed via pip running the following command,

.. code-block::

   Unix/macOS: python3 -m pip install --upgrade finnpy
   Windows: py -m pip install --upgrade finnpy


2. Manual build
---------------

2.1. Open a terminal within the FinnPy directory

2.2. Install the build package,

.. code-block::

   Unix/macOS: python3 -m pip install --upgrade build
   Windows: py -m pip install --upgrade build

2.3. Build FinnPy,

.. code-block::

   Unix/macOS: python3 -m build
   Windows: py -m build

2.4. Install FinnPy

.. code-block::

   Unix/macOS: python3 -m pip install dist/finnpy-<version>-py3-none-any.whl
   Windows: py -m pip install dist/finnpy-<version>-py3-none-any.whl

For more information, see https://packaging.python.org/en/latest/tutorials/packaging-projects/

3. Enabling statistics (optional)
---------------------------------

3.1. The latest version of R proper is available at https://www.r-project.org/ for Windows/macOS/Unix.

3.2. Additional R dependencies need to be installed. These can be installed either from within R or via running the following commands in a terminal

.. code-block::

   From within R: install.packages(c('Matrix', 'car', 'carData', 'lme4'))
   From terminal: R -e "install.packages(c('Matrix', 'car', 'carData', 'lme4'))"


4. External libraries (optional)
--------------------------------

Additional functionality may be enabled by compiling supplementary libraries provided on GitHub (https://github.com/neurophysiological-analysis/FiNN/tree/develop/finnpy). For more details, see :ref:`_ext_build_lib`

Currently, this is required to speed up M/EEG source reconstruction (:ref:`src_rec_ext_speedup_label`) and the export volumetric plots into blender (:ref:`vis_exp_blend_label`).