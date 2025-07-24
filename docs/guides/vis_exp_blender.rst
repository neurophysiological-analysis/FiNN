
.. _vis_exp_blend_label:

Exporting volumetric plots to blender
=====================================

This guide explains how to build and use external libraries to export volumetric plots into a blender-readable format. This feature is required to export volumes into blender for improved visualization. The relevant code can be found in the following folder: *finnpy_ext_export_openvdb*. The required shared libraries (for use in Python) can be build via make. For a more detailed guide, see :ref:`ext_build_lib`.

Requirements
------------

This library requires the presence of the following libraries:

- cmake (https://cmake.org/)
- boost (https://www.boost.org/releases/latest/)
- blosc (https://github.com/Blosc/c-blosc)
- tbb
- openvdb (https://github.com/AcademySoftwareFoundation/openvdb)

Windows
^^^^^^^

Boost, blosc, and tbb can be installed via Msys2,

.. code-block::

  pacman -S mingw-w64-x86_64-boost
  pacman -S mingw-w64-x86_64-c-blosc
  pacman -S mingw-w64-x86_64-tbb

Cmake can be downloaded and installed from `here <https://cmake.org/>`_. 

MacOs
^^^^^

Cmake, boost, blosc, and tbb can be installed via Homebrew,

.. code-block::

  brew install boost c-blosc
  brew install boost tbb
  brew install boost tbb

Linux
^^^^^

Cmake, boost, blosc, and tbb can be installed via apt 

.. code-block::

  apt install libboost-all-dev
  apt install libblosc-dev
  apt install libtbb-dev
  apt install cmake

or pacman

.. code-block::
  
  pacman -S boost
  pacman -S blosc
  pacman -S tbb
  pacman -S cmake


All
^^^

After having provided its dependencies, OpenVDB may be installed as follows. First, the GitHub repository needs to be downloaded (https://github.com/AcademySoftwareFoundation/openvdb). Subsequently, a terminal (e.g. PowerShell) should be opened and navigated to the extracted GitHub repository. Having entered openvdb-master, a build folder needs to be created (*mkdir build*), navigated into (*cd build*), and cmake called on the parent directory (*cmake ..*). 

Subsequently, OpenVDB needs to be build via *make*.

Finally, OpenVDB needs to be either added to the path or placed in the same directory as *finnpy_ext_export_openvdb*. 

Building the library
--------------------

The make command for *finnpy_ext_export_openvdb* may be called, building the library for use in Python. 
