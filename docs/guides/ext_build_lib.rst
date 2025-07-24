
.. _ext_build_lib:

Building an external library
============================

This guide explains how to build an external library for use in Python. 

General
-------

Library code provided in FiNNpy must be downloaded from GitHub and can be found in folders starting with *finnpy_ext_[...]*. While multiple libraries may be build to expand functionality and speed up processing, individual use cases usually only require building a single library. 

Requirements
------------

Often, libraries are not self-contained, but rely on other libraries. While most of these are shipped with modern operating systems, at times individual libraries have to be installed to provide additional functionality. If this is the case, the libraries page comes with a requirements section listing the respective libraries. 

Windows
^^^^^^^

In Windows, libraries may be installed via Msys2 (https://www.msys2.org/). Via the therein included packet manager, libraries can be installed by typing, 

.. code-block::

  pacman -S library_name

Be aware to install the correct version of the library, for example, for use with MinGW, the common library prefix is mingw-w64-x86_64-<library_name>.

To make installed libraries and the tools provided in Msys2 available to Window's PowerShell, the following directories need to be added to the environment variable *Path*,

- C:\<msys64path>\mingw64\lib
- C:\<msys64path>\mingw64\include
- C:\<msys64path>\mingw64\bin
- C:\<msys64path>\usr\bin

In case of a first-time use, the following additional packages are needed:

.. code-block::

  pacman -S make
  pacman -S mingw-w64-x86\_64-toolchain

MacOs
^^^^^

In MacOs, libraries may be installed via homebrew,

.. code-block::

  homebrew install library_name

Linux
^^^^^

Similarly, on Linux a packet manager may be used to install the library, such as apt,

.. code-block::

  apt install library_name

or pacman,

.. code-block::

  pacman -S library_name

among others.

Use *make* to build
-------------------

Navigate the terminal (e.g. PowerShell) into the directory of the makefile. Execute the following command to produce the library, 

.. code-block::

  make all

Thereby, several new files should be generated, including a library file (windows: name.dll; mac: name.dylib; unix: name.so). 

How to use the build library
----------------------------

The final step is to pass the path to this file (including the file name) to the python call requesting it as a parameter. An example is provided below,

.. code-block::
  sen_cov = finnpy.src_rec.sen_cov.run([...]fast_eigendecomp_path = <path_to_new_library>, [...])
