
Installation guide
==================

This guide explains how to install FinnPy automatically (via pip) or manually using the build package. FinnPy proper can be installed either via pip or manually build using the build package. Additionally, to use the statistics module offered by FinnPy, R needs to be installed and visible within the system path.

1. Automatic install
--------------------

FinnPy can be conveniently installed via pip running the following command:
Unix/macOS: python3 -m pip install --upgrade finnpy
Windows: py -m pip install --upgrade finnpy


2. Manual build
---------------

2.1. Open a terminal within the FinnPy directory

2.2. Install the build package::

    Unix/macOS: python3 -m pip install --upgrade build
    Windows: py -m pip install --upgrade build

2.3. Build FinnPy::

    Unix/macOS: python3 -m build
    Windows: py -m build

2.4. Install FinnPy::

    Unix/macOS: python3 -m pip install dist/finnpy-<version>-py3-none-any.whl
    Windows: py -m pip install dist/finnpy-<version>-py3-none-any.whl

For more information, see https://packaging.python.org/en/latest/tutorials/packaging-projects/

3. Installing R & enabling statistics
-------------------------------------

3.1. The latest version of R proper is available at https://www.r-project.org/ for Windows/macOS/Unix.

3.2. Additional R dependencies need to be installed. These can be installed either from within R or via running the following commands in a terminal::

    From within R: install.packages(c('Matrix', 'car', 'carData', 'lme4'))
    From terminal: R -e "install.packages(c('Matrix', 'car', 'carData', 'lme4'))"

