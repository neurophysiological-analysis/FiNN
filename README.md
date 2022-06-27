# **Fi**nd **N**europyhsiological **N**etworks (FiNNPy) using **Py**thon

A Python Toolbox for the analysis of electrophysiological data

The main directory *finnpy* contains the toolbox itself, while *demo* contains a number of demo files describing end-point functionality of the toolbox. *tests* on the other hand contains automated tests to perform unit tests in order to evaluate the state of the toolbox.


## Branches

FiNNPy offers two branches:
- Stable: Contains only stable releases (passed all tests)
- Develop: Contains all latest changes and may be unstable (only unit tested)

## Features

Currently implemented features in this toolbox. Full documentation is accessible in doc/\_build/index.html. Documentation of the development branch is available at https://neurophysiological-analysis.github.io/FiNN/.

   - Artifact rejection
        - Identification of bad channels and (optional) restoration of those from neighboring channels
        - Removal of statistical outliers (identified via z-transform)
   - Basic
        - Functionality to downsample data
        - Common average re-referencing
   - Connectivity
        - Cross frequency coupling
             - Implemented methods are *phase lock value*, *modulation index*, *mean vector length*, and *direct modulation index*.
        - Same frequency coupling
             - Implemented methods are *weighted phase lag index (wPLI)*, *phase slope index*, *magnitude squared coherence*, *imaginary coherence*, and *directional absolute coherence*.
   - File IO
        - Load data from brain vision recordings.
        - A data manager to save & load (unbalanced) data with a minimal memory footprint.
   - Frequency spectrum filters
        - Easy access to scipys butterworth filter
        - An overlap add based implementation of an FIR filter
   - Misc.
        - A parallelization loop for easy access to parallel processing. Accustomed to minimal memory footprint and resource consumption
   - Statistics
        - Easy within Python access to generalized linear mixed models
   - Visualization
        - Plot topographical maps of EEG recordings. Effect sizes may be visualized by color whereas significance may be visualized using dots (black - n.s., half-filled - significant before multiple comparison correction, white - significant after multiple comparison correction)



## Requirements

- Python 3.8 or above.
- R and R packages
  - lm4
  - car
  - carData
  - Matrix

