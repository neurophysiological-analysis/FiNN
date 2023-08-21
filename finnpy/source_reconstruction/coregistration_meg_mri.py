'''
Created on Oct 12, 2022

@author: voodoocode
'''

import functools

import numpy as np
import scipy.optimize
import scipy.spatial
import scipy.linalg

import nibabel.freesurfer

import finnpy.source_reconstruction.utils as finnpy_utils
import os

import mayavi.mlab

import mne

import warnings

import shutil

import finnpy.file_io.data_manager as dm

