'''
Created on Jun 28, 2017

:author: maximilianscherer
'''

import mne
import numpy as np

def run(vhdr_path, badChList = [], montage_file = None, montage_path = None, preload = False, tmin = None, tmax = None, verbose = None):
    """
    Load data recorded via the brain vision software.
    
    :param vhdr: Path to the vhdr file.
    :param misc_ch: MISC - channels.
    :param montage_file: File name of the montage file. 
    :param montage_path: Path to the montage file. 
    :param tmin: Start of the returned recording segment in seconds.
    :param tmax: End of the returned recording segment in seconds.
    :param preload: If 'true' the data is loaded directly in this function. If 'false' only a container is created.
    :param verbose: Verbose the command line output.
    
    :return: A raw data object containing either the data or a dummy container and the address of the 'to be loaded' data.
    """
    
    if (badChList is None):
        metaInfo = mne.io.read_raw_brainvision(vhdr_path, misc = badChList, verbose = verbose)
    else:
        metaInfo = mne.io.read_raw_brainvision(vhdr_path, verbose = verbose)
    if (tmin is not None or tmax is not None):
        if (tmin is None):
            tmin = 0
        metaInfo= metaInfo.crop(tmin = tmin, tmax = tmax)
    
    if (montage_path is not None):                                      
        montage = mne.channels.read_montage(kind = montage_file, path = montage_path)
        for chIdx in range(0, len(montage.ch_names)):
            if (type(montage.ch_names[chIdx]) != np.str_):
                montage.ch_names[chIdx] = montage.ch_names[chIdx]
        metaInfo.set_montage(montage = montage)
    
    if (preload):
        metaInfo.load_data()
    
    return metaInfo

def readMarker(vmrkPath):
    """
    Extracts the markers from a brain vision recording.
    
    :param vmrk_path: Path to the marker file.
    :return: Returns a list of markers. Every marker is a dictionary with a type and an index.
    """
    mrkFile = open(vmrkPath, "r")
    
    lines = mrkFile.readlines()[11:]
    
    mrk = list()
    
    for line in lines:
        newMarker = {"type" : line.split(",")[1], 
                     "idx" : int(line.split(",")[2])}
        
        mrk.append(newMarker)
    
    mrkFile.close()
    
    return mrk
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
