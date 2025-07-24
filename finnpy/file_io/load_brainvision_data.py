"""
Created on Jun 28, 2017.

Reads data from the brain-vision format.

:author: maximilianscherer
"""

import mne

def run(vhdr_path, bad_ch_list = None, preload = False, t_min = None, t_max = None, verbose = None):
    """
    Load data recorded via the brain vision software wrapping mne calls.
    
    Parameters
    ----------
    vhdr_path : str
                Path to the vhdr file.
    bad_ch_list : list
                  List of bad channels.
    preload : boolean
              If 'true' the data is loaded directly in this function. If 'false' only a container is created.
    t_min : float
            Start of the returned recording segment in seconds.
    t_max : float
            End of the returned recording segment in seconds.
    verbose : boolean
              Verbose the command line output.
    
    Returns
    -------
    mne.io.read_raw_brainvision
        A raw data object containing either the data or a dummy container and the address of the 'to be loaded' data.
    """
    if (bad_ch_list is not None):
        meta_info = mne.io.read_raw_brainvision(vhdr_path, misc = bad_ch_list, verbose = verbose)
    else:
        meta_info = mne.io.read_raw_brainvision(vhdr_path, verbose = verbose)
    if (t_min is not None or t_max is not None):
        if (t_min is None):
            t_min = 0
        meta_info = meta_info.crop(tmin = t_min, tmax = t_max)
    
    if (preload):
        meta_info.load_data()
    
    return meta_info

def read_marker(vmrk_path):
    """
    Extract the markers from a brain vision recording.
    
    Parameters
    ----------
    vmrk_path : str
                Path to the marker file.
    
    Returns
    -------
    list
        Returns a list of markers. Every marker is a dictionary with a type and an index.
    """
    mrkFile = open(vmrk_path, "r")  # pylint: disable=unspecified-encoding
    
    lines = mrkFile.readlines()[11:]
    
    mrk = list()
    
    for line in lines:
        newMarker = {"type": line.split(",")[1], 
                     "idx": int(line.split(",")[2])}
        
        mrk.append(newMarker)
    
    mrkFile.close()
    
    return mrk
    
    
    
