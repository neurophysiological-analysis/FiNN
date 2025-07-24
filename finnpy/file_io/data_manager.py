"""
Created on Dec 30, 2020.

Implements a convenient data handler for large, unbalanced data sets. The data manager is to be provided with either a list, a dictionary, or a numpy array. These are traverse in order to save unbalanced data in smaller individual data junks. This avoids excessive memory consumption as observed when using pickle to storge large data containers (>2GB). 

@author: voodoocode
"""

import numpy as np
import pickle
import os

import finnpy.file_io.data_manager_legacy as legacy  # @UnresolvedImport

def save(data, path, max_depth = 2, max_length = 100, compress_np = False, legacy_mode = False, legacy_params = None):
    """
    Save data using the data manager. Allows for the convenient storage of large unbalanced data structures without memory spikes.
    
    Parameters
    ----------
    data : Any
           The data to be stored.
    path : str
           Location for data storage.
    max_depth : int
                The depth to which folders are created prior to storing data via pickle.
    max_length : int
                 Maximum length of a sublist within the to be saved object.
    compress_np : boolean
                  Specifies whether numpy arrays are to be compressed.
    legacy_mode : boolean
                  *Deprecated*, will be removed in a future version.
    legacy_params : Any
                    *Deprecated*, will be removed in a future version.
    """
    if (legacy_mode is True):
        legacy.save(data, path, legacy_params["var_name"], max_depth, legacy_params["ending"])
    
    current_depth = 0
    
    if (path[-1] == "/"):
        path = path[:-1]
    
    structure = _save(data, path, current_depth, max_depth, max_length, compress_np, None)
    if (os.path.exists(path) is False):
        os.mkdir(path)
    file = open(path + "/meta.nfo", "wb")
    pickle.dump(structure, file)
    file.close()
    
def _save(data, path, current_depth, max_depth, max_length, compress_np, structure = None):
    """
    Internally recursively called until either an incompatible data type is discovered or the max_depth is reached.
    
    Parameters
    ----------
    data : Any
           The data to be stored.
    path : str
           Location for data storage.
    current_depth : int
                    Current depth in the data archiving structure.
    max_depth : int
                The depth to which folders are created prior to storing data via pickle.
    max_length : int
                 Maximum length of a sublist within the to be saved object.
    compress_np : boolean
                  Specifies whether numpy arrays are to be compressed.
    structure : list
                Remembers the data structure at the respective level/depth.
    
    Returns
    -------
    structure : list
                The data structure used for data storage.
    """
    if (type(data) is np.ndarray and current_depth < max_depth and len(data) < max_length):
        structure = list(); structure.append("np.ndarray")
            
        for (sub_data_idx, sub_data) in enumerate(data):
            structure.append([sub_data_idx, list()])
            structure[-1][-1] = _save(sub_data, path + "/" + str(sub_data_idx), current_depth + 1, max_depth, max_length, compress_np, structure[-1][-1])
            
    elif (type(data) is dict and current_depth < max_depth and len(data) < max_length):
        structure = list(); structure.append("dict")
            
        for key in data:
            structure.append([key, type(key), list()])
            structure[-1][-1] = _save(data[key], path + "/" + str(key), current_depth + 1, max_depth, max_length, compress_np, structure[-1][-1])
            
    elif (type(data) is list and current_depth < max_depth and len(data) < max_length):
        structure = list(); structure.append("list")
            
        for (sub_data_idx, sub_data) in enumerate(data):
            structure.append([sub_data_idx, list()])
            structure[-1][-1] = _save(sub_data, path + "/" + str(sub_data_idx), current_depth + 1, max_depth, max_length, compress_np, structure[-1][-1])
            
    else:
        os.makedirs(path, exist_ok = True)
        if (type(data) is np.ndarray):
            if (compress_np):
                np.savez_compressed(path + "/data.npz", data)
                structure = "data_npz"
            else:
                np.save(path + "/data.npy", data)
                structure = "data_npy"
        else:
            file = open(path + "/data.pkl", "wb")
            pickle.dump(data, file)
            structure = "data_pkl"
            file.close()
        
    return structure

def load(path, legacy_mode = False):
    """
    Load data stored via the data_manager.
    
    Parameters
    ----------
    path : str
           Location from which the data is to be read.
    legacy_mode : boolean
                  *Deprecated*, will be removed in a future version.
    
    Returns
    -------
    Any
        The read data.
    """
    if (legacy_mode is True):
        return legacy.load(path)
    
    if (path[-1] == "/"):
        path = path[:-1]
    
    if (path[-4:] == ".npy"):
        return np.load(path)
    
    if (path[-4:] == ".npz"):
        return np.load(path + "/data.npz")["arr_0"]
    
    file = open(path + "/meta.nfo", "rb")
    structure = pickle.load(file)
    file.close()
    return _load(structure, path)

def _load(structure, path):
    """
    Internally recursively called to load data.
    
    Parameters
    ----------
    structure : list
                Information on the data format of the to-be-loaded data. 
    path : str
           Current path.
    
    Returns
    -------
    Any
        The read data.
        
    Raises
    ------
    AssertionError
        IF the file cannot be read and is neither *.pkl, *.npy, or *.npz.
    """
    data = None
    
    if (type(structure) is list):
        if (structure[0] == "np.ndarray"):
            data = list()
            for sub_structure in structure[1:]:
                loc_data = _load(sub_structure[-1], path + "/" + str(sub_structure[0]))
                data.append(loc_data)
            data = np.asarray(data)
                
        elif (structure[0] == "dict"):
            data = dict()
            for sub_structure in structure[1:]:
                key = sub_structure[1](sub_structure[0])
                loc_data = _load(sub_structure[-1], path + "/" + str(sub_structure[0]))
                data[key] = loc_data
                
        if (structure[0] == "list"):
            data = list()
            for sub_structure in structure[1:]:
                loc_data = _load(sub_structure[-1], path + "/" + str(sub_structure[0]))
                data.append(loc_data)
    elif (structure[:4] == "data"):
        if (structure == "data_pkl"):
            file = open(path + "/data.pkl", "rb")
            data = pickle.load(file)
            file.close()
        elif (structure == "data_npy"):
            data = np.load(path + "/data.npy", allow_pickle = True)
        elif (structure == "data_npz"):
            data = np.load(path + "/data.npz")["arr_0"]
        else:
            raise AssertionError("File corrupted")
    else:
        raise AssertionError("File corrupted")
    
    return data
    
    
    
    
    
