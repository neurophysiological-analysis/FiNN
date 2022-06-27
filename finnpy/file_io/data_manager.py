'''
Created on Dec 30, 2020

Implements a convenient data handler for large, unbalanced data sets. The data manager is to be provided with either a list, a dictionary, or a numpy array. These are traverse in order to save unbalanced data in smaller individual data junks. This avoids excessive memory consumption as observed when using pickle to storge large data containers (>2GB). 

@author: voodoocode
'''

import numpy as np
import pickle
import os

import finnpy.file_io.data_manager_legacy as legacy

def save(data, path, max_depth = 2, legacy_mode = False, legacy_params = None):
    """
    Saves data using the data manager. Allows for the convenient storage of large unbalanced data structures without memory spikes.
   
    :param data: The data to be stored.
    :param path: Location for data storage.
    :param max_depth: The depth to which folders are created prior to storing data via pickle.
    :param legacy_mode: *depricated* Will be removed in a future version.
    :param legacy_params: *depricated* Will be removed in a future version.
    """
    
    if (legacy_mode == True):
        return legacy.save(data, path, legacy_params["var_name"], max_depth, legacy_params["ending"])
    
    current_depth = 0
    
    if (path[-1] == "/"):
        path = path[:-1]
    
    structure = _save(data, path, current_depth, max_depth, None)
    file = open(path + "/meta.nfo", "wb")
    pickle.dump(structure, file)
    file.close()
    
def _save(data, path, current_depth, max_depth, structure = None):
    """
    
    Internally recursively called until either an incompatible data type is discovered or the max_depth is reached.
    
    :param data: The data to be stored.
    :param path: Location for data storage.
    :param current_depth: The current depth of the tree.
    :param max_depth: The depth to which folders are created prior to storing data via pickle.
    :param structure: Data structure used for data storage.
    
    returns the data structure used for data storage.
    
    """
        
    if (type(data) == np.ndarray and current_depth < max_depth):
        structure = list(); structure.append("np.ndarray")
            
        for (sub_data_idx, sub_data) in enumerate(data):
            structure.append([sub_data_idx, list()])
            structure[-1][-1] = _save(sub_data, path + "/" + str(sub_data_idx), current_depth + 1, max_depth, structure[-1][-1])
            
    elif(type(data) == dict and current_depth < max_depth):
        structure = list(); structure.append("dict")
            
        for key in data:
            structure.append([key, type(key), list()])
            structure[-1][-1] = _save(data[key], path + "/" + str(key), current_depth + 1, max_depth, structure[-1][-1])
            
    elif(type(data) == list and current_depth < max_depth):
        structure = list(); structure.append("list")
            
        for (sub_data_idx, sub_data) in enumerate(data):
            structure.append([sub_data_idx, list()])
            structure[-1][-1] = _save(sub_data, path + "/" + str(sub_data_idx), current_depth + 1, max_depth, structure[-1][-1])
            
    else:
        os.makedirs(path, exist_ok = True)
        if (type(data) == np.ndarray):
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
    Loads data stored via the data_manager.
    
    :param path: Location from which the data is to be read.
    :param legacy_mode: *deprecated* Will be removed in a future version.
    
    :return: The read data.
    """
    
    if (legacy_mode == True):
        return legacy.load(path)
    
    file = open(path + "/meta.nfo", "rb")
    structure = pickle.load(file)
    file.close()
    return _load(structure, path)

def _load(structure, path):
    """
    
    Internally recursively called to load data
    
    :param structure: Data format of the data. 
    :param path: Current path.
    
    :return: Read data.
    
    """
    
    data = None
    
    if (type(structure) == list):
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
        elif(structure == "data_npy"):
            data = np.load(path + "/data.npy")
        else:
            raise AssertionError("File corrupted")
    else:
        raise AssertionError("File corrupted")
    
    return data
    
    
    
    
    
