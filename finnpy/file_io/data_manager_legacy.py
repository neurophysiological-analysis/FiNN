"""**Depricated**. Will be removed in a future version."""


import os.path
import numpy as np
import pickle
import warnings

def save(data, path = "", var_name = "data", max_depth = 2, ending = None):
    """
    **Depricated**. Will be removed in a future version. Saves data using the legacy data manager. Allows for the convenient storage of large unbalanced data structures without memory spikes.
   
    This function saves data.
    
    Parameters
    ----------
    data: Any
          data to be stored.
    path : str
           Location for data storage.
    var_name : str
               Name of the lwo level data containers.
    max_depth : int
                The depth to which folders are created prior to storing data via pickle.
    ending : str
             To be stored data type.
    """
    warnings.warn("The legacy version of the data manager will be removed in a future version")  # pylint: disable=syntax-error
    
    if (ending is not None and ending not in [".npy", ".pkl", ".hdr"]):
        warnings.warn("Unknown file-ending, changing to .pkl or .npy depending on data-type")
        ending = None
    
    if (path[-1] != "/"):
        path += "/"
        
    locPath = path + var_name
    if (type(data) is np.ndarray):
        if (ending is None or ending == ".pkl" or ending == ".hdr"):
            ending = ".npy"
        np.save(locPath + ending, data)
    elif (type(data) is dict):
        if (ending is None or ending == ".npy"):
            ending = ".hdr"
        file = open(locPath + ending, "wb")
        pickle.dump(data, file)
        file.close()
    elif (type(data) is list):
        _save(data, locPath, 1, max_depth, ending)
    
def _save(data, path, curr_depth, max_depth = 2, ending = None):
    """
    **Depricated** Will be removed in a future version. Loads data stored via the data_manager.
    
    This function is recursively called to save data.
    
    Parameters
    ----------
    data : Any
           The data to be stored.
    path : str
           Location for data storage.
    curr_depth : int
                 Current depth of the folder tree.
    max_depth : int
                 The depth to which folders are created prior to storing data via pickle.
    ending : str
             To be stored data type.
    """
    if (path[-1] != "/"):
        path += "/"
    
    if (curr_depth == max_depth):
        if (os.path.exists(path) is False):
            os.makedirs(path, exist_ok = True)
        file = open(path + "data.pkl", "wb")
        pickle.dump(data, file)
        file.close()
    elif (type(data) is np.ndarray):
        if (os.path.exists(path) is False):
            os.makedirs(path, exist_ok = True)
        if (ending is None or ending == ".pkl" or ending == ".hdr"):
            ending = ".npy"
        np.save(path + "data" + ending, data)
    elif (type(data) is dict):
        if (os.path.exists(path) is False):
            os.makedirs(path, exist_ok = True)
        if (ending is None or ending == ".npy"):
            ending = ".hdr"
        file = open(path + "data.hdr", "wb")
        pickle.dump(data, file)
        file.close()
    elif (type(data) is list):
        if (os.path.exists(path) is False):
            os.makedirs(path, exist_ok = True)
        for (sub_data_idx, sub_data) in enumerate(data):
            _save(sub_data, path + str(sub_data_idx), curr_depth + 1, max_depth)

def load(path, verbose = False):
    """
    **Depricated**. Will be removed in a future version. Loads data stored via the data_manager.
    
    This function loads data.
    
    Parameters
    ----------
    path : str
           Location from which the data is to be read.
    verbose : boolean
              Controls the amount of debug output.
    
    Returns
    -------
    Any
        Loaded data.
        
    Raises
    ------
    AssertionError
        If file ending is neither *.npy, *.hdr, nor *.pkl.
    """
    warnings.warn("The legacy version of the data manager will be removed in a future version")
    
    if (os.path.isfile(path) is True):
        if (path[-4:] == ".npy"):
            return np.load(path, allow_pickle = True)
        elif (path[-4:] == ".hdr" or path[-4:] == ".pkl"):
            file = open(path, "rb")
            data = pickle.load(file)
            file.close()
            return data
        else:
            raise AssertionError("Error: File ending must be either *.npy, *.hdr, or *.pkl")
    elif (os.path.isfile(path) is False):
        return _load(path, verbose)

def _load(path, is_top_level = False):
    """
    **Depricated**. Will be removed in a future version. Loads data stored via the data_manager.
    
    This function recursively loads data.
    
    Parameters
    ----------
    path : string
           Location from which the data is to be read.
    is_top_level : boolean
                   Flag indicating whether the current level is the top level
                   
    Returns
    -------
    Any
        Loaded data.
    """
    if (path[-1] != "/"):
        path += "/"
    if (os.path.isdir(path + os.listdir(path)[0]) is False):
        if (os.listdir(path)[0][-4:] == ".pkl"):
            file = open(path + os.listdir(path)[0], "rb")
            data = pickle.load(file)
            file.close()
            return data
        elif (os.listdir(path)[0][-4:] == ".npy"):
            return np.load(path + os.listdir(path)[0], allow_pickle = True)
        elif (os.listdir(path)[0][-4:] == ".hdr"):
            file = open(path + os.listdir(path)[0], "rb")
            data = pickle.load(file)
            file.close()
            return data
    else:
        sublist = list()
        for (folder_idx, folder) in enumerate(list(map(str, np.sort(list(map(int, os.listdir(path))))))):
            if (is_top_level is True):
                print("Progress: %f" % (folder_idx / len(os.listdir(path))))
            sub_path = path + folder
            sublist.append(_load(sub_path))
        return sublist
