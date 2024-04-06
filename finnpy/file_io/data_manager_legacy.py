"""

**Depricated**. Will be removed in a future version. 

"""


import os.path
import numpy as np
import pickle
import warnings

def save(data, path = "", var_name = "data", max_depth = 2, ending = None):
	"""
	**Depricated**. Will be removed in a future version. Saves data using the legacy data manager. Allows for the convenient storage of large unbalanced data structures without memory spikes.
   
    This function saves data.
   
	:param data: The data to be stored.
	:param path: Location for data storage.
	:param var_name: Name of the lwo level data containers.
	:param max_depth: The depth to which folders are created prior to storing data via pickle.
	:param ending: To be stored data type.
	:param legacy_mode: *depricated* Will be removed in a future version.
	:param legacy_params: *depricated* Will be removed in a future version.
	"""
	warnings.warn("The legacy version of the data manager will be removed in a future version")
	
	if (ending is not None and ending not in [".npy", ".pkl", ".hdr"]):
		warnings.warn("Unknown file-ending, changing to .pkl or .npy depending on data-type")
		ending = None
	
	if (path[-1] != "/"):
		path += "/"
		
	locPath = path + var_name
	if (type(data) == np.ndarray):
		if (ending is None or ending == ".pkl" or ending == ".hdr"):
			ending = ".npy"
		np.save(locPath + ending, data)
	elif(type(data) == dict):
		if (ending is None or ending == ".npy"):
			ending = ".hdr"
		file = open(locPath + ending, "wb")
		pickle.dump(data, file)
		file.close()
	elif(type(data) == list):
		_save(data, locPath, 1, max_depth, ending)
	
def _save(data, path, curr_depth, max_depth = 2, ending = None):
	"""
	**Depricated** Will be removed in a future version. Loads data stored via the data_manager.
	
	This function is recursively called to save data.
	
	:param data: The data to be stored.
	:param path: Location for data storage.
	:param curr_depth: Current depth of the folder tree.
	:param max_depth: The depth to which folders are created prior to storing data via pickle.
	:param ending: To be stored data type.
	"""
	
	if (path[-1] != "/"):
		path += "/"
	
	if (curr_depth == max_depth):
		if (os.path.exists(path) == False):
			os.makedirs(path, exist_ok = True)
		file = open(path + "data.pkl", "wb")
		pickle.dump(data, file)
		file.close()
	elif(type(data) == np.ndarray):
		if (os.path.exists(path) == False):
			os.makedirs(path, exist_ok = True)
		if (ending is None or ending == ".pkl" or ending == ".hdr"):
			ending = ".npy"
		np.save(path + "data" + ending, data)
	elif(type(data) == dict):
		if (os.path.exists(path) == False):
			os.makedirs(path, exist_ok = True)
		if (ending is None or ending == ".npy"):
			ending = ".hdr"
		file = open(path + "data.hdr", "wb")
		pickle.dump(data, file)
		file.close()
	elif(type(data) == list):
		if (os.path.exists(path) == False):
			os.makedirs(path, exist_ok = True)
		for (sub_data_idx, sub_data) in enumerate(data):
			_save(sub_data, path + str(sub_data_idx), curr_depth + 1, max_depth)

def load(path, verbose = False):
	"""
	**Depricated**. Will be removed in a future version. Loads data stored via the data_manager.
	
	This function loads data.
	
	:param path: Location from which the data is to be read.
	:param legacy_mode: *deprecated* Will be removed in a future version.
	"""
	warnings.warn("The legacy version of the data manager will be removed in a future version")
	
	if (os.path.isfile(path) == True):
		if (path[-4:] == ".npy"):
			return np.load(path, allow_pickle = True)
		elif(path[-4:] == ".hdr" or path[-4:] == ".pkl"):
			file = open(path, "rb")
			data = pickle.load(file)
			file.close()
			return data
		else:
			raise AssertionError("Error: File ending must be either *.npy, *.hdr, or *.pkl")
	elif(os.path.isfile(path) == False):
		return _load(path, verbose)

def _load(path, is_top_level = False):
	"""
	**Depricated**. Will be removed in a future version. Loads data stored via the data_manager.
	
	This function recursively loads data.
	
	:param path: Location from which the data is to be read.
	:param is_top_level: Flag indicating whether the current level is the top level
	"""
	
	if (path[-1] != "/"):
		path += "/"
	if (os.path.isdir(path + os.listdir(path)[0]) == False):
		if(os.listdir(path)[0][-4:] == ".pkl"):
			file = open(path + os.listdir(path)[0], "rb")
			data = pickle.load(file)
			file.close()
			return data
		elif(os.listdir(path)[0][-4:] == ".npy"):
			return np.load(path + os.listdir(path)[0], allow_pickle = True)
		elif(os.listdir(path)[0][-4:] == ".hdr"):
			file = open(path + os.listdir(path)[0], "rb")
			data = pickle.load(file)
			file.close()
			return data
	else:
		sublist = list()
		for (folder_idx, folder) in enumerate(list(map(str, np.sort(list(map(int, os.listdir(path))))))):
			if (is_top_level == True):
				print("Progress: %f" % (folder_idx/len(os.listdir(path))))
			sub_path = path + folder
			sublist.append(_load(sub_path))
		return sublist
