'''
Created on Oct 17, 2022

@author: voodoocode
'''

import numpy as np
import mpmath
import pickle

import mne
import warnings

import os

import sklearn.covariance
import sklearn.decomposition

def _get_bio_channel_type_idx(raw_file, mask = None):
    """
    Identifies bio channel types.
    
    Parameters
    ----------
    raw_file : mne.io.read_raw_fif
               Scanned MRI file.
               
    Returns
    -------
    valid_ch_indices : numpy.ndarray, shape(ch_cnt,)
                       Binary list identifying channels as valid/invalid.
    meg_ch_indices : list, int
                     Indices of magnetometer channels.
    grad_ch_indices : list, int
                      Indices of gradiometer channels.
    ch_names : list, string
               channel names.
    """
    valid_ch_indices = np.zeros((len(raw_file.info["chs"]), ), dtype = bool)
    meg_ch_indices = list()
    grad_ch_indices = list()
    ch_names = list()
    
    channel_types = np.asarray(raw_file.info["chs"])
    if (mask is not None):
        valid_ch_indices = valid_ch_indices[mask]
        channel_types = channel_types[mask]
    
    for ch_idx in range(len(channel_types)):
        loc_kind = channel_types[ch_idx]["kind"]
        if (loc_kind in [mne.io.constants.FIFF.FIFFV_MEG_CH, mne.io.constants.FIFF.FIFFV_EEG_CH, 
                         mne.io.constants.FIFF.FIFFV_SEEG_CH, mne.io.constants.FIFF.FIFFV_ECOG_CH, 
                         mne.io.constants.FIFF.FIFFV_FNIRS_CH, mne.io.constants.FIFF.FIFFV_DBS_CH]):
            valid_ch_indices[ch_idx] = True
            ch_names.append(raw_file.ch_names[ch_idx])
        if (raw_file.info["chs"][ch_idx]["coil_type"] in [mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T1,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T2,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T3,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T4,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_W]):
            meg_ch_indices.append(ch_idx)
        if (raw_file.info["chs"][ch_idx]["coil_type"] in [mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T1, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T2, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T3, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T4, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_W]):
            grad_ch_indices.append(ch_idx)
    
    return (valid_ch_indices, meg_ch_indices, grad_ch_indices, ch_names)

def _empirically_estimate_cov(cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices):
    """
    Calculates the sensor noise covariance
    
    Parameters
    ----------
    cov_data : numpy.ndarray, shape(samples, channels)
               An (empty room) file to use for sensor noise covariance calculations.
               Important: Evaluate a number of different files to identify a good example.
    meg_ch_indices : list, int
                     Indices of magnetometer channels.
    grad_ch_indices : list, int
                      Indices of gradiometer channels.
    valid_ch_indices : numpy.ndarray, shape(ch_cnt,)
                       Binary list identifying channels as valid/invalid.
               
    Returns
    -------
    cov : numpy.ndarray, shape(meg_ch_cnt, meg_ch_cnt)
          Covariance
    """
    reject_thresholds = {"meg" : 4e-12, "grad" : 4e-10}
    
    def internal(cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices):
        mu = 0; samp_cnt = 0; cov = 0
        for segment_idx in range(cov_data.shape[0]):
            loc_cov_data = cov_data[segment_idx, : , :]
            
            meg_delta = np.max(loc_cov_data[meg_ch_indices, :], axis = 1) - np.min(loc_cov_data[meg_ch_indices, :], axis = 1)
            grad_delta = np.max(loc_cov_data[grad_ch_indices, :], axis = 1) - np.min(loc_cov_data[grad_ch_indices, :], axis = 1)
            #If epoch is bad, skip
            if ((meg_delta > reject_thresholds["meg"]).any() or (grad_delta > reject_thresholds["grad"]).any()):
                continue
            loc_cov_data = loc_cov_data[np.asarray(valid_ch_indices, dtype = bool), :]
            
            mu += np.sum(loc_cov_data, axis = 1)
            cov += np.dot(loc_cov_data, loc_cov_data.T)
            samp_cnt += loc_cov_data.shape[1]
        return (mu, cov, samp_cnt)
    
    (mu, cov, samp_cnt) = internal(cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices)
    if (samp_cnt < 5):
        reject_thresholds = {"meg" : 4e-11, "grad" : 4e-9}
        warnings.warn("Bad epoch threshold increased (x10) for covariance calculation")
        (mu, cov, samp_cnt) = internal(cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices)
        
    cov -= np.expand_dims(mu, axis = 1) * (np.expand_dims(mu, axis = 0) / samp_cnt)
    cov /= (samp_cnt - 1)
    
    return cov

def _calc_sensor_noise_cov(raw_file, method = None, method_params = None, epoch_sz_s = .2):
    """
    Calculates the sensor noise covariance
    
    Parameters
    ----------
    file_path : string
                The (empty room) file to use for sensor noise covariance calculations. Important: Evaluate a number of different files to identify a good example.
    cov_path : string
               Path to a the covariance file. If none exists, the covariance will be saved in this location.
    method : string
             Method to be employed, either "empirically",
             "shrinkage", or "factor_analysis" (default: shrinkage).
    epoch_sz_s : 0.2
                 Size of individual epochs, scaled in s.
               
    Returns
    -------
    cov : numpy.ndarray, shape(meg_ch_cnt, meg_ch_cnt)
          Covariance
           
    ch_names : list, string
               Channel names.
    """
    (valid_ch_indices, meg_ch_indices, grad_ch_indices, ch_names) = _get_bio_channel_type_idx(raw_file)
    
    cov_data = raw_file.get_data()
    
    epoch_splits = np.arange(0, cov_data.shape[1], int(raw_file.info["sfreq"] * epoch_sz_s))
    emp_cov_data = np.asarray(np.split(cov_data, epoch_splits, axis = 1)[1:])
    
    if (method is None or method == "empirically"):
        cov = _empirically_estimate_cov(emp_cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices)
    elif(method == "shrinkage"):
        cov = sklearn.covariance.ShrunkCovariance(**method_params).fit(cov_data.T[:, valid_ch_indices]).covariance_
    elif(method == "factor_analysis"):
        cov = sklearn.decomposition.FactorAnalysis(**method_params).fit(cov_data.T[:, valid_ch_indices]).get_covariance()
    
    return (cov, ch_names)

class Sen_cov():
    """
    Container class, populed with the following items:
    
    Attributes
    ----------
    
    evals: numpy.ndarray, shape(ch_cnt,)
           Eigenvalues.
    evecs: numpy.ndarray, shape(ch_cnt,ch_cnt)
           Eigenvectors/covariance matrix.
    ch_names : list, string
               Channel names.
    """
    
    def __init__(self, evals, evecs, ch_names):
        self.evals = evals
        self.evecs = evecs
        self.ch_names = ch_names

def run(file_path, cov_path, method = None, method_params = None, overwrite = False):
    
    """
    #Determines eigenvectors/values from the sensor noise covariance
    
    Parameters
    ----------
    file_path : string
                The (empty room) file to use for sensor noise covariance calculations. Important: Evaluate a number of different files to identify a good example.
    cov_path : string
               Path to a the covariance file. If none exists, the covariance will be saved in this location.
    method : string
             Method to be employed, either "empirically",
             "shrinkage", or "factor_analysis" (default: shrinkage).
    method_params : dict()
                    Method specific parameters.
                    Only applies to sklearn.covariance.ShrunkCovariance and sklearn.decomposition.FactorAnalysis, 
                    defaults to "shrinkage" : 0.2 - epoch size in s.
    overwrite : boolean
                Flag to overwrite covariance calculation.
               
    Returns
    -------
    sen_cov : finnpy.src_rec.sen_cov.Sen_cov
              Container class.
    """
    
    if ((os.path.exists(cov_path + "eval.npy") == False or 
         os.path.exists(cov_path + "evec.npy") == False) or overwrite):
    
        raw_file = mne.io.read_raw_fif(file_path, preload = True, verbose = "ERROR")
        (bio_sensor_noise_cov, ch_names) = _calc_sensor_noise_cov(raw_file, method, method_params)
    
        mat = mpmath.matrix(bio_sensor_noise_cov)
        mat.ctx.dps = 40
        (eigen_val, eigen_vec) = mpmath.eigsy(mat)
     
        eigen_val = np.asarray(eigen_val.tolist(), dtype = float)
        eigen_vec = np.asarray(eigen_vec.tolist(), dtype = float)
     
        if (os.path.exists(cov_path) == False):
            os.makedirs(cov_path, exist_ok = True)
     
        np.save(cov_path + "eval.npy", eigen_val)
        np.save(cov_path + "evec.npy", eigen_vec)
        pickle.dump(ch_names, open(cov_path + "ch_names.pkl", "wb"))
    else:
        eigen_val = np.load(cov_path + "eval.npy")
        eigen_vec = np.load(cov_path + "evec.npy")
        ch_names = pickle.load(open(cov_path + "ch_names.pkl", "rb"))
        
    return Sen_cov(eigen_val.squeeze(1), eigen_vec.T, ch_names)
    
def load(cov_path):
    
    """
    #Determines eigenvectors/values from the sensor noise covariance
    
    Parameters
    ----------
    cov_path : string
               Path to a the covariance file. If none exists, the covariance will be saved in this location.               
    Returns
    -------
    sen_cov : finnpy.src_rec.sen_cov.Sen_cov
              Container class.
    """
    
    eigen_val = np.load(cov_path + "eval.npy")
    eigen_vec = np.load(cov_path + "evec.npy")
    ch_names = pickle.load(open(cov_path + "ch_names.pkl", "rb"))
    
    return Sen_cov(eigen_val.squeeze(1), eigen_vec.T, ch_names)
    
    
    
