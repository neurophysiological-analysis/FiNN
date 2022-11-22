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

def get_bio_channel_type_idx(raw_file, mask = None):
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

def empirically_estimate_cov(cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices):
    reject_thresholds = {"meg" : 4e-12, "grad" : 4e-10}
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
    
    cov -= np.expand_dims(mu, axis = 1) * (np.expand_dims(mu, axis = 0) / samp_cnt)
    cov /= (samp_cnt - 1)
    
    return cov

import sklearn.covariance
import sklearn.decomposition

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

def get_sensor_noise_cov(raw_file, method = None, method_params = None, epoch_sz_s = .2):
    (valid_ch_indices, meg_ch_indices, grad_ch_indices, ch_names) = get_bio_channel_type_idx(raw_file)
    
    cov_data = raw_file.get_data()
    
    epoch_splits = np.arange(0, cov_data.shape[1], int(raw_file.info["sfreq"] * epoch_sz_s))
    emp_cov_data = np.asarray(np.split(cov_data, epoch_splits, axis = 1)[1:])
    
    if (method is None or method == "empirically"):
        cov = empirically_estimate_cov(emp_cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices)
    elif(method == "shrinkage"):
        cov = sklearn.covariance.ShrunkCovariance(**method_params).fit(cov_data.T[:, valid_ch_indices]).covariance_
    elif(method == "factor_analysis"):
        cov = sklearn.decomposition.FactorAnalysis(**method_params).fit(cov_data.T[:, valid_ch_indices]).get_covariance()
    
    return (cov, ch_names)

def calc_sensor_covariance(file_path, cov_path, method = None, method_params = None, overwrite = False):
    
    """
    method = "shrinkage"
    method_params = {"shrinkage" : 0.2,}
    """
    
    raw_file = mne.io.read_raw_fif(file_path, preload = True, verbose = "ERROR")
    (bio_sensor_noise_cov, ch_names) = get_sensor_noise_cov(raw_file, method, method_params)
    
    if ((os.path.exists(cov_path + "eval.npy") == False or 
         os.path.exists(cov_path + "evec.npy") == False) or overwrite):
        mat = mpmath.matrix(bio_sensor_noise_cov)
        mat.ctx.dps = 40
        (eigen_val, eigen_vec) = mpmath.eigsy(mat)
     
        eigen_val = np.asarray(eigen_val.tolist(), dtype = float)
        eigen_vec = np.asarray(eigen_vec.tolist(), dtype = float)
     
        if (os.path.exists(cov_path) == False):
            os.mkdir(cov_path)
     
        np.save(cov_path + "eval.npy", eigen_val)
        np.save(cov_path + "evec.npy", eigen_vec)
        pickle.dump(ch_names, open(cov_path + "ch_names.pkl", "wb"))
    else:
        eigen_val = np.load(cov_path + "eval.npy")
        eigen_vec = np.load(cov_path + "evec.npy")
        ch_names = pickle.load(open(cov_path + "ch_names.pkl", "rb"))
        
    return (eigen_val.squeeze(1), eigen_vec.T, ch_names)
    
    
    
    
    