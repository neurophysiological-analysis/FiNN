'''
Created on Oct 17, 2022

@author: voodoocode
'''

import numpy as np
import mpmath

import mne
import warnings

import os

def get_bio_channel_type_idx(raw_file, mask = None):
    valid_ch_indices = np.zeros((len(raw_file.info["chs"]), ), dtype = bool)
    meg_ch_indices = list()
    grad_ch_indices = list()
    
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
    
    return (valid_ch_indices, meg_ch_indices, grad_ch_indices)

def get_meg_channel_type_idx(raw_file, mask = None):
    valid_ch_indices = np.zeros((len(raw_file.info["chs"]), ), dtype = bool)
    meg_ch_indices = list()
    grad_ch_indices = list()
    
    channel_types = np.asarray(raw_file.info["chs"])
    if (mask is not None):
        valid_ch_indices = valid_ch_indices[mask]
        channel_types = channel_types[mask]
    
    for ch_idx in range(len(channel_types)):
        loc_kind = channel_types[ch_idx]["kind"]
        if (loc_kind == mne.io.constants.FIFF.FIFFV_MEG_CH):
            valid_ch_indices[ch_idx] = True
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
    
    return (valid_ch_indices, meg_ch_indices, grad_ch_indices)

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

import mne.cov

def factor_analysis_cov_estimation(cov_data):
    tmp = sklearn.covariance.ShrunkCovariance()

def get_sensor_noise_cov(raw_file, method = None, epoch_sz_s = .2):
    (valid_ch_indices, meg_ch_indices, grad_ch_indices) = get_bio_channel_type_idx(raw_file)
    
    cov_data = raw_file.get_data()
    
    epoch_splits = np.arange(0, cov_data.shape[1], int(raw_file.info["sfreq"] * epoch_sz_s))
    cov_data = np.asarray(np.split(cov_data, epoch_splits, axis = 1)[1:])
    
    if (method is None):
        if (cov_data.shape[0] > 40):
            cov = empirically_estimate_cov(cov_data, meg_ch_indices, grad_ch_indices, valid_ch_indices)
        else:
            cov = None
    
    return (cov, valid_ch_indices)

def calc_sensor_covariance(file_path, demo = False):
    raw_file = mne.io.read_raw_fif(file_path, preload = True, verbose = "ERROR")
    (sensor_noise_cov, valid_ch_indices) = get_sensor_noise_cov(raw_file)
    (valid_meg_ch_indices, _, _) = get_meg_channel_type_idx(raw_file, valid_ch_indices)
    meg_sensor_noise_cov = sensor_noise_cov[valid_meg_ch_indices, :]; meg_sensor_noise_cov = meg_sensor_noise_cov[:, valid_meg_ch_indices]
    
    #===========================================================================
    # import mne.cov
    # import mne.rank
    # mne.cov.compute_raw_covariance(raw_file, method = "ledoit_wolf")# <--- returns bad rank; MNE incorrectly estimates tolerance parameters => rank is incorrectly estiamted
    # mne.rank.compute_rank(mne.Covariance(sensor_noise_cov, (np.asarray(raw_file.info["ch_names"])[np.concatenate((np.arange(1, 33, dtype = int), np.arange(44, 350, dtype = int)))]).tolist(), bads = [], projs = [], nfree = 72), info = raw_file.info)
    # # Works perfectly flawless
    #===========================================================================
    
    if (os.path.exists("/home/voodoocode/Downloads/eval3.npy") == False or demo == False):
        mat = mpmath.matrix(meg_sensor_noise_cov)
        mat.ctx.dps = 40
        (eigen_val, eigen_vec) = mpmath.eigsy(mat)
    
        eigen_val = np.asarray(eigen_val.tolist(), dtype = float)
        eigen_vec = np.asarray(eigen_vec.tolist(), dtype = float)
    
        np.save("/home/voodoocode/Downloads/eval3.npy", eigen_val)
        np.save("/home/voodoocode/Downloads/evec3.npy", eigen_vec)
    else:
        warnings.warn("Remove this part")
        
        eigen_val = np.load("/home/voodoocode/Downloads/eval3.npy")
        eigen_vec = np.load("/home/voodoocode/Downloads/evec3.npy")
        
    return (eigen_val.squeeze(1), eigen_vec.T)
    
    
    
    
    