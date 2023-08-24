'''
Created on Oct 12, 2022

@author: voodoocode
'''

import numpy as np
import scipy.linalg

import finnpy.source_reconstruction.utils

import mne

def _get_meg_channel_type_idx(rec_meta_info):
    """
    Returns the channel names of the MEG channels.
    
    Parameters
    ----------
    rec_meta_info : mne.io.read_info
                    MEG scan meta information, obtainable through mne.io.read_info.
               
    Returns
    -------
    sensor_cov_names : list, len(ch_cnt,), string
                       Names of the channels.
    """
    ch_names = list()
    
    channel_types = np.asarray(rec_meta_info["chs"])
    for ch_idx in range(len(channel_types)):
        if (rec_meta_info["chs"][ch_idx]["coil_type"] in [mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T1,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T2,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T3,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T4,
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_W]):
            ch_names.append(rec_meta_info.ch_names[ch_idx])
        if (rec_meta_info["chs"][ch_idx]["coil_type"] in [mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T1, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T2, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T3, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T4, 
                                                     mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_W]):
            ch_names.append(rec_meta_info.ch_names[ch_idx])
    
    return ch_names

def calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names,
                       fwd_sol, rec_meta_info, method = "dSPM"):
    """
    Converts a forward model into an inverse model.
        
    Parameters
    ----------
    sensor_cov_eigen_val : numpy.ndarray, shape(ch_cnt,)
                           Eigenvalues of the sensor noise covariance (eigenvector matrix).
    sensor_cov_eigen_vec : numpy.ndarray, shape(ch_cnt, ch_cnt)
                           Sensor noise covariance.
    sensor_cov_names : list, len(ch_cnt,), string
                       Names of the channels in the sensor covariance matrix.
    fwd_sol : numpy.ndarray, shape(meg_ch_cnt, valid_vtx_cnt)
              Forward model.
    rec_meta_info : mne.io.read_info
                    MEG scan meta information, obtainable through mne.io.read_info.
    method : string
             Method used to calculate the noise normalization vector,
             defaults to "dSPM".
               
    Returns
    -------
    inv_trans : numpy.ndarray, shape(valid_vtx_cnt, meg_ch_cnt)
                Inverse model.
    noise_norm : numpy.ndarray, shape(valid_vtx_cnt,)
                 Noise normalization vector.
    """
     
     
    data_meg_ch_names = _get_meg_channel_type_idx(rec_meta_info)
     
    #Extract relevant indices from the sensor covariance matrix's eigenvectors/values.
    #And construct whitener
    valid_meg_ch_idx = np.empty(((len(data_meg_ch_names))), dtype = int)
    for (ch_idx, data_meg_ch_name) in enumerate(data_meg_ch_names):
        valid_meg_ch_idx[ch_idx] = sensor_cov_names.index(data_meg_ch_name)
    sensor_cov_eigen_val = sensor_cov_eigen_val[valid_meg_ch_idx]
    sensor_cov_eigen_vec = sensor_cov_eigen_vec[valid_meg_ch_idx, :]; sensor_cov_eigen_vec = sensor_cov_eigen_vec[:, valid_meg_ch_idx]
    whitener = finnpy.source_reconstruction.utils.calc_whitener(sensor_cov_eigen_val, sensor_cov_eigen_vec)
    
    #Whiten fwd solution for numerical stability and scale
    white_fwd = np.dot(whitener, fwd_sol)
    scale = np.sqrt(np.sum(sensor_cov_eigen_val > 0) / np.linalg.norm(white_fwd, ord="fro") ** 2)
    white_fwd *= scale
    
    #Calculate source_cov
    source_cov = (np.ones((fwd_sol.shape[1],)) * scale) ** 2
    
    #Employ Tikhonov regularized SVD decomposition, see for reference https://en.wikipedia.org/wiki/Ridge_regression#Relation_to_singular-value_decomposition_and_Wiener_filter,
    #To calculated (whitened) pseudo inverse.
    (U, sigmas, V) = scipy.linalg.svd(white_fwd, full_matrices = False)
    if (0 in sensor_cov_eigen_val):
        raise AssertionError("0 in sensor covariance eigenvalue list.")
    lambda2 = 1/9
    sigma = sigmas/(sigmas ** 2 + lambda2)
    inv_trans = np.dot(V.T, (np.dot(U.T, whitener) * np.expand_dims(sigma, axis = 1)))
    
    #Weigh whitened pseudo inverse according to sensor covariance.
    inv_trans *= np.expand_dims(np.sqrt(source_cov), axis = 1)
     
    #Compute noise normalization vector
    if (method == "dSPM"):
        noise_norm = np.zeros((V.shape[1],))
        for idx in range(V.shape[1]):
            pre_norm_factor = (np.sqrt(source_cov[idx]) * V[:, idx] * sigma)
            noise_norm[idx] = np.linalg.norm(pre_norm_factor)
        noise_norm = 1 / np.abs(noise_norm)
    else:
        raise AssertionError("Noise normalization method not yet implemented")
 
    return (inv_trans, noise_norm)

def apply_inverse_model(sensor_data, inv_model, noise_norm):
    """
    Applies an inverse model and noise normalization vector to sensor space data.
    
    Parameters
    ----------
    sensor_data : TYPE, shape(TEXT)
                     TEXT
    inv_model : TYPE, shape(TEXT)
                     TEXT
    noise_norm : TYPE, shape(TEXT)
                     TEXT
               
    Returns
    -------
    source_data : TYPE, shape(TEXT)
                     TEXT
    """
    source_data = np.dot(inv_model, sensor_data)
    source_data *= np.expand_dims(noise_norm, axis = 1)
    
    return source_data






