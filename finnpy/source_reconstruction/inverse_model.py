'''
Created on Oct 12, 2022

@author: voodoocode
'''

import numpy as np
import scipy.linalg

import finnpy.source_reconstruction.utils

import mne

def get_meg_channel_type_idx(rec_meta_info):
    """
    Returns the channel names of the MEG channels.
    
    :param rec_meta_info: MEG recording meta info.
    
    :return: Channel names.
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
     
     
    data_meg_ch_names = get_meg_channel_type_idx(rec_meta_info)
     
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

#===============================================================================
# def calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names,
#                        fwd_sol, rec_meta_info):
#      
#      
#     data_meg_ch_names = get_meg_channel_type_idx(rec_meta_info)
#      
#     #Extract relevant indices from the sensor covariance matrix's eigenvectors/values.
#     valid_meg_ch_idx = np.empty(((len(data_meg_ch_names))), dtype = int)
#     for (ch_idx, data_meg_ch_name) in enumerate(data_meg_ch_names):
#         valid_meg_ch_idx[ch_idx] = sensor_cov_names.index(data_meg_ch_name)
#     sensor_cov_eigen_val = sensor_cov_eigen_val[valid_meg_ch_idx]
#     sensor_cov_eigen_vec = sensor_cov_eigen_vec[valid_meg_ch_idx, :]; sensor_cov_eigen_vec = sensor_cov_eigen_vec[:, valid_meg_ch_idx]
#      
#     whitener = finnpy.source_reconstruction.utils.calc_whitener(sensor_cov_eigen_val, sensor_cov_eigen_vec)
#     gain_mat = np.dot(whitener, fwd_sol)
#      
#     #MNE: inverse.py - _prepare_forward: 1859
#     scale = np.sqrt(np.sum(sensor_cov_eigen_val > 0) / np.linalg.norm(gain_mat) ** 2)
#     gain_mat *= scale
#     source_cov = np.ones((fwd_sol.shape[1],)) * scale
#     source_cov = source_cov * source_cov
#      
#     (fields, sing, leads) = scipy.linalg.svd(gain_mat, full_matrices = False)
#  
#     # Regularize inverse to be used as noise-weight
#     #Employ Tikhonov regularized SVD decomposition
#     lambda2 = 1/9
#     reg_inv = np.zeros(sing.shape)
#     rank = np.sum(sensor_cov_eigen_val > 0)
#     loc_sing = sing[:rank]
#     reg_inv[:rank] = np.where(loc_sing > 0, loc_sing / (loc_sing ** 2 + lambda2), 0)
#     noise_weight = reg_inv
#      
#     if (0 in sensor_cov_eigen_val):
#         raise AssertionError("0 in sensor covariance eigenvalue list.")
#     lambda2 = 1/9
#     noise_weight2 = sing/(sing ** 2 + lambda2)
#      
#     #Compute noise normalization vector
#     noise_normalization_vec = np.zeros((leads.shape[1],))
#     for idx in range(leads.shape[1]):
#         pre_norm_factor = (np.sqrt(source_cov[idx]) * leads[:, idx] * noise_weight)
#         noise_normalization_vec[idx] = np.linalg.norm(pre_norm_factor)
#     noise_normalization_vec = 1 / np.abs(noise_normalization_vec)
#      
#     pre_inv_trans = np.dot(fields.T, whitener)
#     pre_inv_trans *= np.expand_dims(reg_inv, axis = 1)
#      
#     inv_trans = np.dot(leads.T, pre_inv_trans)
#     inv_trans *= np.expand_dims(np.sqrt(source_cov), axis = 1)
#  
#     return (inv_trans, noise_normalization_vec)
#===============================================================================

def apply_inverse_model(sensor_data, inv_model, noise_norm):
    """
    Applies an inverse model and noise normalization vector to sensor space data.
    
    :param sensor_data: Data in the sensor space.
    :param inv_model: Inverse model.
    :param noise_norm: Noise normalization vector.
    
    :return: Data is source space.
    """
    source_data = np.dot(inv_model, sensor_data)
    source_data *= np.expand_dims(noise_norm, axis = 1)
    
    return source_data





