"""
Created on Oct 17, 2022.

@author: voodoocode
"""

import numpy as np
import pickle
import warnings
import sklearn.covariance
import sklearn.decomposition

import finnpy.src_rec.utils  # @UnresolvedImport

def _empirically_estimate_cov(cov_data, epoch_splits, signal_type, valid_ch_indices, ch_types = None):
    """
    Calculate the sensor noise covariance.
    
    Parameters
    ----------
    cov_data : numpy.ndarray, shape(samples, ch_cnt)
               An (empty room) file to use for sensor noise covariance calculations.
               Important: Evaluate a number of different files to identify a good example.
    epoch_splits : list
                   Epoching-indices for covariance calculation.
    signal_type : string
                  Can be either "EEG" or "MEG", determines how the data is loaded.
    valid_ch_indices : numpy.ndarray, shape(ch_cnt,)
                       Binary list identifying channels as valid/invalid.
    ch_types : list, len(ch_cnt)
              Channels may be either "mag", "grad", or "eeg".
               
    Returns
    -------
    cov : numpy.ndarray, shape(meg_ch_cnt, meg_ch_cnt)
          Covariance
          
    Raises
    ------
    AssertionError
        Signal type is invalid, has to be either 'EEG' or 'MEG'.
    """
    
    def _calc_cov_meg(cov_data, epoch_splits, mag_ch_indices, grad_ch_indices, valid_ch_indices, reject_thresholds = None):
        """
        Calculate the covariance for meg data.
        
        Parameters
        ----------
        cov_data : numpy.ndarray, shape(samples, ch_cnt)
                   An (empty room) file to use for sensor noise covariance calculations.
                   Important: Evaluate a number of different files to identify a good example.
        mag_ch_indices : list
                         List of meg channel ids.
        grad_ch_indices : list
                          List of grad channel ids.
        valid_ch_indices : list
                           List of valid channels.
        reject_thresholds : dict 
                            Threshold for sample rejection for 'mag' and 'grad'.
        
        Returns
        -------
        tuple of (np.ndarray, np.ndarray, int)
            - mu : np.ndarray
                   Average value
            - cov : np.ndarray
                    COvariance
            - samp_cnt : int
                         Number of samples.
        """
        if (reject_thresholds is None):
            reject_thresholds = {"mag": 4e-12, "grad": 4e-10}
        
        cov_data = cov_data.swapaxes(0, 1)
        mu = 0; samp_cnt = 0; cov = 0
        for epoch_idx in range(len(epoch_splits) - 2):
            loc_cov_data = cov_data[:, epoch_splits[epoch_idx]:epoch_splits[epoch_idx + 1]]
            
            mag_delta = np.max(loc_cov_data[mag_ch_indices, :], axis = 1) - np.min(loc_cov_data[mag_ch_indices, :], axis = 1)
            grad_delta = np.max(loc_cov_data[grad_ch_indices, :], axis = 1) - np.min(loc_cov_data[grad_ch_indices, :], axis = 1)
            # If epoch is bad, skip
            
            if ((mag_delta > reject_thresholds["mag"]).any() or (grad_delta > reject_thresholds["grad"]).any()):
                continue
            loc_cov_data = loc_cov_data[np.asarray(valid_ch_indices, dtype = bool), :]
            
            mu += np.sum(loc_cov_data, axis = 1)
            cov += np.dot(loc_cov_data, loc_cov_data.T)
            samp_cnt += loc_cov_data.shape[1]
        return (mu, cov, samp_cnt)
    
    def _calc_cov_eeg(cov_data, epoch_splits, reject_threshold_factor = 2):
        """
        Calculate the covariance for eeg data.
        
        Parameters
        ----------
        cov_data : numpy.ndarray, shape(samples, ch_cnt)
                   An (empty room) file to use for sensor noise covariance calculations.
                   Important: Evaluate a number of different files to identify a good example.
        epoch_splits : list
                       Epoching-indices for covariance calculation.
        reject_threshold_factor : factor 
                                  Number of standard deviations after which a sample is rejected.  
        
        Returns
        -------
        tuple of (np.ndarray, np.ndarray, int)
            - mu : np.ndarray
                   Average value
            - cov : np.ndarray
                    COvariance
            - samp_cnt : int
                         Number of samples.
        """
        cov_data = cov_data.swapaxes(0, 1)
        mu = 0; samp_cnt = 0; cov = 0
        deltas = np.empty((cov_data.shape[0], cov_data.shape[1]))
        for epoch_idx in range(len(epoch_splits) - 2):
            loc_cov_data = cov_data[:, epoch_splits[epoch_idx]:epoch_splits[epoch_idx + 1]]
            deltas[segment_idx] = np.max(loc_cov_data, axis = 1) - np.min(loc_cov_data, axis = 1)
        # Per segment delta percentile (70%) of per channel signal variability * 2 (default; expands to 5)
        # Assumes that values up to 140 % of the average are fair game as artifacts are usually several orders of 
        # magnitude stronger
        reject_threshold = np.percentile(deltas, 70, axis = 0) * reject_threshold_factor
        
        bads = 0
        for segment_idx in range(cov_data.shape[0]):
            delta = deltas[segment_idx, :]
            # If epoch is bad, skip
            if ((delta > reject_threshold).any()):
                bads += 1
                continue
            loc_cov_data = loc_cov_data[np.asarray(valid_ch_indices, dtype = bool), :]
            
            mu += np.sum(loc_cov_data, axis = 1)
            cov += np.dot(loc_cov_data, loc_cov_data.T)
            samp_cnt += loc_cov_data.shape[1]
        return (mu, cov, samp_cnt)
    
    if (signal_type == "MEG"):
        mag_ch_indices = np.argwhere(np.asarray(ch_types) == "mag").squeeze(1)
        grad_ch_indices = np.argwhere(np.asarray(ch_types) == "grad").squeeze(1)
        
        (mu, cov, samp_cnt) = _calc_cov_meg(cov_data, epoch_splits, mag_ch_indices, grad_ch_indices, valid_ch_indices)
        if (samp_cnt < 5):
            warnings.warn("Bad epoch threshold increased (x10) for covariance calculation")
            (mu, cov, samp_cnt) = _calc_cov_meg(cov_data, epoch_splits, mag_ch_indices, grad_ch_indices, valid_ch_indices, reject_thresholds = {"mag": 4e-11, "grad": 4e-9})
    elif (signal_type == "EEG"):
        eeg_ch_indices = np.argwhere(np.asarray(ch_types) == "eeg").squeeze(1)
        cov_data = cov_data[:, :, eeg_ch_indices]
        
        (mu, cov, samp_cnt) = _calc_cov_eeg(cov_data, epoch_splits, reject_threshold_factor = 2)
        if (samp_cnt < 5):
            warnings.warn("Bad epoch threshold increased (x5) for covariance calculation")
            (mu, cov, samp_cnt) = _calc_cov_eeg(cov_data, epoch_splits, reject_threshold_factor = 5)
    else:
        raise AssertionError('Signal type %s not supported, must be "EEG" or "MEG"' % (signal_type,))
        
    cov -= np.expand_dims(mu, axis = 1) * (np.expand_dims(mu, axis = 0) / samp_cnt)
    cov /= (samp_cnt - 1)
    
    return cov

def _calc_sensor_noise_cov(sensor_data, fs, signal_type, 
                           valid_channels, ch_types = None,  
                           method = None, epoch_sz_s = .2, method_params = None):
    """
    Calculate the sensor noise covariance.
    
    Parameters
    ----------
    sensor_data : numpy.ndarray, shape(samples, ch_cnt)
                  An (empty room) file to use for sensor noise covariance calculations.
                  Important: Evaluate a number of different files to identify a good example.
    fs : float
         Sampling frequency
    signal_type : string
                  Can be either "EEG" or "MEG", determines how the data is loaded.
    valid_channels : numpy.ndarray, shape(ch_cnt,)
                     Binary list identifying channels as valid/invalid.
    ch_types : list, len(ch_cnt)
              Channels may be either "mag", "grad", or "eeg".
    method : string
             Method to be employed, either "empirically",
             "shrinkage", or "factor_analysis" (default: shrinkage).
    epoch_sz_s : 0.2
                 Size of individual epochs, scaled in s.
    method_params : variable
                    Passed on to sklearn's ShrunkCovariance & FactorAnalysis.
               
    Returns
    -------
    cov : numpy.ndarray, shape(meg_ch_cnt, meg_ch_cnt)
          Covariance
          
    Raises
    ------
    AssertionError
        Inversion method is invalid, has to be 'empirically', 'shrinkage' or 'factor_analysis'.
    """
    if (method is None or method == "empirically"):
        epoch_splits = np.arange(0, sensor_data.shape[0], int(fs * epoch_sz_s))
        cov = _empirically_estimate_cov(sensor_data, epoch_splits, signal_type, valid_channels, ch_types)
    elif (method == "shrinkage"):
        cov = sklearn.covariance.ShrunkCovariance(**method_params).fit(sensor_data.T[:, valid_channels]).covariance_
    elif (method == "factor_analysis"):
        cov = sklearn.decomposition.FactorAnalysis(**method_params).fit(sensor_data.T[:, valid_channels]).get_covariance()
    else:
        raise AssertionError("Invalid method %s, has to be 'empirically', 'shrinkage' or 'factor_analysis'." % (method,))
    
    return cov

class Sen_cov():
    """
    Container class, populed with the following items.
    
    Parameters
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

def run(sensor_data, fs, signal_type,
        valid_channels, ch_names, ch_types, 
        method = None, float_sz = 64, epoch_sz_s = 0.2, method_params = None, 
        fast_eigendecomp_path = "../FinnPy_speedups/Release/FinnPy_speedups.so"):
    """
    Compute the sensor noise covariance from given data. Of note, must not contain data of interest.
    
    Parameters
    ----------
    sensor_data : numpy.ndarray, shape(samples, ch_cnt)
                  An (empty room) file to use for sensor noise covariance calculations.
                  Important: Evaluate a number of different files to identify a good example.
    fs : float
         Sampling frequency
    signal_type : string
                  Can be either "EEG" or "MEG", determines how the data is loaded.
    valid_channels : numpy.ndarray, shape(ch_cnt,)
                     Binary list identifying channels as valid/invalid.
    ch_names : list, len(ch_cnt)
               Name of the EEG/MEG channels.
    ch_types : list, len(ch_cnt)
               Channels may be either "mag", "grad", or "eeg".
    grad_channels : list, int
                    Indices of gradiometer channels.
    method : string
             Method to be employed, either "empirically",
             "shrinkage", or "factor_analysis" (default: shrinkage).
    float_sz : int
               Floating point precision used, can be either 32, 64, 128, or 256 (recommended).
    epoch_sz_s : 0.2
                 Size of individual epochs, scaled in s.
    method_params : dict()
                    Method specific parameters.
                    Only applies to sklearn.covariance.ShrunkCovariance and sklearn.decomposition.FactorAnalysis, 
                    defaults to "shrinkage" : 0.2 - epoch size in s.
    fast_eigendecomp_path: string
                           Path to the finnpy speedups library
               
    Returns
    -------
    sen_cov : finnpy.src_rec.sen_cov.Sen_cov
              Container class.
    
    Raises
    ------
    AssertionError
        Number of channels is higher than the number of datapoints, as this is very unlikely, the input is rejected.
    """
    if (sensor_data.shape[0] < sensor_data.shape[1]):
        raise AssertionError("Error: More channels than datapoints. It is likely, the matrix is of shape (ch_cnt, samples) rather than (samples, ch_cnt).")
    
    # Calculates the covariance
    sensor_cov = _calc_sensor_noise_cov(sensor_data, fs, signal_type,
                                        valid_channels, ch_types,
                                        method, epoch_sz_s, method_params)
    
    # Extracts the eigenvectors/-values
    (evals, evecs) = finnpy.src_rec.utils.calc_eigendecomposition(sensor_cov, float_sz, fast_eigendecomp_path)
    
    return Sen_cov(evals, evecs.T, ch_names)
    
def load(cov_path):
    """
    Determine eigenvectors/values from the sensor noise covariance.
    
    Parameters
    ----------
    cov_path : string
               Path to a the covariance file. If none exists, the covariance will be saved in this location.
                              
    Returns
    -------
    sen_cov : finnpy.src_rec.sen_cov.Sen_cov
              Container class.
    """
    evals = np.load(cov_path + "eval.npy")
    evecs = np.load(cov_path + "evec.npy")
    ch_names = pickle.load(open(cov_path + "ch_names.pkl", "rb"))
    
    return Sen_cov(evals.squeeze(1), evecs.T, ch_names)
