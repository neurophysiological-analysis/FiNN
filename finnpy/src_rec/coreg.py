"""
Created on Oct 12, 2022.

@author: voodoocode
"""

import functools
import numpy as np
import scipy.optimize
import scipy.spatial
import nibabel.freesurfer
import os
import mne
import warnings
import copy
import pyvista

import finnpy.src_rec.extract_anatomy  # @UnresolvedImport
import finnpy.src_rec.utils  # @UnresolvedImport

class Coreg():  # noqa: DOC605
    """
    Container class containing the following parameters.
    
    For EEG, a EEG cap is projected onto the skull, for MEG registration
    points are matched with MRI surface.
    
    Attributes
    ----------
    rotors : np.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3). MEG only.
    mri_to_meeg_trs : np.ndarray, shape(4, 4)
                      Full affine transformation matrix (MRI -> MEEG). MEG only.
    mri_to_meeg_tr : np.ndarray, shape(4, 4)
                     Rigid affine transformation matrix (MRI -> MEEG). MEG only.
    mri_to_meeg_rs : np.ndarray, shape(4, 4)
                     Rotation & scaling only affine transformation matrix (MRI -> MEEG). MEG only.
    mri_to_meeg_r : np.ndarray, shape(4, 4)
                    Rotation only affine transformation matrix (MRI -> MEEG). MEG only.
    meeg_to_mri_trs : np.ndarray, shape(4, 4)
                      Full affine transformation matrix (MEEG -> MRI). MEG only.
    meeg_to_mri_tr : np.ndarray, shape(4, 4)
                     Rigid affine transformation matrix (MEEG -> MRI). MEG only.
    meeg_to_mri_rs : np.ndarray, shape(4, 4)
                     Rotation & scaling only affine transformation matrix (MEEG -> MRI). MEG only.
    meeg_to_mri_r : np.ndarray, shape(4, 4)
                    Rotation only affine transformation matrix (MEEG -> MRI). MEG only.
    closest_pts_nas : np.ndarray, shape(3, )
                      Closest points to the nas on the MRI. EEG only.
    closest_pts_lpa : np.ndarray, shape(3, )
                      Closest points to the lpa on the MRI. EEG only.
    closest_pts_rpa : np.ndarray, shape(3, )
                      Closest points to the rpa on the MRI. EEG only.
    closest_pts_eeg : np.ndarray, shape(eeg_ch_cnt, )
                      Closest points to the eeg positions on the MRI. EEG only.
    
    Parameters
    ----------
    signal_type : string
                  Either "EEG" or "MEG.
    
    rotors : np.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
    
    closest_pts_nas : np.ndarray, shape(3, )
                      Closest points to the nas on the MRI.
    closest_pts_lpa : np.ndarray, shape(3, )
                      Closest points to the lpa on the MRI.
    closest_pts_rpa : np.ndarray, shape(3, )
                      Closest points to the rpa on the MRI.
    closest_pts_eeg : np.ndarray, shape(eeg_ch_cnt, )
                      Closest points to the eeg positions on the MRI.
    """
    
    # The following parameters are only populated for MEG
    rotors: np.ndarray = None
    
    mri_to_meeg_trs: np.ndarray = None
    mri_to_meeg_tr: np.ndarray = None
    mri_to_meeg_rs: np.ndarray = None
    mri_to_meeg_r: np.ndarray = None
    
    meeg_to_mri_trs: np.ndarray = None
    meeg_to_mri_tr: np.ndarray = None
    meeg_to_mri_rs: np.ndarray = None
    meeg_to_mri_r: np.ndarray = None
    
    # The following paramters are only populated for EEG
    closest_pts_nas: np.ndarray = None
    closest_pts_lpa: np.ndarray = None
    closest_pts_rpa: np.ndarray = None
    closest_pts_eeg: np.ndarray = None
    
    def __init__(self, signal_type,
                 rotors = None,
                 closest_pts_nas = None, closest_pts_lpa = None, closest_pts_rpa = None, closest_pts_eeg = None):
        if (signal_type == "MEG"):
            self.closest_pts_nas = None
            self.closest_pts_lpa = None
            self.closest_pts_rpa = None
            self.closest_pts_eeg = None
            
            self.rotors = rotors
            self.mri_to_meeg_trs = get_transformation_matrix(self.rotors)
            self.mri_to_meeg_tr = _get_trans_and_rot_mat(self.rotors)
            self.mri_to_meeg_rs = _get_rot_and_scale_mat(self.rotors)
            self.mri_to_meeg_r = _get_rot_mat(self.rotors)
            
            self.meeg_to_mri_trs = np.linalg.inv(self.mri_to_meeg_trs)
            self.meeg_to_mri_tr = np.linalg.inv(self.mri_to_meeg_tr)
            self.meeg_to_mri_rs = np.linalg.inv(self.mri_to_meeg_rs)
            self.meeg_to_mri_r = np.linalg.inv(self.mri_to_meeg_r)
        if (signal_type == "EEG"):
            self.closest_pts_nas = closest_pts_nas
            self.closest_pts_lpa = closest_pts_lpa
            self.closest_pts_rpa = closest_pts_rpa
            self.closest_pts_eeg = closest_pts_eeg
            
            self.rotors = None
            
            self.mri_to_meeg_trs = np.eye(4)
            self.mri_to_meeg_tr = np.eye(4)
            self.mri_to_meeg_rs = np.eye(4)
            self.mri_to_meeg_r = np.eye(4)
            
            self.meeg_to_mri_trs = np.eye(4)
            self.meeg_to_mri_tr = np.eye(4)
            self.meeg_to_mri_rs = np.eye(4)
            self.meeg_to_mri_r = np.eye(4)

def run(subj_name, anatomy_path, signal_type, use_nasion = True, rec_info = None):
    """
    Execute the complete coregistration for a specific subject.
    
    Parameters
    ----------
    subj_name : string
                Name of the subject.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject.
    signal_type : string
                  Either "EEG" or "MEG".
    use_nasion : bool
                 Whether to use the nasion for registration, may be deacivated if the nasion is cut in the MRI.
    rec_info : string
               If signal_type is "EEG", rec_info is the EEG setup (1020 or 1005).
               If signal_type is "MEG", rec_info is the path to the FIF.file.
               
    Returns
    -------
    result : tuple of (finnpy.src_rec.coreg.Coreg, tuple of (list, list))
             - coreg : finnpy.src_rec.coreg.Coreg
                       Container class, populed with the following items:
             
                       - rotors : np.ndarray, shape(9,)
                         Sequence of rotors defining rotation (3), translation (3) and scaling (3).
            
                       - mri_to_meeg_trs : np.ndarray, shape(4, 4)
                         Full affine transformation matrix (MRI -> MEG)
                       - mri_to_meeg_tr : np.ndarray, shape(4, 4)
                         Rigid affine transformation matrix (MRI -> MEG)
                       - mri_to_meeg_rs : np.ndarray, shape(4, 4)
                         Rotation & scaling only affine transformation matrix (MRI -> MEG)
                       - mri_to_meeg_r : np.ndarray, shape(4, 4)
                         Rotation only affine transformation matrix (MRI -> MEG)
            
                       - meeg_to_mri_trs : np.ndarray, shape(4, 4)
                         Full affine transformation matrix (MEG -> MRI)
                       - meeg_to_mri_tr : np.ndarray, shape(4, 4)
                         Rigid affine transformation matrix (MEG -> MRI)
                       - meeg_to_mri_rs : np.ndarray, shape(4, 4)
                         Rotation & scaling only affine transformation matrix (MEG -> MRI)
                       - meeg_to_mri_r : np.ndarray, shape(4, 4)
                         Rotation only affine transformation matrix (MEG -> MRI)
             - bad_hsp_pts : (list, list)
                             Bad hsp points from the 1st and 2nd run of the coregistration.

    Raises
    ------
    AssertionError
        Raised if the signal type is not either 'EEG' or 'MEG'.
    """
    if (signal_type == "MEG"):
        sen_ref_pts = _read_meg_pts(mne.io.read_info(rec_info, verbose = "ERROR"))
    elif (signal_type == "EEG"):
        sen_ref_pts = finnpy.src_rec.utils.read_eeg_pts(rec_info)
    else:
        raise AssertionError("Signal type %s invalid, must be either 'EEG' or 'MEG'" % (signal_type,))
    
    (meeg_nasion_key, meeg_lpa_key, meeg_rpa_key) = _find_meeg_keys(sen_ref_pts, use_nasion)    
    
    (coreg_rotors, sen_ref_pts, bad_hsp_indices_outer, _) = calc_coreg(subj_name, anatomy_path, sen_ref_pts, signal_type,
                                                                       meeg_nasion_key, meeg_lpa_key, meeg_rpa_key,
                                                                       registration_scale_type = "free")
    (coreg_rotors[:6], sen_ref_pts, bad_hsp_indices_inner, _) = calc_coreg(subj_name, anatomy_path, sen_ref_pts, signal_type,
                                                                           meeg_nasion_key, meeg_lpa_key, meeg_rpa_key,
                                                                           registration_scale_type = "restricted", scale = coreg_rotors[6:9])
    
    if (signal_type == "EEG"):
        trans_mat = np.linalg.inv(get_transformation_matrix(coreg_rotors))
        sen_ref_pts["chs"] = np.asarray(sen_ref_pts["chs"])
        
        if (use_nasion):
            eeg_mri_nas = np.dot(np.concatenate((np.expand_dims(sen_ref_pts[meeg_nasion_key], axis = 0), np.asarray([[1]])), axis = 1), trans_mat.T)[:, :3]
        eeg_mri_lpa = np.dot(np.concatenate((np.expand_dims(sen_ref_pts[meeg_lpa_key], axis = 0), np.asarray([[1]])), axis = 1), trans_mat.T)[:, :3]
        eeg_mri_rpa = np.dot(np.concatenate((np.expand_dims(sen_ref_pts[meeg_rpa_key], axis = 0), np.asarray([[1]])), axis = 1), trans_mat.T)[:, :3]
        eeg_mri_chs = np.dot(np.concatenate((sen_ref_pts["chs"], np.ones((sen_ref_pts["chs"].shape[0], 1))), axis = 1), trans_mat.T)[:, :3]
        
        if (use_nasion):
            eeg_mri_pts = np.concatenate((eeg_mri_nas, eeg_mri_lpa, eeg_mri_rpa, eeg_mri_chs), axis = 0)
        else:
            eeg_mri_pts = np.concatenate((eeg_mri_lpa, eeg_mri_rpa, eeg_mri_chs), axis = 0)
        (surf_vert, surf_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.seghead")
        surf_vert /= 1000
        
        (_, _, closest_pts) = finnpy.src_rec.utils.find_closest_faces(eeg_mri_pts, surf_vert, surf_faces)        
        
        if (use_nasion): 
            return (Coreg("EEG", None, closest_pts[0, :], closest_pts[1, :], closest_pts[2, :], closest_pts[3:, :]), None)
        else:
            return (Coreg("EEG", None, None, closest_pts[0, :], closest_pts[1, :], closest_pts[2:, :]), None)
    else:
        return (Coreg("MEG", coreg_rotors, None, None, None, None), [bad_hsp_indices_outer, bad_hsp_indices_inner])

def plot_coregistration(coreg, signal_type, anatomy_path, subj_name, use_nasion = True, bad_hsp_pts = None, meg_data_path = None, eeg_setup = "1020"):
    """
    Plot the result of the coregistration from MRI to MEG using pyvista.
    
    Parameters
    ----------
    coreg : np.ndarray, shape(4, 4)
            MEG to MRI coregistration matrix.
    signal_type : string
                  "EEG" or "MEG".
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    use_nasion : boolean
                 Flag whether to use the nasion registration point. Not recommended for defaced patients.
    bad_hsp_pts : (list, list)
                  Bad hsp points from the 1st and 2nd run of the coregistration.
    meg_data_path : string
                    Path to the MEG file used in the coregistration.
    eeg_setup : string
                Setup of the EEG contacts. Defaults to 1020 system. Can be either 1020 or 1005.
                
    Raises
    ------
    AssertionError
        Raised if the signal type is not either 'EEG' or 'MEG'.
    """
    if (signal_type == "MEG"):
        meeg_pts = _read_meg_pts(mne.io.read_info(meg_data_path, verbose = "ERROR"))
    elif (signal_type == "EEG"):
        meeg_pts = finnpy.src_rec.utils.read_eeg_pts(eeg_setup)
        meeg_pts = {"Nz": coreg.closest_pts_nas, "lpa": coreg.closest_pts_lpa, "rpa": coreg.closest_pts_rpa, "chs": coreg.closest_pts_eeg, "labels": meeg_pts["labels"]}
    else:
        raise AssertionError("Signal type %s invalid, must be either 'EEG' or 'MEG'" % (signal_type,))
    
    if (signal_type == "MEG" and bad_hsp_pts is not None):
        for bad_hsp_pt in bad_hsp_pts:
            if (len(bad_hsp_pt) == 0):
                continue
            meeg_pts["hsp"] = np.delete(meeg_pts["hsp"], bad_hsp_pt, axis = 0)
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
        
    (meeg_nasion_key, meeg_lpa_key, meeg_rpa_key) = _find_meeg_keys(meeg_pts, use_nasion)
    
    (vert, faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.seghead")
    vert = vert / 1000
    if (signal_type == "MEG"):
        vert *= coreg.rotors[6:9]
    
    pl = pyvista.Plotter(window_size = (800, 600))
    # Plot head
    pl.add_mesh(pyvista.PolyData(vert, np.asarray([(3, face[0], face[1], face[2]) for face in faces], dtype = int).reshape(-1)), color = (.4, .4, .4), opacity = .9)

    # plot mri reference pts
    mri_pts = _load_mri_ref_pts(anatomy_path, subj_name)
    pl.add_points(np.asarray([mri_pts["LPA"][0], mri_pts["LPA"][1], mri_pts["LPA"][2]]), color = (0, .5, 1), point_size = 10)  # pylint: disable=unsubscriptable-object
    pl.add_points(np.asarray([mri_pts["NASION"][0], mri_pts["NASION"][1], mri_pts["NASION"][2]]), color = (0, .5, 1), point_size = 10)  # pylint: disable=unsubscriptable-object
    pl.add_points(np.asarray([mri_pts["RPA"][0], mri_pts["RPA"][1], mri_pts["RPA"][2]]), color = (0, .5, 1), point_size = 10)  # pylint: disable=unsubscriptable-object
    
    # plot meeg reference pts
    meeg_nasion = np.expand_dims(np.asarray(meeg_pts[meeg_nasion_key]), axis = 0)
    meeg_lpa = np.expand_dims(np.asarray(meeg_pts[meeg_lpa_key]), axis = 0)
    meeg_rpa = np.expand_dims(np.asarray(meeg_pts[meeg_rpa_key]), axis = 0)
    if (signal_type == "MEG"):
        meg_hpi = np.asarray(meeg_pts["hpi"])
        meg_hsp = np.asarray(meeg_pts["hsp"])
    else:
        eeg_chs = np.asarray(meeg_pts["chs"])
        eeg_labels = meeg_pts["labels"]
        
    if (signal_type == "MEG"):
        meeg_nasion = np.dot(np.concatenate((meeg_nasion, np.asarray([[1]])), axis = 1), coreg.meeg_to_mri_tr.T)[:, :3]
        meeg_lpa    = np.dot(np.concatenate((meeg_lpa, np.asarray([[1]])), axis = 1), coreg.meeg_to_mri_tr.T)[:, :3]  # noqa: E221
        meeg_rpa    = np.dot(np.concatenate((meeg_rpa, np.asarray([[1]])), axis = 1), coreg.meeg_to_mri_tr.T)[:, :3]  # noqa: E221
        meg_hpi = np.dot(np.concatenate((meg_hpi, np.ones((meg_hpi.shape[0], 1))), axis = 1), coreg.meeg_to_mri_tr.T)[:, :3]
        meg_hsp = np.dot(np.concatenate((meg_hsp, np.ones((meg_hsp.shape[0], 1))), axis = 1), coreg.meeg_to_mri_tr.T)[:, :3]
    
    pl.add_points(np.asarray(meeg_nasion), color = (1., 0., 0.), point_size = 30, render_points_as_spheres=True)
    pl.add_points(np.asarray(meeg_lpa), color = (1., 0.425, 0.), point_size = 30, render_points_as_spheres=True)
    pl.add_points(np.asarray(meeg_rpa), color = (1., 0.425, 0.), point_size = 30, render_points_as_spheres=True)
    
    if (signal_type == "MEG"):
        pl.add_points(np.asarray(meg_hpi), color = (1., 0.8, 0.), point_size = 15, render_points_as_spheres=True)
        pl.add_points(np.asarray(meg_hsp), color = (1., 1., 0.), point_size = 10, render_points_as_spheres=True)
    else:
        pl.add_points(np.asarray(eeg_chs), color = (1., 1., 0.), point_size = 10, render_points_as_spheres=True)
        for ch_idx in range(len(eeg_chs)):
            pl.add_text(eeg_labels[ch_idx], position = (eeg_chs[ch_idx, 0], eeg_chs[ch_idx, 1], eeg_chs[ch_idx, 2]), font_size = 12)
    
    pl.show()

def _read_meg_pts(rec_meta_info):
    """
    Load MEG reference points.
    
    Parameters
    ----------
    rec_meta_info : mne.io.read_info
                    MEG scan meta info, obtailable via mne.io.read_info
    
    Returns
    -------
    ref_pts : dict, ('nasion', 'lpa', 'rpa', 'hsp', 'coord_frame')
              MEG reference points for coregistration.
              
    Raises
    ------
    AssertionError
        Raised if coordinate frame is bad.
    """
    ref_pts = {"nasion": None, "lpa": None, "rpa": None, "hpi": list(), "hsp": list(), "coord_frame": list()}
    for ref_pt in rec_meta_info["dig"]:
        if (ref_pt["kind"] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL):
            if (ref_pt["ident"].real == mne.io.constants.FIFF.FIFFV_POINT_NASION):
                ref_pts["nasion"] = ref_pt["r"]
            elif (ref_pt["ident"].real == mne.io.constants.FIFF.FIFFV_POINT_LPA):
                ref_pts["lpa"] = ref_pt["r"]
            elif (ref_pt["ident"].real == mne.io.constants.FIFF.FIFFV_POINT_RPA):
                ref_pts["rpa"] = ref_pt["r"]
        elif (ref_pt["kind"] == mne.io.constants.FIFF.FIFFV_POINT_HPI):
            ref_pts["hpi"].append(ref_pt["r"])
        elif (ref_pt["kind"] == mne.io.constants.FIFF.FIFFV_POINT_EXTRA):
            ref_pts["hsp"].append(ref_pt["r"])
        else:
            continue
        ref_pts["coord_frame"].append(ref_pt["coord_frame"].real)
    if (np.sum(np.asarray(ref_pts["coord_frame"]) == ref_pts["coord_frame"][0]) != len(ref_pts["coord_frame"])):
        raise AssertionError("Coordinate frame is not universal")
    ref_pts["coord_frame"] = ref_pts["coord_frame"][0]
    
    return ref_pts

def _find_meeg_keys(meeg_pts, use_nasion = True):
    """
    Find the nasion, lpa, and rpa keys in meeg_pts. E.g. potential nasion names are NASION, Nasion, nasion, NZ, and Nz.
    
    Parameters
    ----------
    meeg_pts : dict
               Dictionary with the following keys populated: nasion, lpa, rpa, and others.
    use_nasion : boolean
                 Flag whether to use the nasion registration point. Not recommended for defaced patients.
    
    Returns
    -------
    result : tuple of (string, string, string)
        - meeg_nasion_key : string
          Name of the nasion key
        - meeg_lpa_key : string
          Name of the lpa key
        - meeg_rpa_key : string
          Name of the rpa key
          
    Raises
    ------
    AssertionError
        Cannot identify keys/names for nasion (if used), lpa, and/or rpa pts.
    """
    if (use_nasion):
        if ("NASION" in meeg_pts.keys()):
            meeg_nasion_key = "NASION"
        elif ("Nasion" in meeg_pts.keys()):
            meeg_nasion_key = "Nasion"
        elif ("nasion" in meeg_pts.keys()):
            meeg_nasion_key = "nasion"
        elif ("NZ" in meeg_pts.keys()):
            meeg_nasion_key = "NZ"
        elif ("Nz" in meeg_pts.keys()):
            meeg_nasion_key = "Nz"
        elif ("Nas" in meeg_pts.keys()):
            meeg_nasion_key = "Nas"
        elif ("nas" in meeg_pts.keys()):
            meeg_nasion_key = "nas"
        else:
            raise AssertionError("Cannot find nasion in meeg points.")
    else:
        meeg_nasion_key = None
        
    if ("LPA" in meeg_pts.keys()):
        meeg_lpa_key = "LPA"
    elif ("Lpa" in meeg_pts.keys()):
        meeg_lpa_key = "Lpa"
    elif ("lpa" in meeg_pts.keys()):
        meeg_lpa_key = "lpa"
    else:
        raise AssertionError("Cannot find lpa key in meeg points.")
    
    if ("RPA" in meeg_pts.keys()):
        meeg_rpa_key = "RPA"
    elif ("Rpa" in meeg_pts.keys()):
        meeg_rpa_key = "Rpa"
    elif ("rpa" in meeg_pts.keys()):
        meeg_rpa_key = "rpa"
    else:
        raise AssertionError("Cannot find lpa key in meeg points.")
    
    return (meeg_nasion_key, meeg_lpa_key, meeg_rpa_key)

def calc_coreg(subj_name, anatomy_path, meeg_pts, signal_type,
                meeg_nasion_key, meeg_lpa_key, meeg_rpa_key,
                registration_scale_type = "free", scale = None, use_nasion = True, max_number_of_iterations = 500, coreg_thresh = 1e-10):
    """
    Coregisters MRI data (src) to MEG data (tgt).
    
    Parameters
    ----------
    subj_name : string
                Name of the subject.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    meeg_pts : dict
               Dictionary with the following keys populated: nasion, lpa, rpa, hpi, hsp.
    signal_type : string
                  "EEG" or "MEG".
    meeg_nasion_key : string
                      Name of the nasion key.
    meeg_lpa_key : string
                   Name of the lpa key.
    meeg_rpa_key : string
                   Name of the rpa key.
    registration_scale_type : string
                              Can be either "free" or "restricted".
                              If free, the initial registration may be scaled with
                              a uniform factor, no scaling with restricted.
    scale : np.ndarray, shape(3)
            Scale anatomy for registration.
    use_nasion : boolean
                 Flag whether to use the nasion registration point. Not recommended for defaced patients.
    max_number_of_iterations : int
                               Number of iterations per registration step (total 3),
                               defaults to 500 per registration step.
    coreg_thresh : float
                   Determines the accuracy of the coregistration operation.
    
    Returns
    -------
    Tuple of (np.ndarray, dict, np.ndarray, np.ndarray)
        - coreg_rotors : np.ndarray, shape(9 or 6,)
                         Coregistration rotors
        - meeg_pts : dict
                     Dictionary with the following keys populated: nasion, lpa, rpa, hpi, hsp.
        - bad_hsp_indices : np.ndarray, shape(ch_cnt,)
                            
        - hd_surf_vert : np.ndarray, shape(vtx_cnt, 3)
                   
    Raises
    ------
    AssertionError
        Only raised for MEG, if HSP registration points cannot be identified.
    """
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    # Find initial solution
    if (signal_type == "MEG"):
        if (len(meeg_pts["hsp"]) == 0):
            raise AssertionError("Cannot find hsp registration points for patient, likely missing. Proper coregistration impossible.")
    meeg_pts = copy.deepcopy(meeg_pts)
    
    mri_pts = _load_mri_ref_pts(anatomy_path, subj_name)
    hd_surf_vert = _load_hd_surf(anatomy_path, subj_name)
    
    if (scale is not None):
        mri_pts["LPA"] = mri_pts["LPA"] * scale
        mri_pts["NASION"] = mri_pts["NASION"] * scale
        mri_pts["RPA"] = mri_pts["RPA"] * scale
        hd_surf_vert *= scale
    
    if (use_nasion is False):
        if ("NASION" in mri_pts.keys()):
            mri_pts.pop("NASION")
        if (meeg_nasion_key in meeg_pts.keys()):
            meeg_pts.pop(meeg_nasion_key)
    mri_pts_initial = np.asarray([mri_pts["LPA"], mri_pts["NASION"], mri_pts["RPA"]])
    meeg_pts_initial = np.asarray([meeg_pts[meeg_lpa_key], meeg_pts[meeg_nasion_key], meeg_pts[meeg_rpa_key]])
    
    # Start with an rigid transformation estimate (less variables -> less complex)
    (coreg_rotors, coreg_mat) = _registrate_3d_points_restricted(mri_pts_initial, meeg_pts_initial, scale = (registration_scale_type == "free"))
    
    # Refine initial solution, 1st run; Allow for non-rigid transformations in refinement
    if (signal_type == "MEG"):
        (ptn_cnt, hsp_cnt) = _get_ref_ptn_cnt(meeg_pts)
        refined_weights = np.ones((ptn_cnt)); refined_weights[hsp_cnt + 1] = 2
    else:
        refined_weights = np.ones((3 + len(meeg_pts["chs"])))
        refined_weights[len(meeg_pts["chs"]) + 0] = 50
        if (use_nasion is True):
            refined_weights[len(meeg_pts["chs"]) + 1] = 200
            refined_weights[len(meeg_pts["chs"]) + 2] = 50
        else:
            refined_weights[len(meeg_pts["chs"]) + 1] = 50
    
    (coreg_rotors, coreg_mat, _, _) = _refine_registration(meeg_pts, hd_surf_vert, meeg_nasion_key, meeg_lpa_key, meeg_rpa_key, 
                                                           coreg_rotors, coreg_mat, refined_weights,
                                                           signal_type, registration_scale_type, 
                                                           coreg_thresh, max_number_of_iterations)
            
    # Remove non-fitting pts
    if (signal_type == "MEG"):
        (meeg_pts["hsp"], bad_hsp_indices) = _rm_bad_head_shape_pts(meeg_pts["hsp"], hd_surf_vert, coreg_mat)
    else:
        bad_hsp_indices = None
    
    # Refine initial solution, 2nd run; only really matters for MEG ref pts as EEG ref pts shouldn't be significantly off
    if (signal_type == "MEG"):
        (ptn_cnt, hsp_cnt) = _get_ref_ptn_cnt(meeg_pts)
        refined_weights = np.ones((ptn_cnt)); refined_weights[hsp_cnt + 1] = 10
        (coreg_rotors, coreg_mat, _, _) = _refine_registration(meeg_pts, hd_surf_vert, meeg_nasion_key, meeg_lpa_key, meeg_rpa_key, 
                                                               coreg_rotors, coreg_mat, refined_weights,
                                                               signal_type, registration_scale_type,
                                                               coreg_thresh, max_number_of_iterations)
    if (registration_scale_type == "restricted"):
        return (coreg_rotors[:6], meeg_pts, bad_hsp_indices, hd_surf_vert)  # No point in returning invalid values
    else:
        return (coreg_rotors, meeg_pts, bad_hsp_indices, hd_surf_vert)

def _load_mri_ref_pts(anatomy_path, subj_name):
    """
    Load MEG reference points.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    
    Returns
    -------
    meg_pts : dict, ('nasion', 'lpa', 'rpa', 'hsp', 'coord_frame')
              MRI reference points for coregistration.
    """
    (pre_mri_ref_pts, _) = mne.io.read_fiducials(anatomy_path + subj_name + "/bem/" + subj_name + "-fiducials.fif", verbose = "ERROR")
    mri_ref_pts = _format_fiducials(pre_mri_ref_pts)
    
    return mri_ref_pts

def _format_fiducials(pre_mri_ref_pts):
    """
    Transform an mne-fiducials object into an dictionary containing the fiducials.
    
    Parameters
    ----------
    pre_mri_ref_pts : list of dict(), obtained via mne.io.read_fiducials
                      MNE-formatted list of MRI fiducials.
                      
    Returns
    -------
    mri_ref_pts : dict(), ('LPA', 'NASION', 'RPA')
                  MRI reference points for coregistration.
    """
    mri_ref_pts = {"LPA": None, "NASION": None, "RPA": None}
        
    for pt_idx in range(len(pre_mri_ref_pts)):
        if (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_LPA):
            mri_ref_pts["LPA"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION):
            mri_ref_pts["NASION"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_RPA):
            mri_ref_pts["RPA"] = pre_mri_ref_pts[pt_idx]["r"]
    
    return mri_ref_pts

def _load_hd_surf(anatomy_path, subj_name):
    """
    Load freesurfer extracted hd surface model vertices. If this information does not yet exists, it is created using freesurfer.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject.
    subj_name : string
                Subject name.
    
    Returns
    -------
    hd_surf_vert : np.ndarray, shape (n, 3)
                   High resulution surface model generated via freesurfer.
    """
    if (os.path.exists(anatomy_path + subj_name + "/surf/lh.seghead") is False):
        finnpy.src_rec.extract_anatomy.get_head_model(anatomy_path, subj_name)
    
    (hd_surf_vert, _) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.seghead")
    hd_surf_vert /= 1000  # scale from m to mm
    
    return hd_surf_vert

def _registrate_3d_points_restricted(src_pts, tgt_pts, weights = None, scale = False):
    """
    Registrates src points to tgt points via Horns method. The resulting 4x4 transformation matrix may contain translation, rotation, and scaling.
    
    Parameters
    ----------
    src_pts : np.ndarray or list, shape(n, 3)
              Source points for the coregistration.
    tgt_pts : np.ndarray or list, shape(n, 3)
              Target points for the coregistration.
    weights : np.ndarray or list, shape(n)
              Weights of the individual pts.
    scale : boolean,
            Flag whether to apply uniform scaling,
            defaults to False
            
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - est_rotors : np.ndarray, shape(9,)
                            Rotors of the estimated transformation.
             - est_mat : np.ndarray, shape(4, 4)
                            Rotation matrix of the estimated transformation.
    """
    if (weights is None):
        weights = [1., 10., 1.]
    
    # Horns method
    # Scale is either uniform or None 
    weights  = np.expand_dims(np.asarray(weights), axis = 1)  # noqa: E221
    weights /= np.sum(weights)
    
    mu_src = np.dot(weights.T, src_pts)
    mu_tgt = np.dot(weights.T, tgt_pts)
    
    sigma_src_tgt = np.dot(src_pts.T, weights * tgt_pts) - np.outer(mu_src, mu_tgt)
    
    (u, _, v) = np.linalg.svd(sigma_src_tgt)
    rot = np.dot(v.T, u.T)
    if (np.linalg.det(rot) < 0):
        dir_mat = np.eye(3)
        dir_mat[2][2] = -1
        rot = np.dot(v.T, np.dot(dir_mat, u.T))
    
    if (scale):
        dev_tgt = tgt_pts - mu_tgt; dev_tgt *= dev_tgt
        dev_src = src_pts - mu_src; dev_src *= dev_src
        
        dev_tgt *= weights
        dev_src *= weights
        
        scale = np.sqrt(np.sum(dev_tgt) / np.sum(dev_src))
    else:
        scale = 1
    
    trans = mu_tgt.T - scale * np.dot(rot, mu_src.T) - (scale == 0) * np.dot(rot, mu_src.T)
    
    est_rotors = np.zeros((9,))
    est_rotors[0:3] = scipy.spatial.transform.Rotation.from_matrix(rot).as_euler("xyz")
    est_rotors[3:6] = trans[:, 0]
    est_rotors[6:9] = scale
    
    est_mat = np.zeros((4, 4))
    est_mat[:3, :3] = rot
    est_mat[:3, 3] = trans[:3, 0]
    est_mat[0, 0] *= scale; est_mat[1, 1] *= scale; est_mat[2, 2] *= scale
    est_mat[3, 3] = 1
    
    return (est_rotors, est_mat)

def _get_ref_ptn_cnt(meg_pts):
    """
    Get the total and hsp counts from MEG pts.
    
    Parameters
    ----------
    meg_pts : dict, ('lpa', rpa', 'nasion', 'hpi', 'hsp', 'coord_frame')
              Reference MEG points for the coregistration.
    
    Returns
    -------
    result : tuple of (int, int)
             - ptn_cnt : int
                         Number of points without hsp-points.
             - hsp_cnt : int
                         Number of hsp-points.
    """
    ptn_cnt = 0
    hsp_cnt = 0
    for key in list(meg_pts.keys()):
        if (key == "coord_frame"):
            continue
        if (key == "lpa"):
            ptn_cnt += 1
        if (key == "rpa"):
            ptn_cnt += 1
        if (key == "nasion"):
            ptn_cnt += 1
        if (key == "hpi"):
            ptn_cnt += len(meg_pts["hpi"])
        if (key == "hsp"):
            ptn_cnt += len(meg_pts["hsp"])
            hsp_cnt = len(meg_pts["hsp"])
    return (ptn_cnt, hsp_cnt)

def _refine_registration(src_pts, tgt_pts, meeg_nasion_key, meeg_lpa_key, meeg_rpa_key, 
                         last_rotors, last_mat, weights,
                         signal_type, registration_scale_type = "free",
                         coreg_thresh = .002, max_number_of_iterations = 500):
    """
    Refines an initial registration.
    
    Parameters
    ----------
    src_pts : np.ndarray or list, shape(n1, 3)
              Source points for the coregistration.
    tgt_pts : np.ndarray or list, shape(n1, 3)
              Target points for the coregistration.
    meeg_nasion_key : string,
                      Name of the nasion channel in meeg data.
    meeg_lpa_key : string,
                   Name of the lpa channel in meeg data.
    meeg_rpa_key : string,
                   Name of the rpa channel in meeg data.
    last_rotors : np.ndarray, shape(9,)
                  Original rotor estimates.
    last_mat : np.ndarray, shape(4, 4)
               Original rotation matrix estimate.
    weights : np.ndarray or list, shape(n,)
              Weights for the coregistration.
    signal_type : string
                  "EEG" or "MEG".
    registration_scale_type : string
                              If "free", scaling is estimated across 3 axis,
                              if "reduced", scaling is uniform,
                              defaults to "free".
    coreg_thresh : float
                   Coregionstration error threshold.
    max_number_of_iterations : int
                               Number of iterations.
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
             - last_rotors : np.ndarray, shape(9,)
                             Updated rotor estimates.
             - last_mat : np.ndarray, shape(4, 4)
                          Updated rotation matrix estimate.
             - tgt_pts_full : np.ndarray, shape(m2, 3)
                              Target points used for the coregistration.
             - src_pts_full : np.ndarray, shape(n2, 3)
                              Source points used for the coregistration.
                              
    Raises
    ------
    AssertionError
        If signal type is invalid, must be either 'EEG' or 'MEG'.
        If the registration type is invalid, has to be eitehr 'free' or 'restricted'.
    """
    for iteration_idx in range(max_number_of_iterations):
        if (signal_type == "MEG"):
            src_pts_partial = list(); src_pts_partial.extend(src_pts["hsp"])
            inv_pre_tgt_pts_partial = finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["hsp"])), last_mat)
            (tgt_indices, tree) = finnpy.src_rec.utils.find_nearest_neighbor(tgt_pts, inv_pre_tgt_pts_partial, "kdtree")
            tgt_pts_partial = list(); tgt_pts_partial.extend(tgt_pts[tgt_indices, :])
        elif (signal_type == "EEG"):
            src_pts_partial = list(); src_pts_partial.extend(src_pts["chs"])
            inv_pre_tgt_pts_partial = finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["chs"])), last_mat)
            (tgt_indices, tree) = finnpy.src_rec.utils.find_nearest_neighbor(tgt_pts, inv_pre_tgt_pts_partial, "kdtree")
            tgt_pts_partial = list(); tgt_pts_partial.extend(tgt_pts[tgt_indices, :])
        else:
            raise AssertionError("Signal type %s unknown, must be either 'EEG' or 'MEG'" % (signal_type,))
        
        src_pts_partial.append(src_pts[meeg_lpa_key])
        tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts[meeg_lpa_key])), last_mat), axis = 0), "kdtree")[0], :])
        src_pts_partial.append(src_pts[meeg_nasion_key])
        tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts[meeg_nasion_key])), last_mat), axis = 0), "kdtree")[0], :])
        src_pts_partial.append(src_pts[meeg_rpa_key])
        tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts[meeg_rpa_key])), last_mat), axis = 0), "kdtree")[0], :])
        
        if (signal_type == "MEG"):
            src_pts_partial.extend(src_pts["hpi"])
            tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["hpi"])), last_mat), "kdtree")[0], :])

        src_pts_full = np.asarray(src_pts_partial, dtype = np.float64)
        tgt_pts_full = np.asarray(tgt_pts_partial, dtype = np.float64)
        
        if (registration_scale_type == "free"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_free(tgt_pts_full, src_pts_full, weights, initial_guess = (0, 0, 0, 0, 0, 0, 1, 1, 1))
        elif (registration_scale_type == "restricted"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_restricted(tgt_pts_full, src_pts_full, weights, scale = 0)
        else:
            raise AssertionError("Invalid registration type %s, must be either 'free' (with scaling) or 'restriced' (w/o scaling)" % (registration_scale_type,))
        
        trans_diff = np.linalg.norm(last_rotors[3:6] - ref_trans_list[3:6]) * 1000
        last_angle = scipy.spatial.transform.Rotation.from_matrix(last_mat[:3, :3]).as_quat()
        ref_angle = scipy.spatial.transform.Rotation.from_matrix(ref_trans_mat[:3, :3]).as_quat()
        angle_diff = np.rad2deg(finnpy.src_rec.utils.calc_quat_angle(ref_angle, last_angle))
        scale_diff = np.max((ref_trans_list[6:9] - last_rotors[6:9]) / last_rotors[6:9] * 100)
        
        last_rotors = ref_trans_list
        last_mat = ref_trans_mat
        
        if (trans_diff < coreg_thresh and angle_diff < coreg_thresh and scale_diff < coreg_thresh):
            break
    
    if (iteration_idx == max_number_of_iterations):
        warnings.warn("Max number of iterations reached")
        
    return (last_rotors, last_mat, tgt_pts_full, src_pts_full)

def _rm_bad_head_shape_pts(meg_pts, mri_pts, trans_mat, distance_thresh = 5 / 1000):
    """
    Identify MEG points whose distance is too far from MRI points and remove those.
    
    Parameters
    ----------
    meg_pts : np.ndarray or list, (m, 3)
              MEG reference points.
    mri_pts : np.ndarray or list, (n, 3)
              MRI reference points.
    trans_mat : np.ndarray, shape(4, 4)
                MEG to MRI transformation matrix.
    distance_thresh : float
                      Maximum distance. Defaults to 5 mm.
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray) or np.ndarray
             If there are over the threshold distances, return surviving meg pts & invalid indices.
             Otherwise, return only surviving meg pts.
             - meg_pts : np.ndarray, shape(m - x, 3)
                         Pruned list of MEG pts.
             - surv_idx : np.ndarray, shape(x, )
                          Indices of invalid points.
    """
    # Applies inverse transformation matrix, hence transforms from MEG -> MRI instead of MRI -> MEG.
    loc_meg_pts = finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(meg_pts)), trans_mat)
    mri_indices = mri_pts[finnpy.src_rec.utils.find_nearest_neighbor(mri_pts, loc_meg_pts, "kdtree")[0], :]
    
    distance = np.linalg.norm(mri_indices - loc_meg_pts, axis = 1)
    
    mask = distance <= distance_thresh
    
    if (np.sum(distance > distance_thresh) > 0):
        return ((np.asarray(meg_pts)[mask, :]).tolist(), np.sort(np.argwhere(distance > distance_thresh).squeeze(1))[::-1])
    else:
        return ((np.asarray(meg_pts)[mask, :]).tolist(), np.asarray([]))

def _registrate_3d_points_free(src_pts, tgt_pts, weights = None, initial_guess = None):
    """
    Registrate src points to tgt points via least squares minimizing. The resulting 4x4 transformation matrix may contain translation, rotation, and scaling.
    
    Parameters
    ----------
    src_pts : np.ndarray, shape(m, 4)
              Source points for the registration.
    tgt_pts : np.ndarray, shape(n, 4)
              Target points for the registration.
    weights : np.ndarray, shape(m, 1)
              (Source) weights for the registration.
    initial_guess : np.ndarray or list or tuple or None, shape(9,)
                    Initial transformation guess, 
                    defaults to None for no translation/rotation/scaling.
                    
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - est_rotors : np.ndarray, shape(9,)
                            Updated rotor estimate.
             - est_mat : np.ndarray, shape(4, 4)
                         Updated transformation matrix.
    """
    if (weights is None):
        weights = [1., 10., 1.]
    
    if (initial_guess is None):
        initial_guess = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype = float)
    
    src_pts = np.concatenate((src_pts, np.ones((src_pts.shape[0], 1))), axis = 1)
    weights = np.expand_dims(weights, axis = 1)
    
    def _update_estimate(guess):
        est = functools.reduce(np.dot, [_translation(guess[3], guess[4], guess[5]),
                                        _rotation(guess[0], guess[1], guess[2]),
                                        _scaling(guess[6], guess[7], guess[8])])
        error = tgt_pts - (np.dot(src_pts, est.T)[:, :3])
 
        error *= weights
 
        return error.ravel()
    
    est_rotors, _, _, _, _ = scipy.optimize.leastsq(_update_estimate, initial_guess, full_output = True)
    
    est_mat = get_transformation_matrix(est_rotors)
    return (est_rotors, est_mat)  # angles are euler angles in xyz format

def _translation(x, y, z):
    """
    Calculate a translation matrix from x, y, and z.
         
    Parameters
    ----------
    x : float
        shift
    y : float
        shift
    z : float
        shift
     
    Returns
    -------
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
    """
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype = float)

def _rotation(x = 0, y = 0, z = 0):
    """
    Calculate a rotation matrix from x, y, and z.
         
    Parameters
    ----------
    x : float
        angle
    y : float
        angle
    z : float
        angle
     
    Returns
    -------
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
    """
    cos_x = np.cos(x); sin_x = np.sin(x)
    cos_y = np.cos(y); sin_y = np.sin(y)
    cos_z = np.cos(z); sin_z = np.sin(z)

    return np.array([[cos_y * cos_z, -cos_x * sin_z + sin_x * sin_y * cos_z, sin_x * sin_z + cos_x * sin_y * cos_z, 0],
                     [cos_y * sin_z, cos_x * cos_z + sin_x * sin_y * sin_z, - sin_x * cos_z + cos_x * sin_y * sin_z, 0],
                     [-sin_y, sin_x * cos_y, cos_x * cos_y, 0],
                     [0, 0, 0, 1]], dtype = float)

def _scaling(x = 1, y = 1, z = 1):
    """
    Calculate a scaling matrix from x, y, and z.
         
    Parameters
    ----------
    x : float
        scale
    y : float
        scale
    z : float
        scale
     
    Returns
    -------
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
    """
    return np.array([[x, 0, 0, 0],
                     [0, y, 0, 0],
                     [0, 0, z, 0],
                     [0, 0, 0, 1]], dtype = float)

def get_transformation_matrix(rotors):
    """
    Produce a full transformation matrix from rotors.
         
    Parameters
    ----------
    rotors : np.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
    """
    mat = functools.reduce(np.dot, [_translation(rotors[3], rotors[4], rotors[5]),
                                    _rotation(rotors[0], rotors[1], rotors[2]),
                                    _scaling(rotors[6], rotors[7], rotors[8])])
    return mat

def _get_rot_and_scale_mat(rotors):
    """
    Produce a rotation and scaling matrix from rotors.
         
    Parameters
    ----------
    rotors : np.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
    """
    mat = functools.reduce(np.dot, [_rotation(rotors[0], rotors[1], rotors[2]),
                                    _scaling(rotors[6], rotors[7], rotors[8])])
    
    return mat

def _get_trans_and_rot_mat(rotors):
    """
    Produce a rigid transformation matrix from rotors.
         
    Parameters
    ----------
    rotors : np.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
    """
    mat = functools.reduce(np.dot, [_translation(rotors[3], rotors[4], rotors[5]),
                                    _rotation(rotors[0], rotors[1], rotors[2])])
    
    return mat

def _get_rot_mat(rotors):
    """
    Produce a rotatio matrix from rotors.
         
    Parameters
    ----------
    rotors : np.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
    """
    mat = _rotation(rotors[0], rotors[1], rotors[2])
    
    return mat
