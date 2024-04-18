'''
Created on Oct 12, 2022

@author: voodoocode
'''

import functools
import numpy as np
import scipy.optimize
import scipy.spatial
import nibabel.freesurfer
import os
import mne
import warnings
import copy
import mayavi.mlab
import pyexcel

import finnpy.file_io.data_manager as dm
import finnpy.src_rec.freesurfer

class Coreg():
    """
    Container class, populed with the following items:
             
    rotors : numpy.ndarray, shape(9,)
    Sequence of rotors defining rotation (3), translation (3) and scaling (3).
    
    mri_to_meg_trs : numpy.ndarray, shape(4, 4)
    Full affine transformation matrix (MRI -> MEG)
    mri_to_meg_tr : numpy.ndarray, shape(4, 4)
    Rigid affine transformation matrix (MRI -> MEG)
    mri_to_meg_rs : numpy.ndarray, shape(4, 4)
    Rotation & scaling only affine transformation matrix (MRI -> MEG)
    mri_to_meg_r : numpy.ndarray, shape(4, 4)
    Rotation only affine transformation matrix (MRI -> MEG)
    
    meg_to_mri_trs : numpy.ndarray, shape(4, 4)
    Full affine transformation matrix (MEG -> MRI)
    meg_to_mri_tr : numpy.ndarray, shape(4, 4)
    Rigid affine transformation matrix (MEG -> MRI)
    meg_to_mri_rs : numpy.ndarray, shape(4, 4)
    Rotation & scaling only affine transformation matrix (MEG -> MRI)
    meg_to_mri_r : numpy.ndarray, shape(4, 4)
    Rotation only affine transformation matrix (MEG -> MRI)
    """
    rotors = None
    
    mri_to_meg_trs = None
    mri_to_meg_tr = None
    mri_to_meg_rs = None
    mri_to_meg_r = None
    
    meg_to_mri_trs = None
    meg_to_mri_tr = None
    meg_to_mri_rs = None
    meg_to_mri_r = None
    
    def __init__(self, rotors):
        self.rotors = rotors
        self.mri_to_meg_trs = _get_transformation_matrix(self.rotors)
        self.mri_to_meg_tr = _get_trans_and_rot_mat(self.rotors)
        self.mri_to_meg_rs = _get_rot_and_scale_mat(self.rotors)
        self.mri_to_meg_r = _get_rot_mat(self.rotors)
        
        self.meg_to_mri_trs = np.linalg.inv(self.mri_to_meg_trs)
        self.meg_to_mri_tr = np.linalg.inv(self.mri_to_meg_tr)
        self.meg_to_mri_rs = np.linalg.inv(self.mri_to_meg_rs)
        self.meg_to_mri_r = np.linalg.inv(self.mri_to_meg_r)

def run(subj_name, anatomy_path, rec_meta_info = None):
    """
    Executes the complete coregistration for a specific subject.
    
    Parameters
    ----------
    subj_name : string
                Name of the subject.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject.
    mode : string
           Either "EEG" or "MEG".
    rec_meta_info : mne.io.read_info
                    MEG scan meta info, obtailable via mne.io.read_info
    Returns
    -------
    coreg : finnpy.src_rec.coreg.Coreg
            Container class, populed with the following items:
             
            rotors : numpy.ndarray, shape(9,)
            Sequence of rotors defining rotation (3), translation (3) and scaling (3).
            
            mri_to_meg_trs : numpy.ndarray, shape(4, 4)
            Full affine transformation matrix (MRI -> MEG)
            mri_to_meg_tr : numpy.ndarray, shape(4, 4)
            Rigid affine transformation matrix (MRI -> MEG)
            mri_to_meg_rs : numpy.ndarray, shape(4, 4)
            Rotation & scaling only affine transformation matrix (MRI -> MEG)
            mri_to_meg_r : numpy.ndarray, shape(4, 4)
            Rotation only affine transformation matrix (MRI -> MEG)
            
            meg_to_mri_trs : numpy.ndarray, shape(4, 4)
            Full affine transformation matrix (MEG -> MRI)
            meg_to_mri_tr : numpy.ndarray, shape(4, 4)
            Rigid affine transformation matrix (MEG -> MRI)
            meg_to_mri_rs : numpy.ndarray, shape(4, 4)
            Rotation & scaling only affine transformation matrix (MEG -> MRI)
            meg_to_mri_r : numpy.ndarray, shape(4, 4)
            Rotation only affine transformation matrix (MEG -> MRI)
    
    """
    sen_ref_pts = load_meg_ref_pts(rec_meta_info)
        
    (coreg_rotors, sen_ref_pts, bad_hsp_indices_outer, hd_surf_vert) = _calc_coreg(subj_name, anatomy_path, sen_ref_pts, registration_scale_type = "free", mode = "MEG")
    (coreg_rotors[:6], sen_ref_pts, bad_hsp_indices_inner, hd_surf_vert) = _calc_coreg(subj_name, anatomy_path, sen_ref_pts, registration_scale_type = "restricted", scale = coreg_rotors[6:9], mode = "MEG")
        
    return (Coreg(coreg_rotors), [bad_hsp_indices_outer, bad_hsp_indices_inner])

def read_EEG_pts(system = "1020"):
    """
    system : string
             "1020" or 1005".
    """
    if (system == "1020"):
        pre_sen_ref_pts = pyexcel.get_sheet(file_name = __file__[:__file__.rindex("/")] + "/../res/1020_system.csv").to_array()
    elif(system == "1005"):
        pre_sen_ref_pts = pyexcel.get_sheet(file_name = __file__[:__file__.rindex("/")] + "/../res/1005_system.csv").to_array()
    else:
        raise NotImplementedError("Unknown setup.")
    sen_ref_pts = {"chs" : [], "labels" : []}
    for pre_sen_ref_pt in pre_sen_ref_pts:
        if (pre_sen_ref_pt[0] == "NAS"):
            label = "nasion"
        elif (pre_sen_ref_pt[0] == "LPA"):
            label = "lpa"
        elif (pre_sen_ref_pt[0] == "RPA"):
            label = "rpa"
        else:
            label = pre_sen_ref_pt[0]
            sen_ref_pts["chs"].append(np.asarray(pre_sen_ref_pt[1:])/1000)
            #sen_ref_pts["chs"].append(np.asarray(pre_sen_ref_pt[1:])/np.asarray([10, 10, 8]))
            sen_ref_pts["labels"].append(pre_sen_ref_pt[0])
        sen_ref_pts[label] = np.asarray(pre_sen_ref_pt[1:])/1000
        #sen_ref_pts[label] = np.asarray(pre_sen_ref_pt[1:])/np.asarray([10, 10, 8])
    return sen_ref_pts

def plot_coregistration(coreg, meg_pts, bad_hsp_pts, anatomy_path, subj_name, mode = "MEG"):
    """
    Plots the result of the coregistration from MRI to MEG using mayavi.
    
    Parameters
    ----------
    coreg : numpy.ndarray, shape(4, 4)
            MEG to MRI coregistration matrix.
    meg_pts : numpy.ndarray, shape(n, 4)
              MEG pts used in the coregistration.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    mode : string
           "EEG" or "MEG".
    """
    
    for bad_hsp_pt in bad_hsp_pts:
        if (len(bad_hsp_pt) == 0):
            continue
        meg_pts["hsp"] = np.delete(meg_pts["hsp"], bad_hsp_pt, axis = 0)
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (vert, faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.seghead")
    vert = vert * coreg.rotors[6:9] / 1000
        
    _ = mayavi.mlab.figure(size = (800, 800))
    mayavi.mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], faces, color = (.4, .4, .4), opacity = 0.9)
    
    mri_pts = _load_mri_ref_pts(anatomy_path, subj_name)
    mayavi.mlab.points3d(mri_pts["LPA"][0], mri_pts["LPA"][1], mri_pts["LPA"][2], scale_factor = .015,  color = (0, .5, 1))
    mayavi.mlab.points3d(mri_pts["NASION"][0], mri_pts["NASION"][1], mri_pts["NASION"][2], scale_factor = .015,  color = (0, .5, 1))
    mayavi.mlab.points3d(mri_pts["RPA"][0], mri_pts["RPA"][1], mri_pts["RPA"][2], scale_factor = .015,  color = (0, .5, 1))
    
    meg_nasion = np.expand_dims(np.asarray(meg_pts["nasion"]), axis = 0)
    meg_lpa = np.expand_dims(np.asarray(meg_pts["lpa"]), axis = 0)
    meg_rpa = np.expand_dims(np.asarray(meg_pts["rpa"]), axis = 0)
    if (mode == "MEG"):
        meg_hpi = np.asarray(meg_pts["hpi"])
        meg_hsp = np.asarray(meg_pts["hsp"])
    else:
        eeg_chs = meg_pts["chs"]
        eeg_labels = meg_pts["labels"]
    
    def _meg_to_mri_trans(coreg, meg_nasion, meg_lpa, meg_rpa, meg_hpi = None, meg_hsp = None, eeg_chs = None):
        meg_nasion  = np.dot(meg_nasion, coreg.meg_to_mri_tr[:3, :3].T); meg_nasion  += coreg.meg_to_mri_tr[:3, 3];
        meg_lpa     = np.dot(meg_lpa,    coreg.meg_to_mri_tr[:3, :3].T); meg_lpa     += coreg.meg_to_mri_tr[:3, 3];
        meg_rpa     = np.dot(meg_rpa,    coreg.meg_to_mri_tr[:3, :3].T); meg_rpa     += coreg.meg_to_mri_tr[:3, 3];
        if (mode == "MEG"):
            meg_hpi     = np.dot(meg_hpi,    coreg.meg_to_mri_tr[:3, :3].T); meg_hpi     += coreg.meg_to_mri_tr[:3, 3];
            meg_hsp     = np.dot(meg_hsp,    coreg.meg_to_mri_tr[:3, :3].T); meg_hsp     += coreg.meg_to_mri_tr[:3, 3];
        else:
            eeg_chs     = np.dot(eeg_chs,    coreg.meg_to_mri_tr[:3, :3].T); eeg_chs     += coreg.meg_to_mri_tr[:3, 3];
        
        if (mode == "MEG"):
            return (meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp)
        else:
            return (meg_nasion, meg_lpa, meg_rpa, eeg_chs)
    
    if (mode == "MEG"):
        (meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp) = _meg_to_mri_trans(coreg, meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp, None)
    else:
        (meg_nasion, meg_lpa, meg_rpa, eeg_chs) = _meg_to_mri_trans(coreg, meg_nasion, meg_lpa, meg_rpa, None, None, eeg_chs)
    
    mayavi.mlab.points3d(meg_nasion[:, 0], meg_nasion[:, 1], meg_nasion[:, 2], scale_factor = .015, color = (1, 0, 0))
    mayavi.mlab.points3d(meg_lpa[:, 0], meg_lpa[:, 1], meg_lpa[:, 2], scale_factor = .015,  color = (1, 0.425, 0))
    mayavi.mlab.points3d(meg_rpa[:, 0], meg_rpa[:, 1], meg_rpa[:, 2], scale_factor = .015,  color = (1, 0.425, 0))
    if (mode == "MEG"):
        mayavi.mlab.points3d(meg_hpi[:, 0], meg_hpi[:, 1], meg_hpi[:, 2], scale_factor = .01,   color = (1, 0.8, 0))
        mayavi.mlab.points3d(meg_hsp[:, 0], meg_hsp[:, 1], meg_hsp[:, 2], scale_factor = .0025, color = (1, 1, 0))
    else:
        mayavi.mlab.points3d(eeg_chs[:, 0], eeg_chs[:, 1], eeg_chs[:, 2], scale_factor = .0025, color = (1, 1, 0))
        for ch_idx in range(eeg_chs.shape[0]):
            mayavi.mlab.text3d(eeg_chs[ch_idx, 0], eeg_chs[ch_idx, 1], eeg_chs[ch_idx, 2], eeg_labels[ch_idx], scale = (.005, .005, .005))
    
    mayavi.mlab.show()

def load_meg_ref_pts(rec_meta_info):
    """
    Load MEG reference points.
    
    Parameters
    ----------
    rec_meta_info : mne.io.read_info
                    MEG scan meta info, obtailable via mne.io.read_info
    
    Returns
    -------
    meg_pts : dict, ('nasion', 'lpa', 'rpa', 'hsp', 'coord_frame')
              MEG reference points for coregistration.
    """
    ref_pts = {"nasion" : None, "lpa" : None, "rpa" : None, "hpi" : list(), "hsp" : list(), "coord_frame" : list()}
    for ref_pt in rec_meta_info["dig"]:
        if (ref_pt["kind"] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL):
            if (ref_pt["ident"].real == mne.io.constants.FIFF.FIFFV_POINT_NASION):
                ref_pts["nasion"] = ref_pt["r"]
            elif (ref_pt["ident"].real == mne.io.constants.FIFF.FIFFV_POINT_LPA):
                ref_pts["lpa"] = ref_pt["r"]
            elif (ref_pt["ident"].real == mne.io.constants.FIFF.FIFFV_POINT_RPA):
                ref_pts["rpa"] = ref_pt["r"]
        elif(ref_pt["kind"] == mne.io.constants.FIFF.FIFFV_POINT_HPI):
            ref_pts["hpi"].append(ref_pt["r"])
        elif(ref_pt["kind"] == mne.io.constants.FIFF.FIFFV_POINT_EXTRA):
            ref_pts["hsp"].append(ref_pt["r"])
        else:
            raise AssertionError("unknown point type")
        ref_pts["coord_frame"].append(ref_pt["coord_frame"].real)
    if (np.sum(np.asarray(ref_pts["coord_frame"]) == ref_pts["coord_frame"][0]) != len(ref_pts["coord_frame"])):
        raise AssertionError("Coordinate frame is not universal")
    ref_pts["coord_frame"] = ref_pts["coord_frame"][0]
    
    return ref_pts

def _calc_coreg(subj_name, anatomy_path, meg_pts, registration_scale_type = "free", scale = None, use_nasion = True,
               max_number_of_iterations = 500, mode = "MEG"):
    """
    Coregisters MRI data (src) to MEG data (tgt).
    
    Parameters
    ----------
    subj_name : string
                Name of the subject.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    meg_pts : dict
              Dictionary with the following keys populated: nasion, lpa, rpa, hpi, hsp.
    registration_scale_type : string
                              Can be either "free" or "restricted".
                              If free, the initial registration may be scaled with
                              a uniform factor, no scaling with restricted.
    scale : numpy.ndarray, shape(3)
            Scale anatomy for registration.
    use_nasion : boolean
                 Flag whether to use the nasion registration point. Not recommended for defaced patients.
    max_number_of_iterations : int
                               Number of iterations per registration step (total 3),
                               defaults to 500 per registration step.
    mode : string
           "EEG" or "MEG".
    
    Returns
    -------
    coreg_rotors : numpy.ndarray, shape (9,)
                   Coregistration rotors
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    ## Find initial solution
    if (mode == "MEG"):
        if (len(meg_pts["hsp"]) == 0):
            raise AssertionError("Cannot find hsp registration points for patient, likely missing. Proper coregistration impossible.")
    meg_pts = copy.deepcopy(meg_pts)
    
    mri_pts = _load_mri_ref_pts(anatomy_path, subj_name)
    hd_surf_vert = _load_hd_surf(anatomy_path, subj_name)
    
    if (scale is not None):
        mri_pts["LPA"] = mri_pts["LPA"] * scale
        mri_pts["NASION"] = mri_pts["NASION"] * scale
        mri_pts["RPA"] = mri_pts["RPA"] * scale
        hd_surf_vert *= scale
    
    if (use_nasion == False):
        mri_pts.pop("NASION")
        meg_pts.pop("nasion")
    mri_pts_initial = np.asarray([mri_pts["LPA"], mri_pts["NASION"], mri_pts["RPA"]])
    meg_pts_initial = np.asarray([meg_pts["lpa"], meg_pts["nasion"], meg_pts["rpa"]])
    
    #Start with an rigid transformation estimate (less variables -> less complex)
    (coreg_rotors, coreg_mat) = _registrate_3d_points_restricted(mri_pts_initial, meg_pts_initial, scale = (registration_scale_type == "free"))
    #===========================================================================
    # if (registration_scale_type == "restricted"):
    #     return (coreg_rotors[:6], meg_pts, None)
    # else:
    #     return (coreg_rotors, meg_pts, None)
    #===========================================================================
    
    thresh = 1e-10
    
    # Refine initial solution, 1st run; Allow for non-rigid transformations in refinement
    if (mode == "MEG"):
        (ptn_cnt, hsp_cnt) = _get_ref_ptn_cnt(meg_pts)
        refined_weights = np.ones((ptn_cnt)); refined_weights[hsp_cnt + 1] = 2
    else:
        refined_weights = np.ones((3 + len(meg_pts["chs"])))
        refined_weights[len(meg_pts["chs"]) + 0] = 50
        refined_weights[len(meg_pts["chs"]) + 1] = 200
        refined_weights[len(meg_pts["chs"]) + 2] = 50
        
        #=======================================================================
        # refined_weights[len(meg_pts["chs"]) + 0] = 10
        # refined_weights[len(meg_pts["chs"]) + 1] = 20
        # refined_weights[len(meg_pts["chs"]) + 2] = 10
        #=======================================================================
    (coreg_rotors, coreg_mat, _, _) = _refine_registration(meg_pts, hd_surf_vert,
                                                           coreg_rotors, coreg_mat, refined_weights,
                                                           trans_thresh = thresh, angle_thresh = thresh, scale_thresh = thresh,
                                                           max_number_of_iterations = max_number_of_iterations,
                                                           registration_scale_type = registration_scale_type, 
                                                           mode = mode)
            
    # Remove non-fitting pts
    if (mode == "MEG"):
        (meg_pts["hsp"], bad_hsp_indices) = _rm_bad_head_shape_pts(meg_pts["hsp"], hd_surf_vert, coreg_mat)
    else:
        bad_hsp_indices = None
    
    # Refine initial solution, 2nd run
    if (mode == "MEG"):
        (ptn_cnt, hsp_cnt) = _get_ref_ptn_cnt(meg_pts)
        refined_weights = np.ones((ptn_cnt)); refined_weights[hsp_cnt + 1] = 10
    else:
        refined_weights = np.ones((3 + len(meg_pts["chs"])))
        refined_weights[len(meg_pts["chs"]) + 0] = 50
        refined_weights[len(meg_pts["chs"]) + 1] = 200
        refined_weights[len(meg_pts["chs"]) + 2] = 50
        
        #=======================================================================
        # refined_weights[len(meg_pts["chs"]) + 0] = 10
        # refined_weights[len(meg_pts["chs"]) + 1] = 20
        # refined_weights[len(meg_pts["chs"]) + 2] = 10
        #=======================================================================
    (coreg_rotors, coreg_mat, _, _) = _refine_registration(meg_pts, hd_surf_vert,
                                                          coreg_rotors, coreg_mat, refined_weights,
                                                          trans_thresh = thresh, angle_thresh = thresh, scale_thresh = thresh,
                                                          max_number_of_iterations = max_number_of_iterations,
                                                          registration_scale_type = registration_scale_type, 
                                                          mode = mode)
    if (registration_scale_type == "restricted"):
        return (coreg_rotors[:6], meg_pts, bad_hsp_indices, hd_surf_vert) # No point in returning invalid values
    else:
        return (coreg_rotors, meg_pts, bad_hsp_indices, hd_surf_vert)

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
    (pre_mri_ref_pts, _) = mne.io.read_fiducials(anatomy_path + subj_name + "/bem/" + subj_name + "-fiducials.fif")
    mri_ref_pts = _format_fiducials(pre_mri_ref_pts)
    
    return mri_ref_pts

def _format_fiducials(pre_mri_ref_pts):
    """
    Transforms an mne-fiducials object into an dictionary containing the fiducials.
    
    Parameters
    ----------
    pre_mri_ref_pts : list of dict(), obtained via mne.io.read_fiducials
                      MNE-formatted list of MRI fiducials.
                      
    Returns
    -------
    mri_ref_pts : dict(), ('LPA', 'NASION', 'RPA')
                  MRI reference points for coregistration.
    """
    mri_ref_pts = {"LPA" : None, "NASION" : None, "RPA" : None}
        
    for pt_idx in range(len(pre_mri_ref_pts)):
        if (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_LPA):
            mri_ref_pts["LPA"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif(pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION):
            mri_ref_pts["NASION"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif(pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_RPA):
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
    hd_surf_vert : numpy.ndarray, shape (n, 3)
                   High resulution surface model generated via freesurfer.
    """
    
    if (os.path.exists(anatomy_path + subj_name + "/surf/lh.seghead") == False):
        finnpy.src_rec.freesurfer.calc_head_model(anatomy_path, subj_name)
    
    (hd_surf_vert, _) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.seghead")
    hd_surf_vert /= 1000 # scale from m to mm
    
    return hd_surf_vert

def _registrate_3d_points_restricted(src_pts, tgt_pts, weights = [1., 10., 1.], scale = False):
    """
    Registrates src points to tgt points via Horns method. The resulting 4x4 transformation matrix may contain translation, rotation, and scaling.
    
    Parameters
    ----------
    src_pts : numpy.ndarray or list, shape(n, 3)
              Source points for the coregistration.
    tgt_pts : numpy.ndarray or list, shape(n, 3)
              Target points for the coregistration.
    weights : numpy.ndarray or list, shape(n)
              Weights of the individual pts.
    scale : boolean,
            Flag whether to apply uniform scaling,
            defaults to False
            
    Returns
    -------
    est_rotors : np.ndarray, shape(9,)
                 Rotors of the estimated transformation.
    est_mat : np.ndarray, shape(4, 4)
              Rotation matrix of the estimated transformation.
    """
    
    #Horns method
    #Scale is either uniform or None 
    
    weights = np.expand_dims(np.asarray(weights), axis = 1)
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
        
        scale = np.sqrt(np.sum(dev_tgt)/np.sum(dev_src))
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
    ptn_cnt : int
              Number of points without hsp-points.
    hsp_cnt : int
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

def _refine_registration(src_pts, tgt_pts, 
                        last_rotors, last_mat, 
                        weights, 
                        trans_thresh = .002, angle_thresh = .002, scale_thresh = .002, 
                        #trans_thresh = .2, angle_thresh = .2, scale_thresh = .2, 
                        max_number_of_iterations = 500, 
                        registration_scale_type = "free", 
                        mode = "MEG"):
    """
    Refines an initial registration.
    
    Parameters
    ----------
    src_pts : numpy.ndarray or list, shape(n1, 3)
              Source points for the coregistration.
    tgt_pts : numpy.ndarray or list, shape(n1, 3)
              Target points for the coregistration.
    last_rotors : np.ndarray, shape(9,)
                  Original rotor estimates.
    last_mat : np.ndarray, shape(4, 4)
               Original rotation matrix estimate.
    weights : numpy.ndarray or list, shape(n,)
              Weights for the coregistration.
    trans_thresh : float
                   Translation error threshold.
    angle_thresh : float
                   Rotation error threshold.
    scale_thresh : float
                   Scaling error threshold.
    max_number_of_iterations : int
                               Number of iterations.
    registration_scale_type : string
                              If "free", scaling is estimated across 3 axis,
                              if "reduced", scaling is uniform,
                              defaults to "free".
    mode : string
           "EEG" or "MEG".
               
    Returns
    -------
    last_rotors : np.ndarray, shape(9,)
                  Updated rotor estimates.
    last_mat : np.ndarray, shape(4, 4)
               Updated rotation matrix estimate.
    tgt_pts_full : np.ndarray, shape(m2, 3)
                   Target points used for the coregistration.
    src_pts_full : np.ndarray, shape(n2, 3)
                   Source points used for the coregistration.
    """
    
    for iteration_idx in range(max_number_of_iterations):
        if (mode == "MEG"):
            src_pts_partial = list(); src_pts_partial.extend(src_pts["hsp"])
            inv_pre_tgt_pts_partial = finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["hsp"])), last_mat)
            (tgt_indices, tree) = finnpy.src_rec.utils.find_nearest_neighbor(tgt_pts, inv_pre_tgt_pts_partial, "kdtree")
            tgt_pts_partial = list(); tgt_pts_partial.extend(tgt_pts[tgt_indices, :])
        else:
            src_pts_partial = list(); src_pts_partial.extend(src_pts["chs"])
            inv_pre_tgt_pts_partial = finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["chs"])), last_mat)
            (tgt_indices, tree) = finnpy.src_rec.utils.find_nearest_neighbor(tgt_pts, inv_pre_tgt_pts_partial, "kdtree")
            tgt_pts_partial = list(); tgt_pts_partial.extend(tgt_pts[tgt_indices, :])
        
        src_pts_partial.append(src_pts["lpa"])
        tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["lpa"])), last_mat), axis = 0), "kdtree")[0], :])
        src_pts_partial.append(src_pts["nasion"])
        tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["nasion"])), last_mat), axis = 0), "kdtree")[0], :])
        src_pts_partial.append(src_pts["rpa"])
        tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["rpa"])), last_mat), axis = 0), "kdtree")[0], :])
        
        if (mode == "MEG"):
            src_pts_partial.extend(src_pts["hpi"])
            tgt_pts_partial.extend(tgt_pts[finnpy.src_rec.utils.find_nearest_neighbor(tree, finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(src_pts["hpi"])), last_mat), "kdtree")[0], :])

        src_pts_full = np.asarray(src_pts_partial, dtype = np.float64)
        tgt_pts_full = np.asarray(tgt_pts_partial, dtype = np.float64)
        
        if (registration_scale_type == "free"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_free(tgt_pts_full, src_pts_full, weights, initial_guess = (0, 0, 0, 0, 0, 0, 1, 1, 1))
        elif(registration_scale_type == "restricted"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_restricted(tgt_pts_full, src_pts_full, weights, scale = 0)
        
        trans_diff = np.linalg.norm(last_rotors[3:6] - ref_trans_list[3:6]) * 1000
        last_angle = scipy.spatial.transform.Rotation.from_matrix(last_mat[:3, :3]).as_quat()
        ref_angle = scipy.spatial.transform.Rotation.from_matrix(ref_trans_mat[:3, :3]).as_quat()
        angle_diff = np.rad2deg(finnpy.src_rec.utils.calc_quat_angle(ref_angle, last_angle))
        scale_diff = np.max((ref_trans_list[6:9] - last_rotors[6:9])/last_rotors[6:9] * 100)
        
        last_rotors = ref_trans_list
        last_mat = ref_trans_mat
        
        if (trans_diff < trans_thresh and angle_diff < angle_thresh and scale_diff < scale_thresh):
            break
        pass
    
    if (iteration_idx == max_number_of_iterations):
        warnings.warn("Max number of iterations reached")
        
    return (last_rotors, last_mat, tgt_pts_full, src_pts_full)

def _finalze_EEG_coreg(meg_pts, hd_surf_vert, last_mat):
    surf_sen_pts = {"chs" : [None for _ in range(len(meg_pts["chs"]))]}
    (surf_sen_pts["lpa"], tree) = finnpy.src_rec.utils.find_nearest_neighbor(hd_surf_vert, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["lpa"])), last_mat), axis = 0), "kdtree")
    surf_sen_pts["lpa"] = hd_surf_vert[surf_sen_pts["lpa"], :].squeeze(0)
    surf_sen_pts["nasion"] = hd_surf_vert[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["nasion"])), last_mat), axis = 0), "kdtree")[0], :].squeeze(0)
    surf_sen_pts["rpa"] = hd_surf_vert[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["rpa"])), last_mat), axis = 0), "kdtree")[0], :].squeeze(0)
    surf_sen_pts["chs"] = hd_surf_vert[finnpy.src_rec.utils.find_nearest_neighbor(tree, finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["chs"])), last_mat), "kdtree")[0], :]
    surf_sen_pts["labels"] = meg_pts["labels"]
    
    return surf_sen_pts

def _rm_bad_head_shape_pts(meg_pts, mri_pts, trans_mat, distance_thresh = 5/1000):
    """
    Identify MEG points whose distance is too far from MRI points and remove those.
    
    Parameters
    ----------
    meg_pts : numpy.ndarray or list, (m, 3)
              MEG reference points.
    mri_pts : numpy.ndarray or list, (n, 3)
              MRI reference points.
    trans_mat : np.ndarray, shape(4, 4)
                MEG to MRI transformation matrix.
    distance_threshold : float
                         Maximum distance. Defaults to 5 mm.
               
    Returns
    -------
    meg_pts : np.ndarray, shape(m - x, 3)
              Pruned list of MEG pts.
    """
    
    #Applies inverse transformation matrix, hence transforms from MEG -> MRI instead of MRI -> MEG.
    loc_meg_pts = finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(meg_pts)), trans_mat)
    mri_indices = mri_pts[finnpy.src_rec.utils.find_nearest_neighbor(mri_pts, loc_meg_pts, "kdtree")[0], :]
    
    distance = np.linalg.norm(mri_indices - loc_meg_pts, axis = 1)
    
    mask = distance <= distance_thresh
    
    if (np.sum(distance > distance_thresh) > 0):
        return ((np.asarray(meg_pts)[mask, :]).tolist(), np.sort(np.argwhere(distance > distance_thresh).squeeze(1))[::-1])
    else:
        return ((np.asarray(meg_pts)[mask, :]).tolist(), np.asarray([]))

def _registrate_3d_points_free(src_pts, tgt_pts, weights = [1., 10., 1.], initial_guess = None):
    """
    Registrates src points to tgt points via least squares minimizing. The resulting 4x4 transformation matrix may contain translation, rotation, and scaling.
    
    Parameters
    ----------
    src_pts : numpy.ndarray, shape(m, 4)
              Source points for the registration.
    tgt_pts : numpy.ndarray, shape(n, 4)
              Target points for the registration.
    weights : numpy.ndarray, shape(m, 1)
              (Source) weights for the registration.
    initial_guess : numpy.ndarray or list or tuple or None, shape(9,)
                    Initial transformation guess, 
                    defaults to None for no translation/rotation/scaling.
                    
    Returns
    -------
    est_rotors : numpy.ndarray, shape(9,)
                 Updated rotor estimate.
    est_mat : numpy.ndarray, shape(4, 4)
              Updated transformation matrix.
    """
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
    
    est_mat = _get_transformation_matrix(est_rotors)
    return (est_rotors, est_mat) #angles are euler angles in xyz format

def _translation(x, y, z):
    """
    Calculates a translation matrix from x, y, and z.
         
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
    trans : numpy.ndarray, shape(4, 4)
            Transformation matrix.
    """
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype = float)

def _rotation(x = 0, y = 0, z = 0):
    """
    Calculates a rotation matrix from x, y, and z.
         
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
    trans : numpy.ndarray, shape(4, 4)
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
    Calculates a scaling matrix from x, y, and z.
         
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
    trans : numpy.ndarray, shape(4, 4)
            Transformation matrix.
    """
    return np.array([[x, 0, 0, 0],
                     [0, y, 0, 0],
                     [0, 0, z, 0],
                     [0, 0, 0, 1]], dtype = float)

def _get_transformation_matrix(rotors):
    """
    Produces a full transformation matrix from rotors.
         
    Parameters
    ----------
    rotors : numpy.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : numpy.ndarray, shape(4, 4)
            Transformation matrix.
    """
    
    mat = functools.reduce(np.dot, [_translation(rotors[3], rotors[4], rotors[5]),
                                    _rotation(rotors[0], rotors[1], rotors[2]),
                                    _scaling(rotors[6], rotors[7], rotors[8])])
    return mat

def _get_rot_and_scale_mat(rotors):
    """
    Produces a rotation and scaling matrix from rotors.
         
    Parameters
    ----------
    rotors : numpy.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : numpy.ndarray, shape(4, 4)
            Transformation matrix.
    """
    mat = functools.reduce(np.dot, [_rotation(rotors[0], rotors[1], rotors[2]),
                                    _scaling(rotors[6], rotors[7], rotors[8])])
    
    return mat

def _get_trans_and_rot_mat(rotors):
    """
    Produces a rigid transformation matrix from rotors.
         
    Parameters
    ----------
    rotors : numpy.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : numpy.ndarray, shape(4, 4)
            Transformation matrix.
    """
    mat = functools.reduce(np.dot, [_translation(rotors[3], rotors[4], rotors[5]),
                                    _rotation(rotors[0], rotors[1], rotors[2])])
    
    return mat

def _get_rot_mat(rotors):
    """
    Produces a rotatio matrix from rotors.
         
    Parameters
    ----------
    rotors : numpy.ndarray, shape(9,)
             Sequence of rotors defining rotation (3), translation (3) and scaling (3).
     
    Returns
    -------
    trans : numpy.ndarray, shape(4, 4)
            Transformation matrix.
    """
    mat = _rotation(rotors[0], rotors[1], rotors[2])
    
    return mat





