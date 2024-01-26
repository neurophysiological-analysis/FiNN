'''
Created on Oct 12, 2022

@author: voodoocode
'''

import functools

import numpy as np
import scipy.optimize
import scipy.spatial
import scipy.linalg

import nibabel.freesurfer

import finnpy.source_reconstruction.utils as finnpy_utils
import os

import mayavi.mlab

import mne

import warnings

import shutil

import finnpy.file_io.data_manager as dm
import finnpy.source_reconstruction.mri_anatomy

def _load_meg_ref_pts(rec_meta_info):
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

def _load_hd_surf(anatomy_path, subj_name):
    """
    Load freesurfer extracted hd surface model vertices. If this information does not yet exists, it is created using freesurfer.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    
    Returns
    -------
    hd_surf_vert : numpy.ndarray, shape (n, 3)
                   High resulution surface model generated via freesurfer.
    """
    
    if (os.path.exists(anatomy_path + subj_name + "/surf/lh.seghead") == False):
        finnpy.source_reconstruction.mri_anatomy._calc_head_model(anatomy_path, subj_name)
    
    (hd_surf_vert, _) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.seghead")
    hd_surf_vert /= 1000 # scale from m to mm
    
    return hd_surf_vert

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
    mri_ref_pts = finnpy_utils.format_fiducials(pre_mri_ref_pts)
    
    return mri_ref_pts

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
    
    def translation(x, y, z):
        return np.array([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]], dtype = float)
    
    def rotation(x=0, y=0, z=0):
        cos_x = np.cos(x); sin_x = np.sin(x)
        cos_y = np.cos(y); sin_y = np.sin(y)
        cos_z = np.cos(z); sin_z = np.sin(z)
 
        return np.array([[cos_y * cos_z, -cos_x * sin_z + sin_x * sin_y * cos_z, sin_x * sin_z + cos_x * sin_y * cos_z, 0],
                         [cos_y * sin_z, cos_x * cos_z + sin_x * sin_y * sin_z, - sin_x * cos_z + cos_x * sin_y * sin_z, 0],
                         [-sin_y, sin_x * cos_y, cos_x * cos_y, 0],
                         [0, 0, 0, 1]], dtype = float)
    
    def scaling(x = 1, y = 1, z = 1):
        return np.array([[x, 0, 0, 0],
                         [0, y, 0, 0],
                         [0, 0, z, 0],
                         [0, 0, 0, 1]], dtype = float)
    
    def update_estimate(guess):
        est = functools.reduce(np.dot, [translation(guess[3], guess[4], guess[5]),
                                         rotation(guess[0], guess[1], guess[2]),
                                         scaling(guess[6], guess[7], guess[8])])
        error = tgt_pts - (np.dot(src_pts, est.T)[:, :3])
 
        error *= weights
 
        return error.ravel()
    
    est_rotors, _, _, _, _ = scipy.optimize.leastsq(update_estimate, initial_guess, full_output = True)
    
    est_mat = functools.reduce(np.dot, [translation(est_rotors[3], est_rotors[4], est_rotors[5]),
                                        rotation(est_rotors[0], est_rotors[1], est_rotors[2]),
                                        scaling(est_rotors[6], est_rotors[7], est_rotors[8])])
    return (est_rotors, est_mat) #angles are euler angles in xyz format

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

def _refine_registration(src_pts, tgt_pts, 
                        last_rotors, last_mat, 
                        weights, 
                        trans_thresh = .002, angle_thresh = .002, scale_thresh = .002, 
                        #trans_thresh = .2, angle_thresh = .2, scale_thresh = .2, 
                        max_number_of_iterations = 500, 
                        registration_scale_type = "free"):
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
        src_pts_partial = list(); src_pts_partial.extend(src_pts["hsp"])
        inv_pre_tgt_pts_partial = finnpy_utils.apply_inv_transformation(np.copy(np.asarray(src_pts["hsp"])), last_mat)
        (tgt_indices, tree) = finnpy_utils.find_nearest_neighbor(tgt_pts, inv_pre_tgt_pts_partial, "kdtree")
        tgt_pts_partial = list(); tgt_pts_partial.extend(tgt_pts[tgt_indices, :])
        
        src_pts_partial.append(src_pts["lpa"]); src_pts_partial.append(src_pts["nasion"]); src_pts_partial.append(src_pts["rpa"]);
        tgt_pts_partial.extend(tgt_pts[finnpy_utils.find_nearest_neighbor(tree, np.expand_dims(finnpy_utils.apply_inv_transformation(np.copy(np.asarray(src_pts["lpa"])), last_mat), axis = 0), "kdtree")[0], :])
        tgt_pts_partial.extend(tgt_pts[finnpy_utils.find_nearest_neighbor(tree, np.expand_dims(finnpy_utils.apply_inv_transformation(np.copy(np.asarray(src_pts["nasion"])), last_mat), axis = 0), "kdtree")[0], :])
        tgt_pts_partial.extend(tgt_pts[finnpy_utils.find_nearest_neighbor(tree, np.expand_dims(finnpy_utils.apply_inv_transformation(np.copy(np.asarray(src_pts["rpa"])), last_mat), axis = 0), "kdtree")[0], :])
        
        src_pts_partial.extend(src_pts["hpi"])
        tgt_pts_partial.extend(tgt_pts[finnpy_utils.find_nearest_neighbor(tree, finnpy_utils.apply_inv_transformation(np.copy(np.asarray(src_pts["hpi"])), last_mat), "kdtree")[0], :])
        src_pts_full = np.asarray(src_pts_partial, dtype = np.float64)
        tgt_pts_full = np.asarray(tgt_pts_partial, dtype = np.float64)
        
        if (registration_scale_type == "free"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_free(tgt_pts_full, src_pts_full, weights, initial_guess = (0, 0, 0, 0, 0, 0, 1, 1, 1))
        elif(registration_scale_type == "restricted"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_restricted(tgt_pts_full, src_pts_full, weights, scale = 0)
        
        trans_diff = np.linalg.norm(last_rotors[3:6] - ref_trans_list[3:6]) * 1000
        last_angle = scipy.spatial.transform.Rotation.from_matrix(last_mat[:3, :3]).as_quat()
        ref_angle = scipy.spatial.transform.Rotation.from_matrix(ref_trans_mat[:3, :3]).as_quat()
        angle_diff = np.rad2deg(finnpy_utils.calc_quat_angle(ref_angle, last_angle))
        scale_diff = np.max((ref_trans_list[6:9] - last_rotors[6:9])/last_rotors[6:9] * 100)
        
        last_rotors = ref_trans_list
        last_mat = ref_trans_mat
        
        if (trans_diff < trans_thresh and angle_diff < angle_thresh and scale_diff < scale_thresh):
            break
        pass
    
    if (iteration_idx == max_number_of_iterations):
        warnings.warn("Max number of iterations reached")
        
    return (last_rotors, last_mat, tgt_pts_full, src_pts_full)

def get_rigid_transform(rotors):
    """
    Transforms coregistration rotors into a rigid transformation.
         
    Parameters
    ----------
    rotors : numpy.ndarray, shape(9,)
             Rotors to derive a rigid transformation from.
     
    Returns
    -------
    rigid_transform : numpy.ndarray, shape(4, 4)
                      Rigid transformation matrix.
    """
    
    rigid_transform = np.zeros((4, 4))
    rigid_transform[:3, :3] = scipy.spatial.transform.Rotation.from_euler("xyz", rotors[0:3]).as_matrix()
    rigid_transform[:3, 3] = rotors[3:6]
    rigid_transform[3, 3] = 1
    
    return rigid_transform

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
    loc_meg_pts = finnpy_utils.apply_inv_transformation(np.copy(np.asarray(meg_pts)), trans_mat)
    mri_indices = mri_pts[finnpy_utils.find_nearest_neighbor(mri_pts, loc_meg_pts, "kdtree")[0], :]
    
    distance = np.linalg.norm(mri_indices - loc_meg_pts, axis = 1)
    
    mask = distance <= distance_thresh
    
    return (np.asarray(meg_pts)[mask, :]).tolist()

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
    
def calc_coreg(subj_name, anatomy_path, rec_meta_info, registration_scale_type = "restricted",
               max_number_of_iterations = 500):    
    """
    Coregisters MRI data (src) to MEG data (tgt).
    
    Parameters
    ----------
    subj_name : string
                Name of the subject.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    rec_meta_info : mne.io.read_info
                    MEG scan meta info, obtailable via mne.io.read_info
    registration_scale_type : string
                              Can be either "free" or "restricted".
                              If free, the initial registration may be scaled with
                              a uniform factor, no scaling with restricted.
    max_number_of_iterations : int
                               Number of iterations per registration step (total 3),
                               defaults to 500 per registration step.
    
    Returns
    -------
    coreg_rotors : numpy.ndarray, shape (9,)
                   Coregistration rotors
    meg_pts : dict, ('nasion', 'lpa', 'rpa', 'hsp', 'coord_frame')
              coregistered MEG/MRI points
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    ## Find initial solution
    meg_pts = _load_meg_ref_pts(rec_meta_info)
    mri_pts = _load_mri_ref_pts(anatomy_path, subj_name)
    hd_surf_vert = _load_hd_surf(anatomy_path, subj_name)
    
    mri_pts_initial = np.asarray([mri_pts["LPA"], mri_pts["NASION"], mri_pts["RPA"]])
    meg_pts_initial = np.asarray([meg_pts["lpa"], meg_pts["nasion"], meg_pts["rpa"]])
    
    (coreg_rotors, coreg_mat) = _registrate_3d_points_restricted(mri_pts_initial, meg_pts_initial, scale = (registration_scale_type == "free"))
    
    thresh = 1e-10
    
    # Refine initial solution, 1st run
    (ptn_cnt, hsp_cnt) = _get_ref_ptn_cnt(meg_pts)
    refined_weights = np.ones((ptn_cnt)); refined_weights[hsp_cnt + 1] = 2
    (coreg_rotors, coreg_mat, _, _) = _refine_registration(meg_pts, hd_surf_vert,
                                                          coreg_rotors, coreg_mat, refined_weights,
                                                          trans_thresh = thresh, angle_thresh = thresh, scale_thresh = thresh,
                                                          max_number_of_iterations = max_number_of_iterations,
                                                          registration_scale_type = registration_scale_type)
            
    # Remove non-fitting pts
    meg_pts["hsp"] = _rm_bad_head_shape_pts(meg_pts["hsp"], hd_surf_vert, coreg_mat)
    
    # Refine initial solution, 2nd run
    (ptn_cnt, hsp_cnt) = _get_ref_ptn_cnt(meg_pts)
    refined_weights = np.ones((ptn_cnt)); refined_weights[hsp_cnt + 1] = 10
    (coreg_rotors, coreg_mat, _, _) = _refine_registration(meg_pts, hd_surf_vert,
                                                          coreg_rotors, coreg_mat, refined_weights,
                                                          trans_thresh = thresh, angle_thresh = thresh, scale_thresh = thresh,
                                                          max_number_of_iterations = max_number_of_iterations,
                                                          registration_scale_type = registration_scale_type)
    
    return (coreg_rotors, meg_pts)

def plot_coregistration(coreg, rec_meta_info, meg_pts, anatomy_path, subj_name):
    """
    Plots the result of the coregistration from MRI to MEG using mayavi.
    
    Parameters
    ----------
    coreg : numpy.ndarray, shape(4, 4)
            MEG to MRI coregistration matrix.
    rec_meta_info : mne.io.read_info
                    MEG scan meta information, obtainable through mne.io.read_info.
    meg_pts : numpy.ndarray, shape(n, 4)
              MEG pts used in the coregistration.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (vert, faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.seghead")
    vert/= 1000
    
    _ = mayavi.mlab.figure(size = (800, 800))
    mayavi.mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], faces, color = (.4, .4, .4), opacity = 0.9)
    
    meg_nasion = np.expand_dims(np.asarray(meg_pts["nasion"]), axis = 0)
    meg_lpa = np.expand_dims(np.asarray(meg_pts["lpa"]), axis = 0)
    meg_rpa = np.expand_dims(np.asarray(meg_pts["rpa"]), axis = 0)
    meg_hpi = np.asarray(meg_pts["hpi"])
    meg_hsp = np.asarray(meg_pts["hsp"])
    
    def meg_to_head_trans(trans, meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp):
        
        inv_trans = scipy.linalg.inv(trans)
        
        meg_nasion  = np.dot(meg_nasion, inv_trans[:3, :3].T);  meg_nasion  += inv_trans[:3, 3]
        meg_lpa     = np.dot(meg_lpa, inv_trans[:3, :3].T);     meg_lpa     += inv_trans[:3, 3]
        meg_rpa     = np.dot(meg_rpa, inv_trans[:3, :3].T);     meg_rpa     += inv_trans[:3, 3]
        meg_hpi     = np.dot(meg_hpi, inv_trans[:3, :3].T);     meg_hpi     += inv_trans[:3, 3]
        meg_hsp     = np.dot(meg_hsp, inv_trans[:3, :3].T);     meg_hsp     += inv_trans[:3, 3]
        
        return (meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp)
    
    (meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp) = meg_to_head_trans(coreg, meg_nasion, meg_lpa, meg_rpa, meg_hpi, meg_hsp)
    
    def mri_to_head_trans(trans, vert):
        
        inv_trans = scipy.linalg.inv(trans)
        
        vert = np.dot(vert, inv_trans[:3, :3].T); vert += inv_trans[:3, 3]
        
        return vert
    
    vert = mri_to_head_trans(rec_meta_info["dev_head_t"]["trans"], vert)
    
    mayavi.mlab.points3d(meg_nasion[:, 0], meg_nasion[:, 1], meg_nasion[:, 2], scale_factor = .015, color = (1, 0, 0))
    mayavi.mlab.points3d(meg_lpa[:, 0], meg_lpa[:, 1], meg_lpa[:, 2], scale_factor = .015,  color = (1, 0.425, 0))
    mayavi.mlab.points3d(meg_rpa[:, 0], meg_rpa[:, 1], meg_rpa[:, 2], scale_factor = .015,  color = (1, 0.425, 0))
    mayavi.mlab.points3d(meg_hpi[:, 0], meg_hpi[:, 1], meg_hpi[:, 2], scale_factor = .01,   color = (1, 0.8, 0))
    mayavi.mlab.points3d(meg_hsp[:, 0], meg_hsp[:, 1], meg_hsp[:, 2], scale_factor = .0025, color = (1, 1, 0))
    
    mayavi.mlab.show()


