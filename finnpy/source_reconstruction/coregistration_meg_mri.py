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

def _load_meg_ref_pts(rec_meta_info):
    """
    Load meg references points.
    
    :param rec_meta_info: Meta information from the MEG scan.
    
    :return: Reference points in the meg scan.
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

def _load_hd_surf(subj_name, subj_path):
    """
    Load freesurfer extracted hd surface model vertices. If this information does not yet exists, it is created using freesurfer.
    
    :param subj_name: Name of the subject
    :param subj_path: Path to the subject's head model data. In case this doesn't exists, it is created.
    
    :return: HD surface vertices (head model).
    """
    
    if (os.path.exists(subj_path + "surf/lh.seghead") == False):
        _calc_head_model(subj_name, subj_path)
    
    (hd_surf_vert, _) = nibabel.freesurfer.read_geometry(subj_path + "surf/lh.seghead")
    hd_surf_vert /= 1000 # scale from m to mm
    
    return hd_surf_vert

def _calc_head_model(subj_name, subj_path):
    """
    Calculate a head model to read hd surface vertices using freesurfer. Removes files not needed by this reconstruction. 
    
    :param subj_name: Name of the subject.
    :param subj_path: Path to the subject's freesurfer folder.
    """
    if (subj_path[-1] == "/"):
        subj_name = subj_path.split("/")[-2]
    else:
        subj_name = subj_path.split("/")[-1]
    
    cmd = [__file__[:__file__.rindex("/")] + "/fs_get_model.sh", subj_name]
    
    finnpy_utils.run_subprocess_in_custom_working_directory(subj_name, cmd)
    
    os.remove(os.environ["SUBJECTS_DIR"] + "/" + subj_name + "/mri/" + "seghead.mgz")
    shutil.rmtree(os.environ["SUBJECTS_DIR"] + "/" + subj_name + "/scripts")
    os.remove(os.environ["SUBJECTS_DIR"] + "/" + subj_name + "/surf/" + "lh.seghead.inflated")

def _load_mri_ref_pts(subj_path, subj_name):
    """
    Load mri reference points.
    
    :param subj_path: Path to subject files.
    :param subj_name: Subject name.
    
    :return: MRI reference points.
    """
    (pre_mri_ref_pts, _) = mne.io.read_fiducials(subj_path + "bem/" + subj_name + "-fiducials.fif")
    mri_ref_pts = finnpy_utils.format_fiducials(pre_mri_ref_pts)
    
    return mri_ref_pts

def _registrate_3d_points_free(src_pts, tgt_pts, weights = [1., 10., 1.], initial_guess = None):
    """
    Registrates src points to tgt points via least squares minimizing. The resulting 4x4 transformation matrix may contain translation, rotation, and scaling.
    
    :param src_pts: Array/list of 3d src pts.
    :param tgt_pts: Array/list of 3d tgt pts.
    :param weights: Weights of the individual pts.
    :param initial_guess: Initial transformation guess. Defaults to no translation/rotation/scaling.
    
    :return: Estimated rotors and transformation matrix (4x4).
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
    
    :param src_pts: Array/list of 3d src pts.
    :param tgt_pts: Array/list of 3d tgt pts.
    :param weights: Weights of the individual pts.
    :param scale: Flag whether to apply uniform scaling. Defaults to False.
    
    :return: Estimated rotors and transformation matrix (4x4).
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

def _refine_registration(meg_pts, mri_vert, 
                        last_rotors, last_mat, 
                        weights, 
                        trans_thresh = .002, angle_thresh = .002, scale_thresh = .002, 
                        #trans_thresh = .2, angle_thresh = .2, scale_thresh = .2, 
                        max_number_of_iterations = 500, 
                        registration_scale_type = "free"):
    """
    Refines an initial registration.
    
    :param meg_pts: MEG pts.
    :param mri_vert: MRI pts.
    :param last_rotors: Guessed rotos.
    :param last_mat: Guessed transformation matrix.
    :param weights: Weights to be applied.
    :param trans_thresh: Translation error threshold.
    :param angle_thresh: Rotation error threshold.
    :param scale_thresh: Scaling error threshold.
    :param max_number_of_iterations: Number of iterations.
    :param registration_scale_type: If "free", scaling is estimated across 3 axis, if "reduced", scaling is uniform. Defaults to "free".
    
    :return: Updated rotors and transformation matrices.
    """
    
    for iteration_idx in range(max_number_of_iterations):
        meg_pts_partial = list(); meg_pts_partial.extend(meg_pts["hsp"])
        inv_pre_mri_pts_partial = finnpy_utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["hsp"])), last_mat)
        (mri_indices, tree) = finnpy_utils.find_nearest_neighbor(mri_vert, inv_pre_mri_pts_partial, "kdtree")
        mri_pts_partial = list(); mri_pts_partial.extend(mri_vert[mri_indices, :])
        
        meg_pts_partial.append(meg_pts["lpa"]); meg_pts_partial.append(meg_pts["nasion"]); meg_pts_partial.append(meg_pts["rpa"]);
        mri_pts_partial.extend(mri_vert[finnpy_utils.find_nearest_neighbor(tree, np.expand_dims(finnpy_utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["lpa"])), last_mat), axis = 0), "kdtree")[0], :])
        mri_pts_partial.extend(mri_vert[finnpy_utils.find_nearest_neighbor(tree, np.expand_dims(finnpy_utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["nasion"])), last_mat), axis = 0), "kdtree")[0], :])
        mri_pts_partial.extend(mri_vert[finnpy_utils.find_nearest_neighbor(tree, np.expand_dims(finnpy_utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["rpa"])), last_mat), axis = 0), "kdtree")[0], :])
        
        meg_pts_partial.extend(meg_pts["hpi"])
        mri_pts_partial.extend(mri_vert[finnpy_utils.find_nearest_neighbor(tree, finnpy_utils.apply_inv_transformation(np.copy(np.asarray(meg_pts["hpi"])), last_mat), "kdtree")[0], :])
        meg_pts_full = np.asarray(meg_pts_partial, dtype = np.float64)
        mri_pts_full = np.asarray(mri_pts_partial, dtype = np.float64)
        
        if (registration_scale_type == "free"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_free(mri_pts_full, meg_pts_full, weights, initial_guess = (0, 0, 0, 0, 0, 0, 1, 1, 1))
        elif(registration_scale_type == "restricted"):
            (ref_trans_list, ref_trans_mat) = _registrate_3d_points_restricted(mri_pts_full, meg_pts_full, weights, scale = 0)
        
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
        
    return (last_rotors, last_mat, mri_pts_full, meg_pts_full)

#===============================================================================
# def score_coregistration(meg_pts, mri_pts, trans):
#     """
#     
#     """
#     scores = scipy.linalg.norm((np.dot(meg_pts, trans[:3, :3].T) + trans[:3, 3]) - mri_pts, axis = 1)
#     return (np.min(scores), np.mean(scores), np.max(scores))
#===============================================================================

def get_rigid_transform(rotors):
    """
    Transforms rotors into a rigid transformation.
    
    :param rotors: Input rotors.
    
    :return: Transformation matrix.
    """
    
    mat = np.zeros((4, 4))
    mat[:3, :3] = scipy.spatial.transform.Rotation.from_euler("xyz", rotors[0:3]).as_matrix()
    mat[:3, 3] = rotors[3:6]
    mat[3, 3] = 1
    
    return mat

def _rm_bad_head_shape_pts(meg_pts, mri_pts, trans_mat, distance_thresh = 5/1000):
    """
    Identify MEG points whose distance is too far from MRI points and remove those.
    
    :param meg_pts: MEG pts.
    :param mri_pts: MRI pts.
    :param trans_mat: MRI to MEG transformation matrix.
    :param distance_threshold: Maximum distance. Defaults to 5 mm.
    
    :return: Pruned list of MEG pts.
    """
    
    loc_meg_pts = finnpy_utils.apply_inv_transformation(np.copy(np.asarray(meg_pts)), trans_mat)
    mri_indices = mri_pts[finnpy_utils.find_nearest_neighbor(mri_pts, loc_meg_pts, "kdtree")[0], :]
    
    distance = np.linalg.norm(mri_indices - loc_meg_pts, axis = 1)
    
    mask = distance <= distance_thresh
    
    return (np.asarray(meg_pts)[mask, :]).tolist()

def _get_ref_ptn_cnt(meg_pts):
    """
    Get the total and hsp counts from MEG pts.
    
    :param meg_pts: MEG pts.
    
    :return: Total point count and hsp point count.
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
    
def calc_coreg(subj_name : str, subj_path, rec_meta_info, registration_scale_type = "restricted",
        max_number_of_iterations = 500):
    """
    Coregisters MRI data (src) to MEG data (tgt).
    
    :param subj_name: string
    Name of the subject.
    :param subj_path: Path to the subject's freesurfer folder.
    :param rec_meta_info: MEG scan meta info.
    :param registration_scale_type: Can be either "free" or "restricted". If free, the initial registration may be scaled with a uniform factor, no scaling with restricted.
    :param max_number_of_iterations: Number of iterations per registration step (total 3).
    
    :return: Coregistration rotors, and coregistered MEG/MRI points.
    """
    ## Find initial solution
    meg_pts = _load_meg_ref_pts(rec_meta_info)
    mri_pts = _load_mri_ref_pts(subj_path, subj_name)
    hd_surf_vert = _load_hd_surf(subj_name, subj_path)
    
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

def plot_coregistration(coreg, rec_meta_info, meg_pts, subj_path):
    """
    Plots the result of the coregistration from MRI to MEG using mayavi.
    
    :param coreg: Coregistration matrix.
    :param rec_meta_info: Meta info from MRI scan.
    :param meg_pts: MEG pts used in the coregistration.
    :param subj_path: Path to the subject's freesurfer files.
    """
    
    (vert, faces) = nibabel.freesurfer.read_geometry(subj_path + "/surf/lh.seghead")
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


