"""
Created on Apr 18, 2024.

@author: voodoocode
"""

import numpy as np
import pyexcel

import finnpy.src_rec.coreg  # @UnresolvedImport
import finnpy.src_rec.utils  # @UnresolvedImport

def run(subj_name, anatomy_path):
    """
    Execute the complete coregistration for a specific subject.
    
    Parameters
    ----------
    subj_name : string
                Name of the subject.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject.
                   
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
    sen_ref_pts = _read_EEG_pts("1005")
        
    (coreg_rotors, sen_ref_pts, _, _) = finnpy.src_rec.coreg.calc_coreg(subj_name, anatomy_path, sen_ref_pts, "EEG", "nasion", "lpa", "rpa",
                                                                        registration_scale_type = "free")
    (coreg_rotors[:6], sen_ref_pts, _, hd_surf_vert) = finnpy.src_rec.coreg.calc_coreg(subj_name, anatomy_path, sen_ref_pts, "EEG", "nasion", "lpa", "rpa",
                                                                                       registration_scale_type = "restricted", scale = coreg_rotors[6:9])
    
    eeg_ref_pts = _finalze_EEG_coreg(sen_ref_pts, hd_surf_vert, finnpy.src_rec.coreg.get_transformation_matrix(coreg_rotors))
    
    coreg = finnpy.src_rec.coreg.Coreg(coreg_rotors)
    eeg_ref_pts["lpa"] = _rescale_EEG(eeg_ref_pts["lpa"], coreg)
    eeg_ref_pts["nasion"] = _rescale_EEG(eeg_ref_pts["nasion"], coreg)
    eeg_ref_pts["rpa"] = _rescale_EEG(eeg_ref_pts["rpa"], coreg)
    eeg_ref_pts["chs"] = _rescale_EEG(eeg_ref_pts["chs"], coreg)
    
    return (coreg_rotors[6:9], eeg_ref_pts)

def _rescale_EEG(pts, coreg):
    """
    Scales EEG pts.
    
    Parameters
    ----------
    pts : np.ndarray, shape(3) or shape(ch_cnt, 3)
    coreg : finnpy.src_rec.coreg.Coreg
    
    Returns
    -------
    np.ndarray, shape(3) or shape(ch_cnt, 3)
        Scaled points
    """
    if (len(pts.shape) == 1):
        pts = np.asarray([*pts, 1])
        pts = np.dot(pts, coreg.meg_to_mri_trs)
        pts = np.dot(pts, coreg.mri_to_meg_tr)
        return pts[:3]
    else:
        pts = np.concatenate([pts, np.ones([pts.shape[0], 1])], axis = 1)
        pts = np.dot(pts, coreg.meg_to_mri_trs)
        pts = np.dot(pts, coreg.mri_to_meg_tr)
        return pts[:, :3]

def _read_EEG_pts(system = "1020"):
    """
    Read the position of the idealized EEG points from a database.
    
    Parameters
    ----------
    system : string
             "1020" or 1005".
    
    Returns
    -------
    sen_ref_pts : dict()
                  - 'chs' contain the eeg positions (np.ndarray, shape(eeg_ch_cnt, 3)).
                  - 'labels' names of the EEG channels.
       
    Raises
    ------
    NotImplementedError
        EEG system is invalid, has to be either '1020' or '1005
    """
    if (system == "1020"):
        pre_sen_ref_pts = pyexcel.get_sheet(file_name = __file__[:__file__.rindex("/")] + "/../res/1020_system.csv").to_array()
    elif (system == "1005"):
        pre_sen_ref_pts = pyexcel.get_sheet(file_name = __file__[:__file__.rindex("/")] + "/../res/1005_system.csv").to_array()
    else:
        raise NotImplementedError("Unknown setup.")
    sen_ref_pts = {"chs": [], "labels": []}
    for pre_sen_ref_pt in pre_sen_ref_pts:
        if (pre_sen_ref_pt[0] == "NAS"):
            label = "nasion"
        elif (pre_sen_ref_pt[0] == "LPA"):
            label = "lpa"
        elif (pre_sen_ref_pt[0] == "RPA"):
            label = "rpa"
        else:
            label = pre_sen_ref_pt[0]
            sen_ref_pts["chs"].append(np.asarray(pre_sen_ref_pt[1:]) / 1000)
            sen_ref_pts["labels"].append(pre_sen_ref_pt[0])
        sen_ref_pts[label] = np.asarray(pre_sen_ref_pt[1:]) / 1000
    return sen_ref_pts

def _finalze_EEG_coreg(eeg_pts, hd_surf_vert, last_mat):
    """
    Identify most probable EEG coordinates.
    
    Identifies the closest points between the high-resolution head model between the EEG, 
    after applying the EEG to MRI coregistration. 
    
    Parameters
    ----------
    eeg_pts : dict()
              Different point types ('lpa', 'nasion', 'rpa', and 'eeg').
    hd_surf_vert : np.ndarray, shape(vtx_cnt, 3)
                   Surface points of the head model.
    last_mat : np.ndarray, shape(4, 4)
               Transformation matrix from the coregistration
               
    Returns
    -------
    dict
        Head points closest to the transformed EEG coordinates.
    """
    surf_sen_pts = {"chs": [None for _ in range(len(eeg_pts["chs"]))]}
    (surf_sen_pts["lpa"], tree) = finnpy.src_rec.utils.find_nearest_neighbor(hd_surf_vert, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(eeg_pts["lpa"])), last_mat), axis = 0), "kdtree")
    surf_sen_pts["lpa"] = hd_surf_vert[surf_sen_pts["lpa"], :].squeeze(0)
    surf_sen_pts["nasion"] = hd_surf_vert[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(eeg_pts["nasion"])), last_mat), axis = 0), "kdtree")[0], :].squeeze(0)
    surf_sen_pts["rpa"] = hd_surf_vert[finnpy.src_rec.utils.find_nearest_neighbor(tree, np.expand_dims(finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(eeg_pts["rpa"])), last_mat), axis = 0), "kdtree")[0], :].squeeze(0)
    surf_sen_pts["chs"] = hd_surf_vert[finnpy.src_rec.utils.find_nearest_neighbor(tree, finnpy.src_rec.utils.apply_inv_transformation(np.copy(np.asarray(eeg_pts["chs"])), last_mat), "kdtree")[0], :]
    surf_sen_pts["labels"] = eeg_pts["labels"]
    
    return surf_sen_pts
