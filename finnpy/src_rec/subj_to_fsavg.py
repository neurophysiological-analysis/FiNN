"""
Created on Feb 22, 2024.

@author: voodoocode
"""

import numpy as np
import scipy.sparse
import nibabel.freesurfer
import os

import finnpy.src_rec.utils  # @UnresolvedImport


class Subj_to_fsavg_mdl():
    """
    Container class, populated with the following items.
    
    trans : numpy.ndarray, shape(valid_subj_vtx_cnt, valid_subj_vtx_cnt)
            Transformation matrix
    lh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                    Valid/supporting vertices for left hemisphere.
    rh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                    Valid/supporting vertices for right hemisphere.
    """
    
    def __init__(self, trans,
                 lh_vert, lh_faces, lh_valid_vert,
                 rh_vert, rh_faces, rh_valid_vert,
                 octa_mdl_vert, octa_mdl_faces):
        self.trans = trans
        
        self.lh_vert = lh_vert
        self.lh_faces = lh_faces
        self.lh_valid_vert = lh_valid_vert

        self.rh_vert = rh_vert
        self.rh_faces = rh_faces
        self.rh_valid_vert = rh_valid_vert
        
        self.octa_mdl_vert = octa_mdl_vert
        self.octa_mdl_faces = octa_mdl_faces
        
def prepare(fs_path, anatomy_path, subj_name):
    """
    Compute precursors of the mri projections.
    
    Parameters
    ---------- 
    fs_path : string
              Path to the freesurfer folder. Should contain the 'bin' folder, your license.txt, and sources.sh.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    """
    _calc_mri_maps(anatomy_path, subj_name, fs_path + "subjects/fsaverage/", "lh", True)
    _calc_mri_maps(anatomy_path, subj_name, fs_path + "subjects/fsaverage/", "rh", True)

def compute(cort_mdl, anatomy_path, subj_name, fs_path, overwrite = False):
    """
    Create a subject to fs average transformation matrix.
    
    Parameters
    ----------
    cort_mdl : finnpy.src_rec.cort_mdl.Cort_mdl
               Container populated with the following items:
               
               - lh_vert : numpy.ndarray, shape(lh_vtx_cnt, 3)
                           White matter surface model vertices (left hemisphere).
               - lh_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                            White matter surface model faces (left hemisphere).
               - lh_valid_vert : numpy.ndarray, shape(lh_vtx_cnt,)
                                 Valid flags for white matter surface model vertices (left hemisphere).
               - rh_vert : numpy.ndarray, shape(rh_vtx_cnt, 3)
                           White matter surface model vertices (right hemisphere).
               - rh_faces : numpy.ndarray, shape(rh_face_cnt, 3)
                            White matter surface model faces (right hemisphere).
               - lh_valid_vert : numpy.ndarray, shape(rh_vtx_cnt,)
                                 Valid flags for white matter surface model vertices (right hemisphere).
               - octa_model_vert : numpy.ndarray, shape(octa_mdl_vtx_cnt, 3)
                                   Octamodel vertices (left hemisphere).
               - octa_model_faces : numpy.ndarray, shape(octa_mdl_face_cnt, 3)
                                  Octamodel faces (right hemisphere).
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    fs_path : string
              Path to the freesurfer folder. Should contain the 'bin' folder, your license.txt, and sources.sh.
    overwrite : boolean
                Flag whether to overwrite MRI maps. Defaults to False.
               
    Returns
    -------
    subj_to_fsavg_mdl : finnpy.src_rec.subj_to_fsavg.Subj_to_fsavg_mdl
                        Container class, populated with the following items:
    
                        trans : numpy.ndarray, shape(valid_subj_vtx_cnt, valid_subj_vtx_cnt)
                                Transformation matrix
                        lh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                        Valid/supporting vertices for left hemisphere.
                        rh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                        Valid/supporting vertices for right hemisphere.
    """
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
        
    if (fs_path[-1] != "/"):
        fs_path += "/"
    
    # Find a transformation for all points
    (lh_sub_vert, lh_sub_faces, avg_lh_vert, lh_mri_map) = _calc_mri_maps(anatomy_path, subj_name, fs_path + "subjects/fsaverage/", "lh", overwrite)
    (rh_sub_vert, rh_sub_faces, avg_rh_vert, rh_mri_map) = _calc_mri_maps(anatomy_path, subj_name, fs_path + "subjects/fsaverage/", "rh", overwrite)
    
    # Calculate a projection from valid/supporting points to all points (subject space only)
    lh_proj = _calc_small_to_default_vertices_proj(cort_mdl.lh_valid_vert, lh_sub_vert, lh_sub_faces)
    rh_proj = _calc_small_to_default_vertices_proj(cort_mdl.rh_valid_vert, rh_sub_vert, rh_sub_faces)
    
    # Combine to derive transformation between subject space and fs average space
    valid_avg_lh_vert = finnpy.src_rec.utils.find_valid_vertices(avg_lh_vert, cort_mdl.octa_mdl_vert)
    valid_avg_rh_vert = finnpy.src_rec.utils.find_valid_vertices(avg_rh_vert, cort_mdl.octa_mdl_vert)
    lh_proj = lh_mri_map[np.where(valid_avg_lh_vert)[0]] * lh_proj
    rh_proj = rh_mri_map[np.where(valid_avg_rh_vert)[0]] * rh_proj
    
    # Reformat
    trans_mat = scipy.sparse.lil_matrix((lh_proj.shape[0] + rh_proj.shape[0], lh_proj.shape[1] + rh_proj.shape[1]))
    trans_mat[:lh_proj.shape[0], :lh_proj.shape[1]] += lh_proj
    trans_mat[lh_proj.shape[0]:, lh_proj.shape[1]:] += rh_proj
    trans_mat = trans_mat.tocsr()
    
    (avg_lh_vert, avg_lh_faces) = nibabel.freesurfer.read_geometry(fs_path + "subjects/fsaverage/surf/lh.white")
    (avg_rh_vert, avg_rh_faces) = nibabel.freesurfer.read_geometry(fs_path + "subjects/fsaverage/surf/rh.white")
    
    return Subj_to_fsavg_mdl(trans_mat,
                             avg_lh_vert / 1000, avg_lh_faces, valid_avg_lh_vert,
                             avg_rh_vert / 1000, avg_rh_faces, valid_avg_rh_vert,
                             cort_mdl.octa_mdl_vert, cort_mdl.octa_mdl_faces)

def apply(subj_to_fsavg_mdl, data):
    """
    Transform data from subject space to fs_average space.
    
    Parameters
    ----------
    subj_to_fsavg_mdl : finnpy.src_rec.subj_to_fsavg
                        - trans : numpy.ndarray, shape(valid_subj_vtx_cnt, valid_subj_vtx_cnt)
                                  Transformation matrix
                        - lh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                          Valid/supporting vertices for left hemisphere.
                        - rh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                          Valid/supporting vertices for right hemisphere.
    data : numpy.ndarray, shape(source_space_ch_cnt, samp_cnt)
           Source space (subject) data.
               
    Returns
    -------
    transformed_data : numpy.ndarray, shape(source_space_ch_cnt, samp_cnt)
                       Source space (fs-average) data.
    """
    return subj_to_fsavg_mdl.trans * data

def _calc_small_to_default_vertices_proj(valid_vert, sub_vert, sub_faces):
    """
    Calculate projection from subject vertices to small sphere vertices.
    
    Parameters
    ----------
    valid_vert : numpy.ndarray, shape(mri_vtx_cnt,)
                 Valid vertices
    sub_vert : numpy.ndarray, shape(mri_vtx_cnt, 3)
               High-definition vertices.
    sub_faces : numpy.ndarray, shape(mri_face_cnt, 3)
                High-definition faces.
               
    Returns
    -------
    proj : numpy.ndarray, shape(mri_face_cnt, valid_vtx_cnt)
           Projection from all vertices to valid vertices.
    """
    # Create an adjacency graph for use below
    adj_mat = scipy.sparse.lil_matrix((np.unique(sub_faces).shape[0], np.unique(sub_faces).shape[0]))
    adjacenies = list()
    for face in sub_faces:
        adjacenies.append([face[0], face[1]])
        adjacenies.append([face[0], face[2]])
        adjacenies.append([face[1], face[0]])
        adjacenies.append([face[1], face[2]])
        adjacenies.append([face[2], face[0]])
        adjacenies.append([face[2], face[1]])
    adjacenies = np.asarray(adjacenies)
    adj_mat[adjacenies[:, 0], adjacenies[:, 1]] = 1
    adj_mat = adj_mat.tocsr()
    adj_mat += scipy.sparse.eye(adj_mat.shape[0])
    
    # Starting from the minimal vertex set used in subject space (e.g. 4098), 
    # expands to include all vertices in subject space.
    vert_idx = np.where(valid_vert)[0]
    proj = scipy.sparse.eye(int(np.sum(valid_vert)), format='csr')
    mult = np.zeros(sub_vert.shape[0])
    recompute = True
    max_iter_cnt = 100
    for iter_idx in range(max_iter_cnt):
        if (iter_idx != 0):
            proj = proj[vert_idx, :]
        used_vert = adj_mat[:, vert_idx] if len(vert_idx) < sub_vert.shape[0] else adj_mat
        proj = used_vert * proj
        
        if (recompute):
            if (len(vert_idx) == sub_vert.shape[0]):
                row_sum = np.asarray(adj_mat.sum(-1))[:, 0]
                recompute = False
            else:
                mult[vert_idx] = 1
                row_sum = adj_mat * mult
                vert_idx = np.where(row_sum)[0]
    
        proj.data /= np.where(row_sum, row_sum, 1).repeat(np.diff(proj.indptr))
        if (len(vert_idx) == sub_vert.shape[0]):
            break
    
    return proj

def _calc_mri_maps(anatomy_path, subj_name, fs_avg_path, hemisphere, overwrite, 
                   floating_error_thresh = 1e-14):
    """
    Find the subject space points corresponding to fs avg space points.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    fs_avg_path : string
                  Path for fs average freesurfer files.
    hemisphere : string
                Hemisphere to compute for.
    overwrite : boolean
                Flag whether to overwrite preexisting mri maps.
    floating_error_thresh : float
                Precision threshold for values indistinguishable from zero.
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
             - sub_vert : numpy.ndarray, shape(mri_vtx_cnt, 3)
                          Vertices of the MRI model.
             - sub_faces : numpy.ndarray, shape(mri_face_cnt, 3)
                          Faces of the MRI model.
             - avg_vert : numpy.ndarray, shape(fs_avg_vtx_cnt, 3)
                          Vertices of fs-avg's sphere model.
             - mri_map : numpy.ndarray, shape(fs_avg_vtx_cnt, mri_vtx_cnt)
                          Translation from subject mri to fs-average sphere model.
    """
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (sub_vert, sub_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/" + hemisphere + ".sphere.reg")
    (avg_vert, _) = nibabel.freesurfer.read_geometry(fs_avg_path + "surf/" + hemisphere + ".sphere.reg")
    
    sub_vert /= np.expand_dims(np.linalg.norm(sub_vert, axis = 1), axis = 1)
    avg_vert /= np.expand_dims(np.linalg.norm(avg_vert, axis = 1), axis = 1)
    
    (neigh_indices, _) = finnpy.src_rec.utils.find_nearest_neighbor(sub_vert, avg_vert, "kdtree")
    
    neigh_faces = [list() for _ in range(sub_vert.shape[0])]
    for (face_idx, face) in enumerate(sub_faces):
        for vortex in face:
            neigh_faces[vortex].append(face_idx)
        
    avg_neigh_faces = [neigh_faces[vertex_idx] for vertex_idx in neigh_indices] 
    
    if ((os.path.exists(anatomy_path + subj_name + "/proj/" + "mri_subj_to_mri_fs_avg_" + "weights_" + hemisphere + ".npy") is False or  # noqa: W504
         os.path.exists(anatomy_path + subj_name + "/proj/" + "mri_subj_to_mri_fs_avg_" + "match_idx_" + hemisphere + ".npy") is False) or overwrite):
        # Prepare faces for the following loop
        sub_face_params = list()
        for sub_face in sub_faces:   
            sub_face_params.append(finnpy.src_rec.utils.compute_baryzentric_params(sub_vert, sub_face))
         
        # Check of points are within a triangle and if yes, save location
        # For reference, see https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
        # and W. Heidrich, Journal of Graphics, GPU, and Game Tools,Volume 10, Issue 3, 2005
        errors = np.ones((avg_vert.shape[0])) * np.iinfo(np.int32).max
        match_idx = np.ones((avg_vert.shape[0]), dtype = int) * -1 
        weights = np.zeros((avg_vert.shape[0], avg_vert.shape[1]))
        for (avg_vert_idx, avg_vortex) in enumerate(avg_vert):
            for sub_face_idx in avg_neigh_faces[avg_vert_idx]:
                sub_face = sub_faces[sub_face_idx]
                
                (alpha, beta, gamma) = finnpy.src_rec.utils.compute_baryzentric_coordinates(avg_vortex, sub_vert, sub_face, sub_face_params[sub_face_idx])
                 
                curr_error = (alpha > 0) * alpha + (beta > 0) * beta + (gamma > 0) * gamma - 1 + (alpha < 0) * np.abs(alpha) + (beta < 0) * np.abs(beta) + (gamma < 0) * np.abs(gamma)  
                 
                if (curr_error > errors[avg_vert_idx]):
                    continue
                 
                errors[avg_vert_idx] = curr_error
                weights[avg_vert_idx][0] = alpha; weights[avg_vert_idx][1] = beta; weights[avg_vert_idx][2] = gamma
                match_idx[avg_vert_idx] = sub_face_idx
     
        # For those points "outside" of faces, find the most close nearby point.
        # For reference, https://math.stackexchange.com/questions/588871/minimum-distance-between-point-and-face
        for bad_weights_idx in np.argwhere(np.abs(errors) > floating_error_thresh).squeeze(1):
            
            sub_face_idx = match_idx[bad_weights_idx]
            proj_pt = finnpy.src_rec.utils.find_closest_pt_in_face(avg_vert[bad_weights_idx], sub_vert, sub_faces[sub_face_idx, :])
            (alpha, beta, gamma) = finnpy.src_rec.utils.compute_baryzentric_coordinates(proj_pt, sub_vert, sub_faces[sub_face_idx, :], sub_face_params[sub_face_idx])
            
            weights[bad_weights_idx][0] = alpha; weights[bad_weights_idx][1] = beta; weights[bad_weights_idx][2] = gamma
        
        if (os.path.exists(anatomy_path + subj_name + "/proj/") is False):
            os.mkdir(anatomy_path + subj_name + "/proj/")
        
        np.save(anatomy_path + subj_name + "/proj/" + "mri_subj_to_mri_fs_avg_" + "weights_" + hemisphere + ".npy", weights)
        np.save(anatomy_path + subj_name + "/proj/" + "mri_subj_to_mri_fs_avg_" + "match_idx_" + hemisphere + ".npy", match_idx)
    else:
        weights = np.load(anatomy_path + subj_name + "/proj/" + "mri_subj_to_mri_fs_avg_" + "weights_" + hemisphere + ".npy")
        match_idx = np.load(anatomy_path + subj_name + "/proj/" + "mri_subj_to_mri_fs_avg_" + "match_idx_" + hemisphere + ".npy")
     
    face_idx = sub_faces[match_idx, :]
    row_ind = np.repeat(np.arange(len(avg_vert)), 3)
    mri_map = scipy.sparse.csr_matrix((weights.ravel(), (row_ind, face_idx.ravel())),
                                      shape=(len(avg_vert), len(sub_vert)))
    
    return (sub_vert, sub_faces, avg_vert, mri_map)
