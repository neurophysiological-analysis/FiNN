'''
Created on Oct 13, 2022

@author: voodoocode
'''

import numpy as np
import sklearn.neighbors
import scipy.sparse
import mpmath
import nibabel.freesurfer
import os
import shutil
import subprocess
import warnings

import mne.io

def format_fiducials(pre_mri_ref_pts):
    mri_ref_pts = {"LPA" : None, "NASION" : None, "RPA" : None}
        
    for pt_idx in range(len(pre_mri_ref_pts)):
        if (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_LPA):
            mri_ref_pts["LPA"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif(pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION):
            mri_ref_pts["NASION"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif(pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_RPA):
            mri_ref_pts["RPA"] = pre_mri_ref_pts[pt_idx]["r"]
    
    return mri_ref_pts

def run_subprocess_in_custom_working_directory(patient_id, cmd):
    path_to_tmp_cwd = "finnpy_" + patient_id + "_freesurfer_tmp_dir/" # A temporary working directory is needed as freesurfer saves intermediate results in files.
        
    if (os.path.exists(path_to_tmp_cwd + ".lock")): # Checks if the current directory is already worked in, if yes, raise error.
        raise AssertionError("Patient is already being worked on as %s already exists" % (path_to_tmp_cwd + ".lock",))
    
    os.makedirs(path_to_tmp_cwd, exist_ok = True)
    file = open(path_to_tmp_cwd + ".lock", "wb")
    file.close()
    
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, cwd = path_to_tmp_cwd, env = os.environ.copy())
    for c in iter(lambda: process.stderr.read(1), b""):
        print(c.decode(), end = "")
    
    shutil.rmtree(path_to_tmp_cwd) # And removed alter on

def init_fs_paths(subjects_path):
    os.environ["FREESURFER_HOME"]   = "/usr/local/freesurfer/7.2.0/"
    os.environ["FSFAST_HOME"]       = "/usr/local/freesurfer/7.2.0/fsfast/"
    os.environ["FSF_OUTPUT_FORMAT"] = "nii.gz"
    os.environ["SUBJECTS_DIR"]      = subjects_path[:-1] if (subjects_path[-1] == "/") else subjects_path
    os.environ["MNI_DIR"]           = "/usr/local/freesurfer/7.2.0/mni/"
    
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FREESURFER_HOME']+"bin/"
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FSFAST_HOME']+"bin/"

def fast_cross_product(a, b):
    """
    Seemingly numpy has too much overhead here to be fast
    """
    
    res = np.empty(a.shape)
    res[:, 0] = a[:, 1]*b[:, 2]-a[:, 2]*b[:, 1]
    res[:, 1] = a[:, 2]*b[:, 0]-a[:, 0]*b[:, 2]
    res[:, 2] = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    
    return res

def norm_vert(vert):
    size = np.sqrt(vert[:, 0]*vert[:, 0]+vert[:, 1]*vert[:, 1]+vert[:, 2]*vert[:, 2])
    vert[size > 0] /= size[size > 0, np.newaxis]
    return vert

def find_nearest_neighbor(src_model, tgt_model, method = "kdtree"):
    
    if (type(src_model) == sklearn.neighbors.KDTree or type(src_model) == sklearn.neighbors.BallTree):
        tree = src_model
        if (method == "kdtree"):
            neigh_indices = tree.query(tgt_model, k = 1)[1].squeeze(1)
        elif(method == "balltree"):
            neigh_indices = tree.query(tgt_model, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")
    else:
        if (method == "kdtree"):
            tree = sklearn.neighbors.KDTree(src_model)
            neigh_indices = tree.query(tgt_model, k = 1)[1].squeeze(1)
        elif(method == "balltree"):
            tree = sklearn.neighbors.BallTree(src_model)
            neigh_indices = tree.query(tgt_model, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")

    return (neigh_indices, tree)

def find_valid_vertices(mri_vert, model_vert, max_neighborhood_size = 5):
    (nearest, nearest_tree) = find_nearest_neighbor(mri_vert, model_vert)
    vert_valid = np.zeros((mri_vert.shape[0]))
    vert_valid[nearest] = 1
    (unique_nearest, duplicate_cnt) = np.unique(nearest, return_counts = True)
    for duplicate_idx in np.argwhere(duplicate_cnt > 1):
        duplicates_resolved = False
        for curr_neigborhood_size in range(2, max_neighborhood_size):
            (distance, neigh_candidates) = nearest_tree.query(unique_nearest[duplicate_idx], k = curr_neigborhood_size) 
            neigh_candidates = neigh_candidates.squeeze(1); neigh_candidates = neigh_candidates[np.argsort(distance)]
            for neighbor_candidate in neigh_candidates:
                if (vert_valid[neighbor_candidate] == 0):
                    vert_valid[neighbor_candidate] = 1
                    duplicates_resolved = True
                    break
                # In case all neighbors have been checked and are already valid, go on.
                duplicates_resolved = False
        if (duplicates_resolved == False):
            raise AssertionError("Freesurfer reconstruction seems incorrectly distributed.")
    
    return vert_valid

def magn_of_vec(vec):
    return np.sqrt(vec[:, 0]*vec[:, 0]+vec[:, 1]*vec[:, 1]+vec[:, 2]*vec[:, 2])

def apply_inv_transformation(data, trans):
    inv_trans = np.linalg.inv(trans)
    tmp = np.dot(inv_trans[:3, :3], data.T).T
    return tmp + inv_trans[:3, 3]    

def calc_quat_angle(a, b):
    w = a[3]*b[3] + a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    x = a[3]*b[0] - a[0]*b[3] - a[1]*b[2] + a[2]*b[1]
    y = a[3]*b[1] - a[1]*b[3] - a[0]*b[2] + a[2]*b[0]
    z = a[3]*b[2] - a[2]*b[3] - a[0]*b[1] + a[1]*b[0]
    
    return 2 * np.arctan2(np.linalg.norm(np.asarray((x, y, z))), np.abs(w))

def orient_mat_to_block_format(orient_mat):
    bdn = orient_mat.shape[0]
    tmp = np.arange(orient_mat.shape[1] * bdn, dtype=np.int64).reshape(bdn, orient_mat.shape[1])
    tmp = np.tile(tmp, (1, 1))
    ii = tmp.ravel()
    
    jj = np.arange(orient_mat.shape[0])[None, :]
    jj = jj * np.ones(orient_mat.shape[1], dtype = np.int64)[:, None]
    jj = jj.T.ravel()
    
    rot = scipy.sparse.coo_matrix((orient_mat.ravel(),
                                   np.concatenate((np.expand_dims(ii, axis = 1),
                                                   np.expand_dims(jj, axis = 1)), axis = 1).T)).tocsc()
    
    return rot

def get_orientation(acc_normals, valid_vert, patch_info, patch_indices,
                    coreg_trans_mat, double_precision = 40):
    
    rot_acc_normals = np.dot(acc_normals, coreg_trans_mat[:3, :3].T)
    
    pre_trans = np.zeros((np.where(valid_vert)[0].shape[0], 3))
    for (vertex_idx, vertex_id) in enumerate(np.searchsorted(np.where(valid_vert)[0], np.where(valid_vert)[0])): 
        pre_trans[vertex_idx] = np.sum(rot_acc_normals[patch_info[patch_indices[vertex_id]], :], axis = 0)
    pre_trans /= np.linalg.norm(pre_trans, axis = 1, keepdims = True)
    
    tmp = np.empty((pre_trans.shape[0], 3, 3))
    for idx in range(tmp.shape[0]):
        tmp[idx, :, :] = np.matmul(pre_trans[[idx], :].T, pre_trans[[idx], :])
    
    evec = list()
    for mat_idx in range(tmp.shape[0]):
        if ((tmp[mat_idx, :, :] == tmp[mat_idx, :, :].T).all() == False):
            raise AssertionError("Something went wrong with the orientation matrix")
        
        mat = mpmath.matrix(np.eye(3) - tmp[mat_idx, :, :])
        mat.ctx.dps = double_precision
        (_, loc_evec) = mpmath.eigsy(mat)
        
        evec.append(loc_evec.tolist())
    
    evec = np.asarray(evec, dtype = float)
    evec = evec[:, :, ::-1]
    direction = np.sign(np.matmul(np.expand_dims(pre_trans, axis = 1), evec[:, :, -1:]))
    direction[direction == 0] = 1
    evec *= direction
    evec = evec.swapaxes(1, 2)
    evec = evec.reshape(-1, 3)
    
    return evec

def calc_acc_hem_normals(geom_white_vert, geom_white_faces):
    vert0 = geom_white_vert[geom_white_faces[:, 0], :]
    vert1 = geom_white_vert[geom_white_faces[:, 1], :]
    vert2 = geom_white_vert[geom_white_faces[:, 2], :]
    
    vortex_normals = fast_cross_product(vert1 - vert0, vert2 - vert1)
    
    len_vortex_normals = np.linalg.norm(vortex_normals, axis = 1)
    
    vortex_normals[len_vortex_normals > 0] /= np.expand_dims(len_vortex_normals[len_vortex_normals > 0], axis = 1)
    
    acc_normals = np.zeros((geom_white_vert.shape[0], 3))
    for outer_idx in range(3):
        for inner_idx in range(3):
            value = np.zeros((geom_white_vert.shape[0],))
            for vertex_idx in range(geom_white_faces.shape[0]):
                value[geom_white_faces[vertex_idx, outer_idx]] += vortex_normals[vertex_idx, inner_idx]
            acc_normals[:, inner_idx] += value
    
    len_acc_normals = np.linalg.norm(acc_normals, axis = 1)
    acc_normals[len_acc_normals > 0] /= np.expand_dims(len_acc_normals[len_acc_normals > 0], axis = 1)
    
    return acc_normals

def find_vertex_patches(geom_white_vert, valid_geom_vert, geom_white_faces):
    lh_edges = scipy.sparse.coo_matrix((np.ones(3 * geom_white_faces.shape[0]),
                                        (np.concatenate((geom_white_faces[:, 0], geom_white_faces[:, 1], geom_white_faces[:, 2])),
                                         np.concatenate((geom_white_faces[:, 1], geom_white_faces[:, 2], geom_white_faces[:, 0])))),
                                        shape = (geom_white_vert.shape[0], geom_white_vert.shape[0]))
    lh_edges = lh_edges.tocsr()
    lh_edges += lh_edges.T
    lh_edges = lh_edges.tocoo()
    lh_edges_dists = np.linalg.norm(geom_white_vert[lh_edges.row, :] - geom_white_vert[lh_edges.col, :], axis = 1)
    lh_edges_adjacency = scipy.sparse.csr_matrix((lh_edges_dists, (lh_edges.row, lh_edges.col)), shape = lh_edges.shape)
    min_dist, _, min_idx = scipy.sparse.csgraph.dijkstra(lh_edges_adjacency, indices = np.where(valid_geom_vert)[0], min_only = True, return_predecessors = True)
    min_dist /= 1000
    
    sort_near_idx = np.argsort(min_idx)
    sort_min_idx = min_idx[sort_near_idx]
    breaks = np.where(sort_min_idx[1:] != sort_min_idx[:-1])[0] + 1
    
    starts = [0] + breaks.tolist()
    ends = breaks.tolist() + [len(min_idx)]
    
    patch_info = list()
    for patch_idx in range(len(starts)):
        patch_info.append(np.sort(sort_near_idx[starts[patch_idx]:ends[patch_idx]]))
    
    pre_patch_indices = sort_min_idx[breaks - 1]
    patch_indices = np.searchsorted(pre_patch_indices, np.where(valid_geom_vert)[0])
    
    return (patch_info, patch_indices)

def reset_model_orientation(lh_geom_white_vert, lh_geom_white_faces, valid_geom_lh_vert,
                            rh_geom_white_vert, rh_geom_white_faces, valid_geom_rh_vert,
                            fwd_sol, coreg_trans_mat):
     
    lh_acc_normals = calc_acc_hem_normals(lh_geom_white_vert, lh_geom_white_faces)
    rh_acc_normals = calc_acc_hem_normals(rh_geom_white_vert, rh_geom_white_faces)
     
    # Figures out which real vertices are represented by the same model vortex
    (lh_patch_info, lh_patch_indices) = find_vertex_patches(lh_geom_white_vert, valid_geom_lh_vert, lh_geom_white_faces)
    (rh_patch_info, rh_patch_indices) = find_vertex_patches(rh_geom_white_vert, valid_geom_rh_vert, rh_geom_white_faces)
     
    lh_orient = get_orientation(lh_acc_normals, valid_geom_lh_vert, lh_patch_info, lh_patch_indices, coreg_trans_mat)
    rh_orient = get_orientation(rh_acc_normals, valid_geom_rh_vert, rh_patch_info, rh_patch_indices, coreg_trans_mat)
     
    orient = np.concatenate((lh_orient, rh_orient), axis = 0)
     
    #get block matrix
    rot = orient_mat_to_block_format(orient[2::3, :])
     
    return fwd_sol * rot
    
def calc_mri_maps(subj_path, fs_avg_path, hemisphere, overwrite):
    (sub_vert, sub_faces) = nibabel.freesurfer.read_geometry(subj_path + "surf/" + hemisphere + ".sphere.reg")
    (avg_vert, _) = nibabel.freesurfer.read_geometry(fs_avg_path + "surf/" + hemisphere + ".sphere.reg")
    
    sub_vert /= np.expand_dims(np.linalg.norm(sub_vert, axis = 1), axis = 1)
    avg_vert /= np.expand_dims(np.linalg.norm(avg_vert, axis = 1), axis = 1)
    
    (neigh_indices, _) = find_nearest_neighbor(sub_vert, avg_vert, "kdtree")
    
    neigh_faces = [list() for _ in range(sub_vert.shape[0])]
    for (face_idx, face) in enumerate(sub_faces):
        for vortex in face:
            neigh_faces[vortex].append(face_idx)
        
    avg_neigh_faces = [neigh_faces[vertex_idx] for vertex_idx in neigh_indices] 
    
    if ((os.path.exists(subj_path + "proj/" + "mri_subj_to_mri_fs_avg_" + "weights_" + hemisphere + ".npy") == False or
         os.path.exists(subj_path + "proj/" + "mri_subj_to_mri_fs_avg_" + "match_idx_" + hemisphere + ".npy") == False) or overwrite):
        ## W. Heidrich, Journal of Graphics, GPU, and Game Tools,Volume 10, Issue 3, 2005
        #prep_faces
        sub_face_params = list()
        for sub_face in sub_faces:
            u = sub_vert[sub_face[1], :] - sub_vert[sub_face[0], :]
            v = sub_vert[sub_face[2], :] - sub_vert[sub_face[0], :]
            n = np.cross(u, v)
            
            tmp = np.dot(n, n)
            n_norm = np.zeros(tmp.shape)
            n_norm[tmp != 0] = 1 / tmp[tmp != 0]
            
            #n_norm = 1 / np.dot(n, n)
             
            sub_face_params.append([u, v, n, n_norm])
         
        ## W. Heidrich, Journal of Graphics, GPU, and Game Tools,Volume 10, Issue 3, 2005
        errors = np.ones((avg_vert.shape[0])) * np.iinfo(np.int32).max
        match_idx = np.ones((avg_vert.shape[0]), dtype = int) * -1; 
        weights = np.zeros((avg_vert.shape[0], avg_vert.shape[1])); 
        for (avg_vert_idx, avg_vortex) in enumerate(avg_vert):
            if (avg_vert_idx % 10000 == 0):
                print(avg_vert_idx)
            for sub_face_idx in avg_neigh_faces[avg_vert_idx]:
                sub_face = sub_faces[sub_face_idx]
                 
                w = avg_vortex - sub_vert[sub_face[0], :]
                gamma = np.dot(np.cross(sub_face_params[sub_face_idx][0], w), sub_face_params[sub_face_idx][2]) * sub_face_params[sub_face_idx][3]
                beta = np.dot(np.cross(w, sub_face_params[sub_face_idx][1]), sub_face_params[sub_face_idx][2]) * sub_face_params[sub_face_idx][3]
                alpha = 1 - gamma - beta
                 
                alpha_err = (alpha < 0) * np.abs(alpha) + (alpha > 1) * (alpha - 1)
                beta_err = (beta < 0) * np.abs(beta) + (beta > 1) * (beta - 1)
                gamma_err = (gamma < 0) * np.abs(gamma) + (gamma > 1) * (gamma - 1)
                 
                curr_error = alpha_err + beta_err + gamma_err
                if (curr_error > errors[avg_vert_idx]):
                    continue
                 
                errors[avg_vert_idx] = curr_error
                weights[avg_vert_idx][0] = alpha; weights[avg_vert_idx][1] = beta; weights[avg_vert_idx][2] = gamma
                match_idx[avg_vert_idx] = sub_face_idx
     
        # Fix outside points
        # https://math.stackexchange.com/questions/588871/minimum-distance-between-point-and-face
        for bad_weights_idx in np.argwhere(errors != 0).squeeze(1):
         
            if (bad_weights_idx != 530):
                continue
             
            bad_weights_idx = 202
             
            sub_face_idx = match_idx[bad_weights_idx]
            weight = weights[bad_weights_idx]
            face_vortex = sub_vert[sub_faces[sub_face_idx], :]
            avg_vortex = avg_vert[bad_weights_idx]
             
            is_edge = False
            if (weight[0] >= 0 and weight[1] >= 0 and weight[2] < 0):
                p0 = face_vortex[0, :]
                p1 = face_vortex[1, :]
                is_edge = True
            if (weight[0] >= 0 and weight[1] < 0 and weight[2] >= 0):
                p0 = face_vortex[0, :]
                p1 = face_vortex[2, :]
                is_edge = True
            if (weight[0] < 0 and weight[1] >= 0 and weight[2] >= 0):
                p0 = face_vortex[1, :]
                p1 = face_vortex[2, :]
                is_edge = True
            if (weight[0] >= 0 and weight[1] < 0 and weight[2] < 0):
                alpha = 1; beta = 0; gamma = 0
            if (weight[0] < 0 and weight[1] >= 0 and weight[2] < 0):
                alpha = 0; beta = 1; gamma = 0
            if (weight[0] < 0 and weight[1] < 0 and weight[2] >= 0):
                alpha = 0; beta = 0; gamma = 1
             
            if (is_edge):
                d_vec = (p1 - p0)/np.linalg.norm(p1 - p0)
                tmp_pnt = np.dot(d_vec, avg_vortex - p0) * d_vec + p0
                 
                sub_face = sub_faces[sub_face_idx]
                w = tmp_pnt - sub_vert[sub_face[0], :]
                gamma = np.dot(np.cross(sub_face_params[sub_face_idx][0], w), sub_face_params[sub_face_idx][2]) * sub_face_params[sub_face_idx][3]
                beta = np.dot(np.cross(w, sub_face_params[sub_face_idx][1]), sub_face_params[sub_face_idx][2]) * sub_face_params[sub_face_idx][3]
                alpha = 1 - gamma - beta
             
            weights[bad_weights_idx][0] = alpha; weights[bad_weights_idx][1] = beta; weights[bad_weights_idx][2] = gamma
        
        if (os.path.exists(subj_path + "proj/") == False):
            os.mkdir(subj_path + "proj/")
        
        np.save(subj_path + "proj/" + "mri_subj_to_mri_fs_avg_" + "weights_" + hemisphere + ".npy", weights)
        np.save(subj_path + "proj/" + "mri_subj_to_mri_fs_avg_" + "match_idx_" + hemisphere + ".npy", match_idx)
    else:
        weights = np.load(subj_path + "proj/" + "mri_subj_to_mri_fs_avg_" + "weights_" + hemisphere + ".npy")
        match_idx = np.load(subj_path + "proj/" + "mri_subj_to_mri_fs_avg_" + "match_idx_" + hemisphere + ".npy")
     
    face_idx = sub_faces[match_idx, :]
    row_ind = np.repeat(np.arange(len(avg_vert)), 3)
    mri_map = scipy.sparse.csr_matrix((weights.ravel(), (row_ind, face_idx.ravel())),
                                  shape=(len(avg_vert), len(sub_vert)))
    
    return (sub_vert, sub_faces, avg_vert, mri_map)

def calc_model_orientation_fs_avg_single_hem(valid_vert, model_vert, subj_path, fs_avg_path, hemisphere, overwrite):
    ###
    # Move src space data into fs_average space
    # Calculate transformation
    (sub_vert, sub_faces, avg_vert, mri_map) = calc_mri_maps(subj_path, fs_avg_path, hemisphere, overwrite)
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
    
    # Select the used vertices and connected ones
    # Start with the vertices used in valid_vert
    # Iteratively add the connected ones until no more ones are added
    vert_idx = np.where(valid_vert)[0]
    trans = scipy.sparse.eye(int(np.sum(valid_vert)), format='csr')
    mult = np.zeros(sub_vert.shape[0])
    recompute = True
    max_iter_cnt = 100
    for iter_idx in range(max_iter_cnt):
        if (iter_idx != 0):
            trans = trans[vert_idx, :]
        used_vert = adj_mat[:, vert_idx] if len(vert_idx) < sub_vert.shape[0] else adj_mat
        trans = used_vert * trans
        
        if (recompute):
            if (len(vert_idx) == sub_vert.shape[0]):
                row_sum = np.asarray(adj_mat.sum(-1))[:, 0]
                recompute = False
            else:
                mult[vert_idx] = 1
                row_sum = adj_mat * mult
                vert_idx = np.where(row_sum)[0]
    
        trans.data /= np.where(row_sum, row_sum, 1).repeat(np.diff(trans.indptr))
        if (len(vert_idx) == sub_vert.shape[0]):
            break
    
    return (avg_vert, mri_map, trans)

def get_mri_subj_to_fs_avg_trans_mat(valid_lh_vert, valid_rh_vert,
                                     model_vert, subj_path, fs_avg_path, overwrite):
    
    (avg_lh_vert, lh_mir_map, lh_trans) = calc_model_orientation_fs_avg_single_hem(valid_lh_vert, model_vert, subj_path, fs_avg_path, "lh", overwrite)
    (avg_rh_vert, rh_mir_map, rh_trans) = calc_model_orientation_fs_avg_single_hem(valid_rh_vert, model_vert, subj_path, fs_avg_path, "rh", overwrite)
    
    valid_avg_lh_vert = find_valid_vertices(avg_lh_vert, model_vert)
    valid_avg_rh_vert = find_valid_vertices(avg_rh_vert, model_vert)
    lh_trans = lh_mir_map[np.where(valid_avg_lh_vert)[0]] * lh_trans
    rh_trans = rh_mir_map[np.where(valid_avg_rh_vert)[0]] * rh_trans
    
    trans_mat = scipy.sparse.lil_matrix((lh_trans.shape[0] + rh_trans.shape[0], lh_trans.shape[1] + rh_trans.shape[1]))
    
    trans_mat[:lh_trans.shape[0], :lh_trans.shape[1]] += lh_trans
    trans_mat[lh_trans.shape[0]:, lh_trans.shape[1]:] += rh_trans
    trans_mat = trans_mat.tocsr()
    
    return (trans_mat, valid_avg_lh_vert, valid_avg_rh_vert)
    
def apply_mri_subj_to_fs_avg_trans_mat(trans_mat, data):
    return trans_mat * data

def calc_whitener(eigen_val, eigen_vec):
    eigen_val_white = np.copy(eigen_val)
    eigen_val_white[eigen_val_white < 0] = 0 #Cannot be negative in a cov-matrix; may happen due to numerical innaccuracy
    eigen_val_white[eigen_val_white > 0] = 1/np.sqrt(eigen_val_white[eigen_val_white > 0])
    whitener = np.expand_dims(eigen_val_white, axis = 1) * eigen_vec
    
    return whitener

