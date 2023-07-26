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
    """
    Transforms an mne-fiducials object into an dictionary containing the fiducials.
    
    :param pre_mri_ref_pts: The mne-fiducials object in question.
    
    :return: Dictionary containing the fiducial points.
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

def run_subprocess_in_custom_working_directory(subject_id, cmd):
    """
    Creates a custom work directory to run a freesurfer command within. 
    
    :param subject_id: Name of the subject whose data is worked on.
    :param cmd: The freesurfer command to be executed in the custom environment. 
    """
    path_to_tmp_cwd = "finnpy_" + subject_id + "_freesurfer_tmp_dir/" # A temporary working directory is needed as freesurfer saves intermediate results in files.
        
    if (os.path.exists(path_to_tmp_cwd + ".lock")): # Checks if the current directory is already worked in, if yes, raise error.
        raise AssertionError("Subject is already being worked on as %s already exists" % (path_to_tmp_cwd + ".lock",))
    
    os.makedirs(path_to_tmp_cwd, exist_ok = True)
    file = open(path_to_tmp_cwd + ".lock", "wb")
    file.close()
    
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, cwd = path_to_tmp_cwd, env = os.environ.copy())
    for c in iter(lambda: process.stderr.read(1), b""):
        print(c.decode(), end = "")
    
    shutil.rmtree(path_to_tmp_cwd) # And removed alter on

def init_fs_paths(subjects_path):
    """
    Runs some freesurfer initialization steps.
    
    :param subjects_path: Path to the subject folder.
    """
    os.environ["FREESURFER_HOME"]   = "/usr/local/freesurfer/7.2.0/"
    os.environ["FSFAST_HOME"]       = "/usr/local/freesurfer/7.2.0/fsfast/"
    os.environ["FSF_OUTPUT_FORMAT"] = "nii.gz"
    os.environ["SUBJECTS_DIR"]      = subjects_path[:-1] if (subjects_path[-1] == "/") else subjects_path
    os.environ["MNI_DIR"]           = "/usr/local/freesurfer/7.2.0/mni/"
    
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FREESURFER_HOME']+"bin/"
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FSFAST_HOME']+"bin/"

def fast_cross_product(a, b):
    """
    Calculates the cross product between two vectors. Seemingly numpy has too much overhead here to be fast.
    
    :param a: The 1st vector in the cross product.
    :param b: The 2nd vector in the cross product.
    
    :return: Crossproduct of vectors a x b.
    """
    
    res = np.empty(a.shape)
    res[:, 0] = a[:, 1]*b[:, 2]-a[:, 2]*b[:, 1]
    res[:, 1] = a[:, 2]*b[:, 0]-a[:, 0]*b[:, 2]
    res[:, 2] = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    
    return res

def norm_vert(vert):
    """
    Normalizes a vector. Seemingly numpy has too much overhead here to be fast.
    
    :param vert: The to be normalized vector.
    
    :return: The normalized vector.
    """
    size = np.sqrt(vert[:, 0]*vert[:, 0]+vert[:, 1]*vert[:, 1]+vert[:, 2]*vert[:, 2])
    vert[size > 0] /= size[size > 0, np.newaxis]
    return vert

def find_nearest_neighbor(src_model, tgt_pt, method = "kdtree"):
    """
    Employs two methods to find the nearest neighbors. In case a src_model a list of points is provided, a model is trained, otherwise, pretrained models are used if a (KDTree or BallTree objects) are provided.
    
    :param src_model: Either a list of data points or a kdtree/balltree object.
    :param tgt_pt: The tgt_point to be queried.
    
    :return: The nearest neighbor of the respective point.
    """
    
    if (type(src_model) == sklearn.neighbors.KDTree or type(src_model) == sklearn.neighbors.BallTree):
        tree = src_model
        if (method == "kdtree"):
            neigh_indices = tree.query(tgt_pt, k = 1)[1].squeeze(1)
        elif(method == "balltree"):
            neigh_indices = tree.query(tgt_pt, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")
    else:
        if (method == "kdtree"):
            tree = sklearn.neighbors.KDTree(src_model)
            neigh_indices = tree.query(tgt_pt, k = 1)[1].squeeze(1)
        elif(method == "balltree"):
            tree = sklearn.neighbors.BallTree(src_model)
            neigh_indices = tree.query(tgt_pt, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")

    return (neigh_indices, tree)

def find_valid_vertices(mri_vert, model_vert, max_neighborhood_size = 5):
    """
    Matches freesurfer reconstructed mri vertices (sphere) with model vertices (octahedron).
    
    :param mri_vert: Freesurfer based vertices (sphere).
    :param model vert: Octahedron based vertices.
    
    :return: Binary list of Freesurfer vertices that have a match in the model vertices (octahedron).
    """    
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
    """
    Calculates the magnitude of a vector
    
    :param vec: The vector in question.
    
    :return: The magnitude of vec.
    """
    
    return np.sqrt(vec[:, 0]*vec[:, 0]+vec[:, 1]*vec[:, 1]+vec[:, 2]*vec[:, 2])

def apply_inv_transformation(data, trans):
    """
    Applies the inverse of trans to data.
    
    :param data: The data to be transformed.
    :param trans: The transformation matrix (4x4 matrix; rotation + translation only).
    
    :return: The transformed data.    
    """
    inv_trans = np.linalg.inv(trans)
    tmp = np.dot(inv_trans[:3, :3], data.T).T
    return tmp + inv_trans[:3, 3]    

def calc_quat_angle(a, b):
    """
    Calculate the angle between two quaternions.
    
    :param a: The 1st quaternion.
    :param b: The 2nd quaternion.
    
    :return: The angle between a and b.
    """
    w = a[3]*b[3] + a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    x = a[3]*b[0] - a[0]*b[3] - a[1]*b[2] + a[2]*b[1]
    y = a[3]*b[1] - a[1]*b[3] - a[0]*b[2] + a[2]*b[0]
    z = a[3]*b[2] - a[2]*b[3] - a[0]*b[1] + a[1]*b[0]
    
    return 2 * np.arctan2(np.linalg.norm(np.asarray((x, y, z))), np.abs(w))

def orient_mat_to_block_format(orient_mat):
    """
    Transforms an orientation matrix (rotation matrix) into (sparse) block format.
    
    :param orient_mat: The orientation matrix in question.
    
    :return: The transformed orientation matrix.
    """
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

def get_orientation(vortex_normals, valid_vert, cluster_grp, cluster_indices,
                    mri_to_head_trans, double_precision = 40):
    """
    Calculates an orthonormal basis of eigenvector/values for each supporting/valid point.
    
    :param vortex_normals: Normals of the supporting vertices.
    :param valid_vert: List of valid/supporting vertices.
    :param cluster_grp: Clusters represented by a single vortex.
    :param cluster_indices: Cluster indices.
    :param mri_to_head_trans: Transformation from MRI to head coordinates.
    :param double_precision: Numerical precision of the eigenvectors/values (default: 40 digits).
    
    :return: Orthonormal eigenbasis.
    """
    
    #Transforms normals into head space
    rot_vortex_normals = np.dot(vortex_normals, mri_to_head_trans[:3, :3].T)
        
    #Accumulates normals from clusters, effectively interpolating/steering/weighing them into an more accurate depiction
    weighed_normals = np.zeros((np.where(valid_vert)[0].shape[0], 3))
    for (vertex_idx, vertex_id) in enumerate(np.searchsorted(np.where(valid_vert)[0], np.where(valid_vert)[0])): 
        weighed_normals[vertex_idx] = np.sum(rot_vortex_normals[cluster_grp[cluster_indices[vertex_id]], :], axis = 0)
    weighed_normals /= np.linalg.norm(weighed_normals, axis = 1, keepdims = True)
    
    #Calculate an orthonormal eigenvector basis for each accumulated/weighted normals, resulting in (normals x 3) eigenvectors/values.
    pre_ev = np.empty((weighed_normals.shape[0], 3, 3))
    for idx in range(pre_ev.shape[0]):
        pre_ev[idx, :, :] = np.matmul(weighed_normals[[idx], :].T, weighed_normals[[idx], :])

    #Per normal basis - eigenvector/value calculation
    evec = list()
    for mat_idx in range(pre_ev.shape[0]):
        if ((pre_ev[mat_idx, :, :] == pre_ev[mat_idx, :, :].T).all() == False):
            raise AssertionError("Something went wrong with the orientation matrix")
        
        mat = mpmath.matrix(np.eye(3) - pre_ev[mat_idx, :, :])
        mat.ctx.dps = double_precision
        (_, loc_evec) = mpmath.eigsy(mat)
         
        evec.append(loc_evec.tolist())
    
    #Concatenation
    evec = np.asarray(evec, dtype = float)
    evec = evec[:, :, ::-1]
    direction = np.sign(np.matmul(np.expand_dims(weighed_normals, axis = 1), evec[:, :, -1:]))
    direction[direction == 0] = 1
    evec *= direction
    evec = evec.swapaxes(1, 2)
    evec = evec.reshape(-1, 3)
     
    return evec

def calc_acc_hem_normals(geom_white_vert, geom_white_faces):
    """
    Generatoes vertex normals from accumulated face normals. See https://en.wikipedia.org/wiki/Vertex_normal.
    
    :param geom_white_vert: Array of vertices. 
    :param geom_white_faces: Array of faces. 
    
    :return: Array of surface normals for the input faces/vertices.
    """
    
    #Calculate surface normals
    vert0 = geom_white_vert[geom_white_faces[:, 0], :]
    vert1 = geom_white_vert[geom_white_faces[:, 1], :]
    vert2 = geom_white_vert[geom_white_faces[:, 2], :]
    vortex_normals = fast_cross_product(vert1 - vert0, vert2 - vert1)
    len_vortex_normals = np.linalg.norm(vortex_normals, axis = 1)
    vortex_normals[len_vortex_normals > 0] /= np.expand_dims(len_vortex_normals[len_vortex_normals > 0], axis = 1)
    
    #Accumulate face normals
    acc_normals = np.zeros((geom_white_vert.shape[0], 3))
    for outer_idx in range(3):
        for inner_idx in range(3):
            value = np.zeros((geom_white_vert.shape[0],))
            for vertex_idx in range(geom_white_faces.shape[0]):
                value[geom_white_faces[vertex_idx, outer_idx]] += vortex_normals[vertex_idx, inner_idx]
            acc_normals[:, inner_idx] += value
    
    #Normalize face normals
    len_acc_normals = np.linalg.norm(acc_normals, axis = 1)
    acc_normals[len_acc_normals > 0] /= np.expand_dims(len_acc_normals[len_acc_normals > 0], axis = 1)
    
    return acc_normals

def find_vertex_clusters(geom_white_vert, valid_geom_vert, geom_white_faces):
    """
    Identifies which input vertices (geom_white_vert) are presented by which valid vortex.
    
    :param geom_white_vert: To be clustered vertices.
    :param valid_geom_vert: Binary list of valid vertices.
    :param geom_white_faces: Corresponding faces.
    
    :return: (cluster_grp, cluster_indices) - Information on clusters.
    """
    
    #Get adjacency matrix to identify nearest valid vortex
    edges = scipy.sparse.coo_matrix((np.ones(3 * geom_white_faces.shape[0]),
                                        (np.concatenate((geom_white_faces[:, 0], geom_white_faces[:, 1], geom_white_faces[:, 2])),
                                         np.concatenate((geom_white_faces[:, 1], geom_white_faces[:, 2], geom_white_faces[:, 0])))),
                                        shape = (geom_white_vert.shape[0], geom_white_vert.shape[0]))
    edges = edges.tocsr()
    edges += edges.T
    edges = edges.tocoo()
    edges_dists = np.linalg.norm(geom_white_vert[edges.row, :] - geom_white_vert[edges.col, :], axis = 1)
    edges_adjacency = scipy.sparse.csr_matrix((edges_dists, (edges.row, edges.col)), shape = edges.shape)
    _, _, min_idx = scipy.sparse.csgraph.dijkstra(edges_adjacency, indices = np.where(valid_geom_vert)[0], min_only = True, return_predecessors = True)
    
    #Accumulates the clusters
    sort_near_idx = np.argsort(min_idx)
    sort_min_idx = min_idx[sort_near_idx]
    breaks = np.where(sort_min_idx[1:] != sort_min_idx[:-1])[0] + 1
    starts = [0] + breaks.tolist()
    ends = breaks.tolist() + [len(min_idx)]
    cluster_grp = list()
    for cluster_idx in range(len(starts)):
        cluster_grp.append(np.sort(sort_near_idx[starts[cluster_idx]:ends[cluster_idx]]))
    pre_cluster_indices = sort_min_idx[breaks - 1]
    cluster_indices = np.searchsorted(pre_cluster_indices, np.where(valid_geom_vert)[0])
    
    return (cluster_grp, cluster_indices)

def cortical_surface_reorient_fwd_model(lh_geom_white_vert, lh_geom_white_faces, valid_geom_lh_vert,
                                        rh_geom_white_vert, rh_geom_white_faces, valid_geom_rh_vert,
                                        fwd_sol, mri_to_head_trans):
    """
    Transforms a fwd model into surface orientation (orthogonal to the respective surface cluster; allowing for cortically constrained inverse modeling) and shrinks it by making closeby channels project to the same destination. This is different from a 'default' 3D transformation. 
    
    :param lh_geom_white_vert: Lh vertices.
    :param lh_geom_white_faces: Lh faces.
    :param valid_geom_lh_vert: Valid/supporting lh vertices.
    :param rh_geom_white_vert: Rh vertices.
    :param rh_geom_white_faces: Rh faces.
    :param valid_geom_rh_vert: Valid/supporting rh vertices.
    :param fwd_sol: Previous forward solution.
    :param mri_to_head_trans: MRI to head transformation matrix.
    
    :return: Transformed surface model.
    """
     
    #Computes vertex normals
    lh_acc_normals = calc_acc_hem_normals(lh_geom_white_vert, lh_geom_white_faces)
    rh_acc_normals = calc_acc_hem_normals(rh_geom_white_vert, rh_geom_white_faces)
     
    # Figures out which real vertices are represented by the same model vortex
    (lh_cluster_grp, lh_cluster_indices) = find_vertex_clusters(lh_geom_white_vert, valid_geom_lh_vert, lh_geom_white_faces)
    (rh_cluster_grp, rh_cluster_indices) = find_vertex_clusters(rh_geom_white_vert, valid_geom_rh_vert, rh_geom_white_faces)
    
    #Determines the orientation 
    lh_orient = get_orientation(lh_acc_normals, valid_geom_lh_vert, lh_cluster_grp, lh_cluster_indices, mri_to_head_trans)
    rh_orient = get_orientation(rh_acc_normals, valid_geom_rh_vert, rh_cluster_grp, rh_cluster_indices, mri_to_head_trans)
    
    #Concatenates the eigenvector bases 
    orient = np.concatenate((lh_orient, rh_orient), axis = 0)
     
    #Transforms eigenvector matrix into block matrix for ease of use
    rot = orient_mat_to_block_format(orient[2::3, :])
    
    #Applies rotation matrix to the fwd solution 
    return fwd_sol * rot
    
def calc_mri_maps(subj_path, fs_avg_path, hemisphere, overwrite):
    """
    Find the subject space points corresponding to fs avg space points.
    
    :param subj_path: Path to subject freesurfer files.
    :param fs_avg_path: Path fo fs average freesurfer files.
    :param hemisphere: hemisphere to compute for.
    :param overwrite: Flag whether to overwrite preexisting mri maps.
    
    :return: 
    """
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
        #Prepare faces for the following loop
        sub_face_params = list()
        for sub_face in sub_faces:
            u = sub_vert[sub_face[1], :] - sub_vert[sub_face[0], :]
            v = sub_vert[sub_face[2], :] - sub_vert[sub_face[0], :]
            n = np.cross(u, v)
            
            tmp = np.dot(n, n)
            n_norm = np.zeros(tmp.shape)
            n_norm[tmp != 0] = 1 / tmp[tmp != 0]
            
            sub_face_params.append([u, v, n, n_norm])
         
        #Check of points are within a triangle and if yes, save location
        #For reference, see https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
        #and W. Heidrich, Journal of Graphics, GPU, and Game Tools,Volume 10, Issue 3, 2005
        errors = np.ones((avg_vert.shape[0])) * np.iinfo(np.int32).max
        match_idx = np.ones((avg_vert.shape[0]), dtype = int) * -1; 
        weights = np.zeros((avg_vert.shape[0], avg_vert.shape[1])); 
        for (avg_vert_idx, avg_vortex) in enumerate(avg_vert):
            #===================================================================
            # if (avg_vert_idx % 10000 == 0):
            #     print(avg_vert_idx)
            #===================================================================
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
     
        #For those points "outside" of faces, find the most close nearby point.
        #For reference, https://math.stackexchange.com/questions/588871/minimum-distance-between-point-and-face
        for bad_weights_idx in np.argwhere(errors != 0).squeeze(1):
            
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
    
def calc_small_to_default_vertices_proj(valid_vert, sub_vert, sub_faces):
    """
    Identifies 
    """
    #Create an adjacency graph for use below
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
    
    #Starting from the minimal vertex set used in subject space (e.g. 4098), 
    #expands to include all vertices in subject space.
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

def get_mri_subj_to_fs_avg_trans_mat(valid_lh_vert, valid_rh_vert,
                                     model_vert, subj_path, fs_avg_path, overwrite):
    """
    Create a subject to fs average transformation matrix.
    
    :param valid_lh_vert: Valid lh vertices (subject space).
    :param valid_rh_vert: Valid rh vertices (subject space).
    :param model_vert: Model vertices.
    :param subj_path: Path to subject's freesurfer files.
    :param fs_avg_path: Path to fs average's freesurfer files.
    :param overwrite: Flag whether to overwrite MRI maps.
    
    :return: (trans_mat, valid_avg_lh_vert, valid_avg_rh_vert) - Transformation matrix, valid/supporting vertices for left & right hemisphere 
    """
    
    #Find a transformation for all points
    (lh_sub_vert, lh_sub_faces, avg_lh_vert, lh_mri_map) = calc_mri_maps(subj_path, fs_avg_path, "lh", overwrite)
    (rh_sub_vert, rh_sub_faces, avg_rh_vert, rh_mri_map) = calc_mri_maps(subj_path, fs_avg_path, "rh", overwrite)
    
    #Calculate a projection from valid/supporting points to all points (subject space only)
    lh_proj = calc_small_to_default_vertices_proj(valid_lh_vert, lh_sub_vert, lh_sub_faces)
    rh_proj = calc_small_to_default_vertices_proj(valid_rh_vert, rh_sub_vert, rh_sub_faces)
    
    #Combine to derive transformation between subject space and fs average space
    valid_avg_lh_vert = find_valid_vertices(avg_lh_vert, model_vert)
    valid_avg_rh_vert = find_valid_vertices(avg_rh_vert, model_vert)
    lh_proj = lh_mri_map[np.where(valid_avg_lh_vert)[0]] * lh_proj
    rh_proj = rh_mri_map[np.where(valid_avg_rh_vert)[0]] * rh_proj
    
    #Reformat
    trans_mat = scipy.sparse.lil_matrix((lh_trans.shape[0] + rh_trans.shape[0], lh_trans.shape[1] + rh_trans.shape[1]))
    trans_mat[:lh_trans.shape[0], :lh_trans.shape[1]] += lh_trans
    trans_mat[lh_trans.shape[0]:, lh_trans.shape[1]:] += rh_trans
    trans_mat = trans_mat.tocsr()
    
    return (trans_mat, valid_avg_lh_vert, valid_avg_rh_vert)
    
def apply_mri_subj_to_fs_avg_trans_mat(trans_mat, data):
    """
    Transforms data from subject space to fs_average space
    
    :param trans_mat: Transformation from subject to fs average space.
    :param data: Data in subject space.
    
    :return: Data in fs avg space. 
    """
    return trans_mat * data

def calc_whitener(eigen_val, eigen_vec):
    """
    Calculate PCA based whitener from provided eigenvalues and eigenvectors.
    
    :param eigen_val: Eigenvalues.
    :param eigen_vec: Eigenvectors.
    
    :return: Whitener.
    """
    eigen_val_white = np.copy(eigen_val)
    eigen_val_white[eigen_val_white < 0] = 0 #Cannot be negative in a cov-matrix; may happen due to numerical innaccuracy
    eigen_val_white[eigen_val_white > 0] = 1/np.sqrt(eigen_val_white[eigen_val_white > 0])
    whitener = np.expand_dims(eigen_val_white, axis = 1) * eigen_vec
    
    return whitener

