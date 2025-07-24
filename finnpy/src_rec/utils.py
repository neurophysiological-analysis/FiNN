"""
Created on Oct 13, 2022.

@author: voodoocode
"""

import numpy as np
import scipy.sparse
import mpmath
import os
import sklearn.neighbors
import ctypes
import mne
import csv

def read_eeg_pts(system = "1020"):
    """
    Read EEG channel names and locations from file.
    
    Parameters
    ----------
    system : string
             "1020" or 1005".
             
    Returns
    -------
    sen_ref_pts : dict()
                  chs: list(np.ndarray(3,)
                       List of channel locations.
                  labels: list(string)
                          List of channel names.
    
    Raises
    ------
    NotImplementedError
        EEG setup has to be either '1020' or '1005'.
    """
    if (system == "1020"):
        ref_file_path = __file__[:__file__.rindex("/")] + "/../res/1020_system.csv"
    elif (system == "1005"):
        ref_file_path = __file__[:__file__.rindex("/")] + "/../res/1005_system.csv"
    else:
        raise NotImplementedError("Unknown setup. Setup has to be either 1020 or 1005.")
    ref_file = open(ref_file_path, "r")
    csv_rdr = csv.reader(ref_file)
    pre_sen_ref_pts = [row for row in csv_rdr]
    ref_file.close()

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
            sen_ref_pts["chs"].append(np.asarray(pre_sen_ref_pt[1:], dtype = float) / 1000)
            sen_ref_pts["labels"].append(pre_sen_ref_pt[0])
        sen_ref_pts[label] = np.asarray(pre_sen_ref_pt[1:], dtype = float) / 1000
    return sen_ref_pts

def get_meg_bio_channels(file_path, mask = None):
    """
    Identify bio channel types.
    
    Parameters
    ----------
    file_path : mne.io.read_raw_fif
                Scanned MRI file.
    mask : np.ndarray
           list of valid channels.
               
    Returns
    -------
    result : tuple of (np.ndarray, int, list, list)
             - valid_ch_indices : numpy.ndarray, shape(ch_cnt,)
                                  Binary list identifying channels as valid/invalid.
             - meg_ch_indices : list, int
                                Indices of magnetometer channels.
             - grad_ch_indices : list, int
                                 Indices of gradiometer channels.
             - ch_names : list, string
                          channel names.
    """
    raw_file = mne.io.read_raw_fif(file_path, preload = True, verbose = "ERROR")
    
    valid_ch_indices = np.zeros((len(raw_file.info["chs"]), ), dtype = bool)
    meg_ch_indices = list()
    grad_ch_indices = list()
    ch_names = list()
    
    channel_types = np.asarray(raw_file.info["chs"])
    if (mask is not None):
        valid_ch_indices = valid_ch_indices[mask]
        channel_types = channel_types[mask]
    
    for ch_idx in range(len(channel_types)):
        loc_kind = channel_types[ch_idx]["kind"]
        if (loc_kind in [mne.io.constants.FIFF.FIFFV_MEG_CH, mne.io.constants.FIFF.FIFFV_EEG_CH, 
                         mne.io.constants.FIFF.FIFFV_SEEG_CH, mne.io.constants.FIFF.FIFFV_ECOG_CH, 
                         mne.io.constants.FIFF.FIFFV_FNIRS_CH, mne.io.constants.FIFF.FIFFV_DBS_CH]):
            valid_ch_indices[ch_idx] = True
            ch_names.append(raw_file.ch_names[ch_idx])
        if (raw_file.info["chs"][ch_idx]["coil_type"] in [mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T1,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T2,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T3,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T4,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_W]):
            meg_ch_indices.append(ch_idx)
        if (raw_file.info["chs"][ch_idx]["coil_type"] in [mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T1,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T2,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T3,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_T4,
                                                          mne.io.constants.FIFF.FIFFV_COIL_VV_PLANAR_W]):
            grad_ch_indices.append(ch_idx)
    
    data = raw_file.get_data()
    return (data, valid_ch_indices, meg_ch_indices, grad_ch_indices, ch_names)

def fast_3D_cross_product_single(a, b):
    """
    Calculate the cross product between two 3D vectors.
    
    Parameters
    ----------
    a : numpy.ndarray, shape(3,)
        The 1st vector in the cross product.
    b : numpy.ndarray(3,)
        The 2nd vector in the cross product.
               
    Returns
    -------
    res : numpy.ndarray, shape(3,)
          Crossproduct of vectors a x b.
    """
    res = np.empty(a.shape)
    res[0] = a[1] * b[2] - a[2] * b[1]
    res[1] = a[2] * b[0] - a[0] * b[2]
    res[2] = a[0] * b[1] - a[1] * b[0]
    
    return res

def fast_3D_cross_product_multi(a, b):
    """
    Calculate the cross product between two groups of 3D vectors.
    
    Parameters
    ----------
    a : numpy.ndarray, shape(n, 3)
        The 1st vectors in the cross product.
    b : numpy.ndarray(n, 3)
        The 2nd vectors in the cross product.
               
    Returns
    -------
    res : numpy.ndarray, shape(n, 3)
          Crossproduct of a x b.
    """
    res = np.empty(a.shape)
    res[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    res[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    res[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    
    return res

def fast_3D_dot_product_single(a, b):
    """
    Calculate the dot product between two 3D vectors.
    
    Parameters
    ----------
    a : numpy.ndarray, shape(3,)
        The 1st vector in the cross product.
    b : numpy.ndarray(3,)
        The 2nd vector in the cross product.
               
    Returns
    -------
    res : float
          Dotproduct of vectors a . b.
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def fast_3D_normalize_vec_multi(vec):
    """
    Normalize multiple 3D vectors.
    
    Parameters
    ----------
    vec : numpy.ndarray, shape(n, 3)
          The to be normalized vectors.
               
    Returns
    -------
    vec : numpy.ndarray, shape(n, 3)
          The normalized vectors.
    """
    size = np.sqrt(vec[:, 0] * vec[:, 0] + vec[:, 1] * vec[:, 1] + vec[:, 2] * vec[:, 2])
    vec[size > 0] /= size[size > 0, np.newaxis]
    return vec

def fast_3D_norm_vec_multi(vec):
    """
    Calculate the euclidean norm/magnitude of multiple 3D vectors across the 2nd axis.
    
    Parameters
    ----------
    vec : numpy.ndarray, shape(n, 3)
          The vector in question.
               
    Returns
    -------
    magn : float
           Euclidean norm/magnitude of the 3D vectors.
    """
    return np.sqrt(vec[:, 0] * vec[:, 0] + vec[:, 1] * vec[:, 1] + vec[:, 2] * vec[:, 2])

def fast_3D_norm_vec_single(vec):
    """
    Calculate the euclidean norm/magnitude of a 3D vector.
    
    Parameters
    ----------
    vec : numpy.ndarray, shape(n,)
          The vector in question.
               
    Returns
    -------
    magn : float
           Euclidean norm/magnitude of the vector.
    """
    return np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

def fast_3D_sum_multi(vec):
    """
    Fast way to sum up multiple 3D vectors across the 2nd axis.
    
    Parameters
    ----------
    vec : np.ndarray, shape(m, n)
          Vectors to be summed up.
               
    Returns
    -------
    sum : np.ndarray, shape(m,)
          Vector with sum of elements calculated along the 2nd axis.
    """
    return vec.dot(np.ones(vec.shape[1]))

def apply_inv_transformation(data, trans):
    """
    Apply the inverse of trans to data.
    
    Parameters
    ----------
    data : np.ndarray, shape(n, 3)
           The data to be transformed.
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
               
    Returns
    -------
    trans_data : np.ndarray, shape(n, 3)
                 Transformed data.
    """
    inv_trans = np.linalg.inv(trans)
    tmp = np.dot(inv_trans[:3, :3], data.T).T
    return tmp + inv_trans[:3, 3]    

def calc_quat_angle(a, b):
    """
    Calculate the angle between two quaternions.
    
    Parameters
    ----------
    a : numpy.ndarray, shape(4,)
        The 1st quaternion.
    b : numpy.ndarray, shape(4,)
        The 2nd quaternion.
               
    Returns
    -------
    angle : float
            The angle between a and b. (scale: radians)
    """
    w = a[3] * b[3] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    x = a[3] * b[0] - a[0] * b[3] - a[1] * b[2] + a[2] * b[1]
    y = a[3] * b[1] - a[1] * b[3] - a[0] * b[2] + a[2] * b[0]
    z = a[3] * b[2] - a[2] * b[3] - a[0] * b[1] + a[1] * b[0]
    
    return 2 * np.arctan2(np.linalg.norm(np.asarray((x, y, z))), np.abs(w))

def orient_mat_to_block_format(orient_mat):
    """
    Transform an orientation matrix (rotation matrix) into (sparse) block format.
    
    Parameters
    ----------
    orient_mat : numpy.ndarray, shape(valid_vtx_cnt, 3)
                 The non-block matrix formatted orientation matrix.
               
    Returns
    -------
    rot : numpy.ndarray, shape(valid_vtx_cnt * 3, valid_vtx_cnt)
          The block matrix formatted orientation matrix.
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

def get_eigenbasis(vortex_normals, valid_vert, cluster_grp, cluster_indices,
                   mri_to_meeg_trans, double_precision = 40):
    """
    Calculate an orthonormal basis of eigenvector/values for each supporting/valid point.
    
    Parameters
    ----------
    vortex_normals : numpy.ndarray, shape(vtx_cnt, 3)
                     Normals of the supporting vertices.
    valid_vert : numpy.ndarray, shape(vtx_cnt,)
                 List of valid/supporting vertices.
    cluster_grp : list, len(n,)
                  Clusters represented by a single vortex.
    cluster_indices : list, len(n,)
                      Cluster indices.
    mri_to_meeg_trans : numpy.ndarray, shape(4, 4)
                        Transformation from MRI to head coordinates.
    double_precision : double
                       Numerical precision of the eigenvectors/values, 
                       defaults to 40 digits.
               
    Returns
    -------
    evec : numpy.ndarray, shape(valid_vtx_cnt * 3, 3)
           Orthonormal eigenbasis.
           
    Raises
    ------
    AssertionError
        A diagonal matrix is expected for the nomal basis calculation.
    """
    # Transforms normals into head space
    rot_vortex_normals = np.dot(vortex_normals, mri_to_meeg_trans[:3, :3].T)
        
    # Accumulates normals from clusters, effectively interpolating/steering/weighing them into an more accurate depiction
    weighed_normals = np.zeros((np.where(valid_vert)[0].shape[0], 3))
    for (vertex_idx, vertex_id) in enumerate(np.searchsorted(np.where(valid_vert)[0], np.where(valid_vert)[0])): 
        weighed_normals[vertex_idx] = np.sum(rot_vortex_normals[cluster_grp[cluster_indices[vertex_id]], :], axis = 0)
    weighed_normals /= np.linalg.norm(weighed_normals, axis = 1, keepdims = True)
    
    # Calculate an orthonormal eigenvector basis for each accumulated/weighted normals, resulting in (valid vortices  x 3) eigenvectors/values.
    pre_ev = np.empty((weighed_normals.shape[0], 3, 3))
    for idx in range(pre_ev.shape[0]):
        pre_ev[idx, :, :] = np.matmul(weighed_normals[[idx], :].T, weighed_normals[[idx], :])

    # Per normal basis - eigenvector/value calculation
    evec = list()
    for mat_idx in range(pre_ev.shape[0]):
        if ((pre_ev[mat_idx, :, :] == pre_ev[mat_idx, :, :].T).all() is False):
            raise AssertionError("Matrix should be diagonal.")
        
        mat = mpmath.matrix(np.eye(3) - pre_ev[mat_idx, :, :])
        mat.ctx.dps = double_precision
        (_, loc_evec) = mpmath.eigsy(mat)
         
        evec.append(loc_evec.tolist())
    
    # Concatenation
    evec = np.asarray(evec, dtype = float)
    evec = evec[:, :, ::-1]
    direction = np.sign(np.matmul(np.expand_dims(weighed_normals, axis = 1), evec[:, :, -1:]))
    direction[direction == 0] = 1
    evec *= direction
    evec = evec.swapaxes(1, 2)
    evec = evec.reshape(-1, 3)
     
    return evec

def find_nearest_neighbor(src_pts, tgt_pts, method = "kdtree"):
    """
    Employ one of two methods to find the nearest neighbors.
    
    In case a src_pts a list of points is provided, a model is trained, otherwise,
    pretrained models are used if a (KDTree or BallTree objects) are provided.
    
    Parameters
    ----------
    src_pts : numpy.ndarray, shape(m, 3)
              Points to build the kd-tree from.
    tgt_pts : numpy.ndarray, shape(n, 3)
              Point to match to the kd-tree.
    method : string
             Type of kd-tree chosen to identify neighbors, has to be eitehr 'kdtree' or 'balltree'.
               
    Returns
    -------
    result : tuple of (np.ndarray, sklearn.neighbors.KDTree or sklearn.neighbors.BallTree)
             - neigh_indices : np.ndarray, shape(n,)
                               Indices of the nearest neighbors.
             - tree : sklearn.neighbors.KDTree or sklearn.neighbors.BallTree
                      Tree build from the src pts. To be used in subsequently method calls to avoid rebuilding the tree.
    
    Raises
    ------
    NotImplementedError
        Type of kd-tree is invalid, has to be eitehr 'kdtree' or 'balltree'.
    """
    if (type(src_pts) is sklearn.neighbors.KDTree or type(src_pts) is sklearn.neighbors.BallTree):
        tree = src_pts
        if (method == "kdtree"):
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        elif (method == "balltree"):
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")
    else:
        if (method == "kdtree"):
            tree = sklearn.neighbors.KDTree(src_pts)
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        elif (method == "balltree"):
            tree = sklearn.neighbors.BallTree(src_pts)
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")

    return (neigh_indices, tree)

def find_valid_vertices(vertices_a, vertices_b, max_neighborhood_size = 5):
    """
    Match freesurfer reconstructed mri vertices (sphere) with model vertices (octahedron).
    
    Parameters
    ----------
    vertices_a : numpy.ndarray, shape(m, 3)
                 Freesurfer based vertices (sphere).
    vertices_b : numpy.ndarray, shape(n, 3)
                 Octahedron based vertices.
    max_neighborhood_size : int
                            Maximum size of the neighborhood.
               
    Returns
    -------
    vert_valid : numpy.ndarray, shape(m,)
                 Binary list of Freesurfer vertices that have a match in the model vertices (octahedron).
                 
    Raises
    ------
    AssertionError
        If duplicates cannot be resolved, the freesurfer reconstruction may be invalid.
    """    
    (nearest, nearest_tree) = find_nearest_neighbor(vertices_a, vertices_b)
    vert_valid = np.zeros((vertices_a.shape[0]))
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
        if (duplicates_resolved is False):
            raise AssertionError("Freesurfer reconstruction seems incorrectly distributed.")
    
    return vert_valid

def calc_eigendecomposition(cov, float_sz, finnpy_speedups_path):
    """
    Calculate the eigenvalue decomposition of a matrix with arbitrary precision.
    
    Parameters
    ----------
    cov : numpy.ndarray, shape(n, n)
          Matrix for which to calculate an eigenvalue decomposition
    float_sz : int
               Resolution of the floating point precision. Must be 32, 64, 80, 128, or 256.
    finnpy_speedups_path : string
                           Path to c_written finnpy_speedups. Currently only available via Github and need to be precompiled locally.
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - evals : numpy.ndarray, shape(n,)
                       Eigenvalues
             - evecs : numpy.ndarray, shape(n, n)
                       Eigenvectors
    
    Raises
    ------
    AssertionError
        Floating point precision invalid, has to be 32, 64, 80, 128, or 256.
    """
    if (float_sz not in [32, 64, 80, 128, 256]):
        raise AssertionError("Floating point precision %i not supported. Must be 32, 64, 80, 128, or 256." % (float_sz,))
    
    if (float_sz == 64):
        mat = np.asarray(cov, dtype = float)
        (evals, evecs) = np.linalg.eigh(mat)
    else:
        if (os.path.exists(finnpy_speedups_path)):
            speedup_fncts = ctypes.CDLL(finnpy_speedups_path)
            
            precision_c = ctypes.c_uint(float_sz)
            size = int(cov.shape[0])
            size_c = ctypes.c_uint(size)
            bio_sensor_noise_cov_c = (ctypes.c_double * int(size * size))(*cov.reshape(-1))
            evecs_c = (ctypes.c_double * int(size * size))(*np.empty((size * size,)))
            evals_c = (ctypes.c_double * size)(*np.empty((size,)))
            
            speedup_fncts.finnpy_eigen_decomp(bio_sensor_noise_cov_c, size_c, evals_c, evecs_c, precision_c)
            
            evals = np.asarray(evals_c)
            evecs = np.asarray(evecs_c).reshape((size, size)).T
        else:
            if (float_sz == 32):
                deci_plcs = 7
            elif (float_sz == 64):
                deci_plcs = 15
            elif (float_sz == 80):
                deci_plcs = 19
            elif (float_sz == 128):
                deci_plcs = 36
            elif (float_sz == 256):
                deci_plcs = 71
            else:
                raise AssertionError("Unknown float size")
            
            mat = mpmath.matrix(cov)
            # mat.ctx.dps = 40 #64 bit float is 15
            mat.ctx.dps = deci_plcs
            (evals, evecs) = mpmath.eigsy(mat)
         
            evals = np.asarray(evals.tolist(), dtype = float).squeeze(1)
            evecs = np.asarray(evecs.tolist(), dtype = float)
    
    return (evals, evecs)

def compute_baryzentric_params(face_verts, face):
    """
    Compute auxilaries for baryzentric parameters.
    
    Computes baryzentric parameters used to calculate baryzentric coordinates. In case
    many coordinates need to be computed for the same face, it may be advantageous
    to precompute the only face-dependnet baryzentric parameters separately.
    
    Parameters
    ----------
    face_verts : numpy.ndarray, shape(n, 3)
                 Vertices of all faces.
    face : numpy.ndarray, shape(3,)
           Used to select specific vertices from face vertices.
               
    Returns
    -------
    barycentric_params : (float, float, float, float)
                         Barycentric parameters, required in baryzentric coordinate calculation. 
                         Format: (u, v, n, n_norm).
    """
    u = face_verts[face[1], :] - face_verts[face[0], :]
    v = face_verts[face[2], :] - face_verts[face[0], :]
    n = fast_3D_cross_product_single(u, v)
        
    tmp = fast_3D_dot_product_single(n, n)
    n_norm = np.zeros(tmp.shape)
    n_norm[tmp != 0] = 1 / tmp[tmp != 0]
    
    return (u, v, n, n_norm)

def compute_baryzentric_coordinates(src_vort, face_verts, face, barycentric_params = None):
    """
    Compute the baryzentric coordinates of a vortex relative to a given face.
    
    Parameters
    ----------
    src_vort : numpy.ndarray, shape(3,)
               Seed vortex.
    face_verts : numpy.ndarray, shape(n, 3)
                 Vertices of all faces.
    face : numpy.ndarray, shape(3,)
           Used to select specific vertices from face vertices.
    barycentric_params : (float, float, float, float)
                         Optional, can be used to provide parameters required in baryzentric coordinate calculation. 
                         Format: (u, v, n, n_norm). As these are independent of the seed vortex, they may be pre-
                         computed to speed up repeated baryzentric coordinate computations.
               
    Returns
    -------
    barycentric_coordinates : (float, float, float)
                              Barycentric coordinate (alpha, beta, gamma) defining a vortex relative to a face's vertices. 
    
    Reference
    ---------
    From https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    """
    if (barycentric_params is None):
        (u, v, n, n_norm) = compute_baryzentric_params(face_verts, face)
    else:
        (u, v, n, n_norm) = barycentric_params
    
    w = src_vort - face_verts[face[0], :]
    gamma = fast_3D_dot_product_single(fast_3D_cross_product_single(u, w), n) * n_norm
    beta = fast_3D_dot_product_single(fast_3D_cross_product_single(w, v), n) * n_norm
    alpha = 1 - gamma - beta
     
    return (alpha, beta, gamma)

def compute_faces_bounding_boxes(face_verts, faces):
    """
    Compute the centers and maximum axis length of a list of faces, defined by the vertices and face indices.
    
    Parameters
    ----------
    face_verts : numpy.ndarray, shape(n, 3)
                 Vertices of all faces.
    faces : numpy.ndarray, shape(3,)
            Used to select specific vertices from face vertices.
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - faces_centers : numpy.ndarray, shape(n, 3)
                               Optional, centers of each face.
             - max_faces_ax_sz : numpy.ndarray, shape(n,)
                                 Optional, longest side of each face.
    """
    centers = np.asarray([(np.max(face_verts[faces, 0], axis = 1) + np.min(face_verts[faces, 0], axis = 1)) / 2,
                          (np.max(face_verts[faces, 1], axis = 1) + np.min(face_verts[faces, 1], axis = 1)) / 2,
                          (np.max(face_verts[faces, 2], axis = 1) + np.min(face_verts[faces, 2], axis = 1)) / 2]).T 
    
    max_ax_szs = np.asarray([np.max(face_verts[faces, 0], axis = 1) - np.min(face_verts[faces, 0], axis = 1), 
                             np.max(face_verts[faces, 1], axis = 1) - np.min(face_verts[faces, 1], axis = 1),
                             np.max(face_verts[faces, 2], axis = 1) - np.min(face_verts[faces, 2], axis = 1)]).T
    
    return (centers, max_ax_szs)

def find_closest_faces(src_vertices, face_verts, faces, faces_centers = None, max_faces_ax_sz = None, kdtree = None):
    """
    For a list of vertices, determine which face (and point within) are closest to each seed vortex.
    
    First, a kd-tree is constructed from the centers of all possible faces. Subsequently, the closest face-centerpoint
    to a vortex in question is determined. Then, all faces within max(distance to face center, longest face axis) are
    queried for distance. This is done by projecting the point on the face and measuring the distance between
    the original point and the projected point.
    
    The reduction of faces to their center points allows the use of kd trees. Furthermore, via the use of maximum of
    distance to face center and longest face axis, a bounding box may be constructed to guarantee the closest face
    is identified.
    
    Parameters
    ----------
    src_vertices : numpy.ndarray, shape(m, 3)
                   Seed vortex.
    face_verts : numpy.ndarray, shape(n, 3)
                 Vertices of all faces.
    faces : numpy.ndarray, shape(3,)
            Used to select specific vertices from face vertices. 
    faces_centers : numpy.ndarray, shape(n, 3)
                    Optional, centers of each face. 
    max_faces_ax_sz : numpy.ndarray, shape(n,)
                      Optional, longest side of each face.
    kdtree : numpy.ndarray, shape(3,)
             kdtree populated with the faces' centers. 
               
    Returns
    -------
    proj_vortex : numpy.ndarray, shape(3,)
                  Vortex projected inside the face.
    """
    if (faces_centers is None or max_faces_ax_sz is None):
        (faces_centers, max_faces_ax_sz) = compute_faces_bounding_boxes(face_verts, faces)
    
    if (kdtree is None):
        kdtree = sklearn.neighbors.KDTree(faces_centers)
    
    closest_face_ids = np.empty((len(src_vertices,)))
    proj_pts = np.empty((len(src_vertices), 3))
    distances = np.empty((len(src_vertices,)))
    
    for (src_pt_idx, src_pt) in enumerate(src_vertices):
        closest_idx = int(kdtree.query([src_pt], 1)[1])
        # Use the longest axis of the closest bounding box to draw a circle around the src pt.
        loc_radius = np.max(max_faces_ax_sz[closest_idx])
        # +1e-12 to fix the < radius vs. <= radius in query_radius
        closest_face_id_cands = kdtree.query_radius([src_pt], np.max([loc_radius, np.linalg.norm(src_pt - faces_centers[closest_idx])]) + 1e-12)[0]
        
        cand_dist = np.empty(len(closest_face_id_cands))
        cand_proj_pts = np.empty((len(closest_face_id_cands), 3))
        for (iter_idx, closest_face_id_cand) in enumerate(closest_face_id_cands):
            cand_proj_pts[iter_idx] = find_closest_pt_in_face(src_pt, face_verts, faces[closest_face_id_cand])
            cand_dist[iter_idx] = np.linalg.norm(src_pt - cand_proj_pts[iter_idx])
        
        closest_idx = int(np.argmin(cand_dist))
        
        closest_face_ids[src_pt_idx] = closest_face_id_cands[closest_idx]
        proj_pts[src_pt_idx] = cand_proj_pts[closest_idx]
        distances[src_pt_idx] = cand_dist[closest_idx]
    
    closest_face_ids = np.asarray(closest_face_ids, dtype = int)
    return (closest_face_ids, distances, proj_pts)

def find_closest_pt_in_face(vortex, face_verts, face):
    """
    Calculate the closest point 'within' a face relative to a given vortex.
    
    Parameters
    ----------
    vortex : numpy.ndarray, shape(3,)
             Seed vortex.
    face_verts : numpy.ndarray, shape(n, 3)
                 Vertices of all faces.
    face : numpy.ndarray, shape(3,)
           Used to select specific vertices from face vertices. 
               
    Returns
    -------
    proj_vortex : numpy.ndarray, shape(3,)
                  Vortex projected inside the face.
    """
    # Reference: https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
    ba = face_verts[face[1], :] - face_verts[face[0], :]
    ca = face_verts[face[2], :] - face_verts[face[0], :]
    xa = vortex - face_verts[face[0], :]
    xb = vortex - face_verts[face[1], :]
    xc = vortex - face_verts[face[2], :]
    
    d1 = fast_3D_dot_product_single(ba, xa)
    d2 = fast_3D_dot_product_single(ca, xa)
    if (d1 < 0. and d2 < 0.):
        return face_verts[face[0], :]
    
    d3 = fast_3D_dot_product_single(ba, xb)
    d4 = fast_3D_dot_product_single(ca, xb)
    if (d3 >= 0. and d4 <= d3):
        return face_verts[face[1], :]
    
    d5 = fast_3D_dot_product_single(ba, xc)
    d6 = fast_3D_dot_product_single(ca, xc)
    if (d6 >= 0. and d5 <= d6):
        return face_verts[face[2], :]

    vc = d1 * d4 - d3 * d2
    if (vc < 0. and d1 >= 0. and d3 <= 0.):
        v = d1 / (d1 - d3)
        return face_verts[face[0], :] + v * ba
    
    vb = d5 * d2 - d1 * d6
    if (vb <= 0. and d2 >= 0. and d6 <= 0.):
        v = d2 / (d2 - d6)
        return face_verts[face[0], :] + v * ca
    
    va = d3 * d6 - d5 * d4
    if (va <= 0. and (d4 - d3) >= 0. and (d5 - d6) >= 0.):
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return face_verts[face[1], :] + v * (face_verts[face[2], :] - face_verts[face[1], :])
    
    proj_vortex = face_verts[face[0], :] + vb * 1. / (va + vb + vc) * ba + vc * 1. / (va + vb + vc) * ca
    
    return proj_vortex
