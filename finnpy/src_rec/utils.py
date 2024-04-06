'''
Created on Oct 13, 2022

@author: voodoocode
'''

import numpy as np
import scipy.sparse
import mpmath
import os
import shutil
import subprocess
import sklearn.neighbors

def run_subprocess_in_custom_working_directory(subject_id, cmd):
    """
    Creates a custom work directory to run a freesurfer command within. 
    
    Parameters
    ----------
    subject_id : string
                 Name of the subject whose data is worked on.
    cmd : string
          The freesurfer command to be executed in the custom environment. 
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
    
    shutil.rmtree(path_to_tmp_cwd) # And removed later on

def fast_cross_product(a, b):
    """
    Calculates the cross product between two vectors. 
    Seemingly numpy has too much overhead here to be fast.
    
    Parameters
    ----------
    a : numpy.ndarray, shape(n,)
        The 1st vector in the cross product.
    b : numpy.ndarray(n,)
        The 2nd vector in the cross product.
               
    Returns
    -------
    res : numpy.ndarray, shape(n,)
          Crossproduct of vectors a x b.
    """
    
    res = np.empty(a.shape)
    res[:, 0] = a[:, 1]*b[:, 2]-a[:, 2]*b[:, 1]
    res[:, 1] = a[:, 2]*b[:, 0]-a[:, 0]*b[:, 2]
    res[:, 2] = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    
    return res

def norm_vert(vert):
    """
    Normalizes a vector. Seemingly numpy has too much overhead here to be fast.
    
    Parameters
    ----------
    vert : numpy.ndarray, shape(n,)
           The to be normalized vector.
               
    Returns
    -------
    vert : numpy.ndarray, shape(n,)
           The normalized vector.
    """
    size = np.sqrt(vert[:, 0]*vert[:, 0]+vert[:, 1]*vert[:, 1]+vert[:, 2]*vert[:, 2])
    vert[size > 0] /= size[size > 0, np.newaxis]
    return vert

def magn_of_vec(vec):
    """
    Calculates the magnitude of a vector
    
    Parameters
    ----------
    vec : numpy.ndarray, shape(n,)
          The vector in question.
               
    Returns
    -------
    magn : float
           The magnitude of vec.
    """
    
    return np.sqrt(vec[:, 0]*vec[:, 0]+vec[:, 1]*vec[:, 1]+vec[:, 2]*vec[:, 2])

def apply_inv_transformation(data, trans):
    """
    Applies the inverse of trans to data.
    
    Parameters:
    -----------
    data : np.ndarray, shape(n, 3)
           The data to be transformed.
    trans : np.ndarray, shape(4, 4)
            Transformation matrix.
               
    Results
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
    w = a[3]*b[3] + a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    x = a[3]*b[0] - a[0]*b[3] - a[1]*b[2] + a[2]*b[1]
    y = a[3]*b[1] - a[1]*b[3] - a[0]*b[2] + a[2]*b[0]
    z = a[3]*b[2] - a[2]*b[3] - a[0]*b[1] + a[1]*b[0]
    
    return 2 * np.arctan2(np.linalg.norm(np.asarray((x, y, z))), np.abs(w))

def orient_mat_to_block_format(orient_mat):
    """
    Transforms an orientation matrix (rotation matrix) into (sparse) block format.
    
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
                   mri_to_meg_trans, double_precision = 40):
    """
    Calculates an orthonormal basis of eigenvector/values for each supporting/valid point.
    
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
    mri_to_meg_trans : numpy.ndarray, shape(4, 4)
                        Transformation from MRI to head coordinates.
    double_precision : double
                       Numerical precision of the eigenvectors/values, 
                       defaults to 40 digits.
               
    Returns
    -------
    evec : numpy.ndarray, shape(valid_vtx_cnt * 3, 3)
           Orthonormal eigenbasis.
    """
    
    #Transforms normals into head space
    rot_vortex_normals = np.dot(vortex_normals, mri_to_meg_trans[:3, :3].T)
        
    #Accumulates normals from clusters, effectively interpolating/steering/weighing them into an more accurate depiction
    weighed_normals = np.zeros((np.where(valid_vert)[0].shape[0], 3))
    for (vertex_idx, vertex_id) in enumerate(np.searchsorted(np.where(valid_vert)[0], np.where(valid_vert)[0])): 
        weighed_normals[vertex_idx] = np.sum(rot_vortex_normals[cluster_grp[cluster_indices[vertex_id]], :], axis = 0)
    weighed_normals /= np.linalg.norm(weighed_normals, axis = 1, keepdims = True)
    
    #Calculate an orthonormal eigenvector basis for each accumulated/weighted normals, resulting in (valid vortices  x 3) eigenvectors/values.
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

def find_nearest_neighbor(src_pts, tgt_pts, method = "kdtree"):
    """
    Employs two methods to find the nearest neighbors. In case a src_pts a list of points is provided, a model is trained, otherwise, pretrained models are used if a (KDTree or BallTree objects) are provided.
    
    Parameters:
    -----------
    src_pts : numpy.ndarray, shape(m, 3)
              Points to build the kd-tree from.
    tgt_pts : numpy.ndarray, shape(n, 3)
              Point to match to the kd-tree.
               
    Results
    -------
    neigh_indices : np.ndarray, shape(n,)
                    Indices of the nearest neighbors.
    tree : sklearn.neighbors.KDTree or sklearn.neighbors.BallTree
           Tree build from the src pts. To be used in subsequently method calls to avoid rebuilding the tree.
    """
    
    if (type(src_pts) == sklearn.neighbors.KDTree or type(src_pts) == sklearn.neighbors.BallTree):
        tree = src_pts
        if (method == "kdtree"):
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        elif(method == "balltree"):
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")
    else:
        if (method == "kdtree"):
            tree = sklearn.neighbors.KDTree(src_pts)
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        elif(method == "balltree"):
            tree = sklearn.neighbors.BallTree(src_pts)
            neigh_indices = tree.query(tgt_pts, k = 1)[1].squeeze(1)
        else:
            raise NotImplementedError("This type of tree is not implemented.")

    return (neigh_indices, tree)

def find_valid_vertices(vertices_a, vertices_b, max_neighborhood_size = 5):
    """
    Matches freesurfer reconstructed mri vertices (sphere) with model vertices (octahedron).
    
    Parameters
    ----------
    vertices_a : numpy.ndarray, shape(m, 3)
                 Freesurfer based vertices (sphere).
    vertices_b : numpy.ndarray, shape(n, 3)
                 Octahedron based vertices.
               
    Returns
    -------
    vert_valid : numpy.ndarray, shape(m,)
                 Binary list of Freesurfer vertices that have a match in the model vertices (octahedron).
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
        if (duplicates_resolved == False):
            raise AssertionError("Freesurfer reconstruction seems incorrectly distributed.")
    
    return vert_valid




