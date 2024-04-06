'''
Created on Oct 13, 2022

@author: voodoocode
'''

import os
import numpy as np
import scipy.spatial

import finnpy.src_rec.utils

def _tessellate_sphere(vert, faces, level):
    """    
    Inflates the sphere.
    
    Parameters
    ----------
    vert : numpy.ndarray, shape(initial_vtx_cnt, 3)
           Vertices of the sphere seed.
    faces : numpy.ndarray, shape(initial_face_cnt, 3)
            Faces of the sphere seed.
    level : int
            Target level of the sphere model.
               
    Returns
    -------
    vert : numpy.ndarray, shape(inflated_vtx_cnt, 3)
           Inflated vertices of the sphere.
    faces : numpy.ndarray, shape(inflated_face_cnt, 3)
            Inflated faces of the sphere.
    """

    new_vert = [None, None, None]
    new_vert_len = [None, None, None]
    new_faces = [None, None, None, None]
    for _ in range(0, level - 1):
        
        new_vert[0] = vert[faces[:, 0]] + vert[faces[:, 1]]; new_vert[0] = finnpy.src_rec.utils.norm_vert(new_vert[0])
        new_vert[1] = vert[faces[:, 1]] + vert[faces[:, 2]]; new_vert[1] = finnpy.src_rec.utils.norm_vert(new_vert[1])
        new_vert[2] = vert[faces[:, 0]] + vert[faces[:, 2]]; new_vert[2] = finnpy.src_rec.utils.norm_vert(new_vert[2])
        
        new_vert_cnt = np.cumsum([len(vert), len(new_vert[0]), len(new_vert[1]), len(new_vert[2])])
        new_vert_len[0] = np.arange(new_vert_cnt[0], new_vert_cnt[1])
        new_vert_len[1] = np.arange(new_vert_cnt[1], new_vert_cnt[2])
        new_vert_len[2] = np.arange(new_vert_cnt[2], new_vert_cnt[3])

        new_faces[0] = [faces[:, 0], new_vert_len[0], new_vert_len[2]]
        new_faces[1] = [new_vert_len[0], faces[:, 1], new_vert_len[1]]
        new_faces[2] = [new_vert_len[2], new_vert_len[1], faces[:, 2]]
        new_faces[3] = [new_vert_len[0], new_vert_len[1], new_vert_len[2]]
        
        vert = np.concatenate((vert, new_vert[0], new_vert[1], new_vert[2]))
        faces = np.asarray(new_faces)
        faces = np.moveaxis(faces, [0, 1, 2], [1, 2, 0])
        faces = np.vstack(faces)
    
    return (vert, faces)

def calculate_sphere_from_octahedron(level):
    """    
    Inflates the sphere.
    
    Parameters
    ----------
    level : int
            Target level of the sphere model.
               
    Returns
    -------
    vert : numpy.ndarray, shape(inflated_vtx_cnt, 3)
           Vertices of the sphere.
    faces : numpy.ndarray, shape(inflated_face_cnt, 3)
            Faces of the sphere.
    """
    vert = np.asarray([[1., 0., 0.],[0., 1., 0.], [0., 0., 1.], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype = float)
    faces = np.asarray([[0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5]], dtype = int)
        
    (vert, faces) = _tessellate_sphere(vert, faces, level)
    
    return (vert, faces)

def read_sphere_from_icosahedron_in_fs_order(fs_path, level):
    """
    Reads an icosahedron from freesurfer.
    
    Parameters
    ----------
    fs_path : string
              Path of the freesurfer directory.
    level : int
            Level of the icosahedron
               
    Returns
    -------
    vert : numpy.ndarray, shape(model_vtx_cnt, 3)
           Vertices of the icosahedron.
    faces : numpy.ndarray, shape(model_face_cnt, 3)
            Faces of the icosahedron.
    """
    
    file = open(fs_path + "lib/bem/ic" + str(level) + ".tri", "r")
    
    vert_cnt = int(file.readline().replace("\n", ""))
    vert = np.empty((vert_cnt, 3), dtype = float)
    
    for _ in range(vert_cnt):
        line = file.readline()
        (vort_id, x, y, z) = line.split()
        vort_id = int(vort_id); x = float(x); y = float(y); z = float(z) 
        vert[vort_id - 1, :] = [x, y, z]
        
    face_cnt = int(file.readline().replace("\n", ""))
    faces = np.empty((face_cnt, 3), dtype = float)
    
    for _ in range(face_cnt):
        line = file.readline()
        (face_id, x, y, z) = line.split()
        face_id = int(face_id); x = float(x); y = float(y); z = float(z) 
        faces[face_id - 1, :] = [x, y, z]
        
    file.close()
    
    return (vert, faces)

def calculate_sphere_from_icosahedron(level):
    """
    Calculates an icosahedron from a specific seed.
    
    Parameters
    ----------
    level : int
            Level of the icosahedron
               
    Returns
    -------
    vert : numpy.ndarray, shape(model_vtx_cnt, 3)
           Vertices of the icosahedron.
    faces : numpy.ndarray, shape(model_face_cnt, 3)
            Faces of the icosahedron.
    """
    vert = np.asarray([[.0000, .0000, 1.0000], [.8944, .0000, .4472], [.2764, .8507, .4472], [-.7236, .5257, .4472],
                       [-.7236, -.5257, .4472], [.2764, -.8507, .4472], [.7236, -.5257, -.4472], [.7236, .5257, -.4472],
                       [-.2764, .8507, -.4472], [-.8944, .0000, -.4472], [-.2764, -.8507, -.4472], [.0000, .0000, -1.0000]], dtype = float)
    
    faces = np.asarray([[ 0,  3,  4], [ 0,  4,  5], [ 0,  5,  1], [ 0,  1,  2], [ 0,  2,  3],
                        [ 3,  2,  8], [ 3,  8,  9], [ 3,  9,  4], [ 4,  9, 10], [ 4, 10,  5],
                        [ 5, 10,  6], [ 5,  6,  1], [ 1,  6,  7], [ 1,  7,  2], [ 2,  7,  8],
                        [ 8, 11,  9], [ 9, 11, 10], [10, 11,  6], [ 6, 11,  7], [ 7, 11,  8]], dtype = int)
        
    (vert, faces) = _tessellate_sphere(vert, faces, level = level + 1)
    (vert, faces) = prune_closeby_vert(vert, faces)
    
    return (vert, faces)

def prune_closeby_vert(vert, faces, threshold = 1e-6):
    """
    Prunes below threshold vertices from the model.
    
    Parameters
    ----------
    vert : numpy.ndarray, shape(model_vtx_cnt, 3)
           Vertices of the icosahedron.
    faces : numpy.ndarray, shape(model_face_cnt, 3)
            Faces of the icosahedron.
    level : float
            Distance threshold.
               
    Returns
    -------
    vert : numpy.ndarray, shape(pruned_vtx_cnt, 3)
           Vertices of the icosahedron.
    faces : numpy.ndarray, shape(pruned_face_cnt, 3)
            Faces of the icosahedron.
    """
    distances = scipy.spatial.distance_matrix(vert, vert, p = 2)
    distances = distances < threshold
    np.fill_diagonal(distances, False)

    valid_vertices_cnt = 0
    bad_vertices = np.zeros((vert.shape[0]), dtype = bool)
    for vortex_idx in range(len(vert)):
        if (bad_vertices[vortex_idx]):
            continue        
        
        loc_distances = distances[vortex_idx]
        
        loc_bad_vertices = np.zeros((vert.shape[0]), dtype = bool); loc_bad_vertices[vortex_idx:] = loc_distances[vortex_idx:]
        bad_vertices[loc_bad_vertices] = 1
        
        # Remove the references to bad vertices in faces & offset faces due to previously deleted vertices
        for old_idx in [vortex_idx, *np.where(loc_bad_vertices)[0].tolist()]:
            loc_bad_face_idx = np.argwhere(faces == old_idx)
            faces[loc_bad_face_idx[:, 0], loc_bad_face_idx[:, 1]] = valid_vertices_cnt
                
        valid_vertices_cnt += 1
        
    vert = vert[np.invert(bad_vertices), :]
    
    return (vert, faces)












