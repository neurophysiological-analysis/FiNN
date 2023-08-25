'''
Created on Oct 13, 2022

@author: voodoocode
'''

import nibabel.freesurfer

import finnpy.source_reconstruction.utils
import finnpy.source_reconstruction.sphere_model
import warnings

def _create_mesh(octahedron_level = 6):
    """
    Creates a sphere from an octahedron and prunes duplicate vertices/faces.
    
    Parameters
    ----------
    octahedron_level : int
                       Level of the octahedron used as the mesh model.
               
    Returns
    -------
    vert : numpy.ndarray, shape(octa_vtx_cnt, 3)
           Vertices of the spherical model (octahedron).
    faces : numpy.ndarray, shape(octa_face_cnt, 3)
            Faces of the spherical model (octahedron).
    """
    (vert, faces) = finnpy.source_reconstruction.sphere_model.calculate_sphere_from_octahedron(octahedron_level)
    (vert, faces) = finnpy.source_reconstruction.sphere_model.prune_closeby_vert(vert, faces)
    
    return (vert, faces)

def create_source_mesh_model():
    """
    Creates a source mesh model by warping an octahedron towards the surface sphere created by freesurfer.
               
    Returns
    -------
    octa_model_vert : numpy.ndarray, shape(octa_vtx_cnt, 3)
                      Vertices of the spherical model (octahedron).
    octa_model_faces : numpy.ndarray, shape(octa_face_cnt, 3)
                       Faces of the spherical model (octahedron).               
    """
    #Get reference mesh
    (octa_model_vert, octa_model_faces) = _create_mesh()
    
    return (octa_model_vert, octa_model_faces)

def match_source_mesh_model(lh_sphere_vert, rh_sphere_vert,
                            octa_model_vert):
    """
    Creates a source mesh model by warping an octahedron towards the surface sphere created by freesurfer.
    
    Parameters
    ----------
    lh_sphere_vert : numpy.ndarray, shape(lh_sphere_vtx_cnt, 3)
                     Spherical freesurfer head model vertices (left hemisphere).
    rh_sphere_vert : numpy.ndarray, shape(rh_sphere_vtx_cnt, 3)
                     Spherical freesurfer head model vertices (right hemisphere).
    octa_model_vert : numpy.ndarray, shape(octa_vtx_cnt, 3)
                      Vertices of the spherical model (octahedron).
               
    Returns
    -------
    lh_valid_vert : numpy.ndarray, shape(lh_white_vtx_cnt,)
                          Vertices with a match in the spherical model (left hemisphere).
    rh_valid_vert : numpy.ndarray, shape(rh_white_vtx_cnt,)
                          Vertices with a match in the spherical model (right hemisphere).            
    """
    
    #Get valid vertices
    lh_valid_vert = finnpy.source_reconstruction.utils.find_valid_vertices(lh_sphere_vert, octa_model_vert)
    rh_valid_vert = finnpy.source_reconstruction.utils.find_valid_vertices(rh_sphere_vert, octa_model_vert)
    
    return (lh_valid_vert, rh_valid_vert)


