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
    vert : numpy.ndarray, shape(n3, 3)
           Vertices of the spherical model (octahedron).
    faces : numpy.ndarray, shape(m3, 3)
            Faces of the spherical model (octahedron).
    """
    (vert, faces) = finnpy.source_reconstruction.sphere_model.calculate_sphere_from_octahedron(octahedron_level)
    (vert, faces) = finnpy.source_reconstruction.sphere_model.prune_closeby_vert(vert, faces)
    
    return (vert, faces)

def _read_cortical_data(subject_path):
    """
    Reads cortical freesurfer data.
    
    Parameters
    ----------
    subject_path : string
                   Subject's freesurfer path.
               
    Returns
    -------
    lh_geom_white_vert : numpy.ndarray, shape(m1, 3)
                         White matter surface model vertices (left hemisphere).
    lh_geom_white_faces : numpy.ndarray, shape(n1, 3)
                          White matter surface model faces (left hemisphere).
    rh_geom_white_vert : numpy.ndarray, shape(m2, 3)
                         White matter surface model vertices (right hemisphere).
    rh_geom_white_faces : numpy.ndarray, shape(n2, 3)
                          White matter surface model faces (right hemisphere).
    lh_geom_sphere_vert : numpy.ndarray, shape(m1, 3)
                          Spherical freesurfer head model vertices (left hemisphere).
    rh_geom_sphere_vert : numpy.ndarray, shape(m2, 3)
                          Spherical freesurfer head model vertices (right hemisphere).
    """
    (lh_geom_white_vert, lh_geom_white_faces) = nibabel.freesurfer.read_geometry(subject_path + "surf/lh.white")
    (rh_geom_white_vert, rh_geom_white_faces) = nibabel.freesurfer.read_geometry(subject_path + "surf/rh.white")
    
    (lh_geom_sphere_vert, _) = nibabel.freesurfer.read_geometry(subject_path + "surf/lh.sphere")
    (rh_geom_sphere_vert, _) = nibabel.freesurfer.read_geometry(subject_path + "surf/rh.sphere")
    
    return (lh_geom_white_vert, lh_geom_white_faces,
            rh_geom_white_vert, rh_geom_white_faces,
            lh_geom_sphere_vert,
            rh_geom_sphere_vert)

def create_source_mesh_model(subject_path):
    """
    Creates a source mesh model by warping an octahedron towards the surface sphere created by freesurfer.
    
    Parameters
    ----------
    subject_path : string
                   Subject's freesurfer path.
               
    Returns
    -------
    lh_geom_white_vert : numpy.ndarray, shape(m1, 3)
                         White matter surface model vertices (left hemisphere).
    lh_geom_white_faces : numpy.ndarray, shape(n1, 3)
                          White matter surface model faces (left hemisphere).
    rh_geom_white_vert : numpy.ndarray, shape(m2, 3)
                         White matter surface model vertices (right hemisphere).
    rh_geom_white_faces : numpy.ndarray, shape(n2, 3)
                          White matter surface model faces (right hemisphere).
    valid_geom_lh_vert : numpy.ndarray, shape(m1,)
                         Vertices with a match in the spherical model (left hemisphere).
    valid_geom_rh_vert : numpy.ndarray, shape(m2,)
                         Vertices with a match in the spherical model (right hemisphere).
    model_vert : numpy.ndarray, shape(n3, 3)
                 Vertices of the spherical model (octahedron).
    model_faces : numpy.ndarray, shape(m3, 3)
                  Faces of the spherical model (octahedron).               
    """
    #Get reference mesh
    (model_vert, model_faces) = _create_mesh()
    
    #Load surface data from freesurfer reconstructions
    (lh_geom_white_vert, lh_geom_white_faces,
     rh_geom_white_vert, rh_geom_white_faces,
     lh_geom_sphere_vert,
     rh_geom_sphere_vert) = _read_cortical_data(subject_path)
        
    #Normalize geometric
    lh_geom_sphere_vert = finnpy.source_reconstruction.utils.norm_vert(lh_geom_sphere_vert)
    rh_geom_sphere_vert = finnpy.source_reconstruction.utils.norm_vert(rh_geom_sphere_vert)
    
    #Get valid vertices
    valid_geom_lh_vert = finnpy.source_reconstruction.utils.find_valid_vertices(lh_geom_sphere_vert, model_vert)
    valid_geom_rh_vert = finnpy.source_reconstruction.utils.find_valid_vertices(rh_geom_sphere_vert, model_vert)
    
    return (lh_geom_white_vert, lh_geom_white_faces,
            rh_geom_white_vert, rh_geom_white_faces,
            valid_geom_lh_vert, valid_geom_rh_vert,
            model_vert, model_faces)


