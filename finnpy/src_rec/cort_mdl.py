'''
Created on Oct 21, 2022

@author: voodoocode
'''

import nibabel.freesurfer
import numpy as np
import scipy.spatial

import finnpy.src_rec.utils


class Cort_mdl():
    """
    Container populated with the following items:
               
    lh_vert : numpy.ndarray, shape(lh_vtx_cnt, 3)
              White matter surface model vertices (left hemisphere).
    lh_faces : numpy.ndarray, shape(lh_face_cnt, 3)
               White matter surface model faces (left hemisphere).
    lh_valid_vert : numpy.ndarray, shape(lh_vtx_cnt,)
                    Valid flags for white matter surface model vertices (left hemisphere).
    rh_vert : numpy.ndarray, shape(rh_vtx_cnt, 3)
              White matter surface model vertices (right hemisphere).
    rh_faces : numpy.ndarray, shape(rh_face_cnt, 3)
               White matter surface model faces (right hemisphere).
    lh_valid_vert : numpy.ndarray, shape(rh_vtx_cnt,)
                    Valid flags for white matter surface model vertices (right hemisphere).
    octa_model_vert : numpy.ndarray, shape(octa_mdl_vtx_cnt, 3)
                      Octamodel vertices (left hemisphere).
    octa_model_faces : numpy.ndarray, shape(octa_mdl_face_cnt, 3)
                       Octamodel faces (right hemisphere).
    """
    
    def __init__(self,
                 lh_vert, lh_faces, lh_valid_vert, 
                 rh_vert, rh_faces, rh_valid_vert, 
                 octa_mdl_vert, octa_mdl_faces):
        
        self.lh_vert = lh_vert
        self.lh_faces = lh_faces
        self.lh_valid_vert = lh_valid_vert
        
        self.rh_vert = rh_vert
        self.rh_faces = rh_faces
        self.rh_valid_vert = rh_valid_vert
        
        self.octa_mdl_vert = octa_mdl_vert
        self.octa_mdl_faces = octa_mdl_faces

def get(anatomy_path, subj_name, coreg, bem_reduced_vert):
    """
    Reads and filters cortical freesurfer data.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
               
    Returns
    -------
    cort_mdl : finnpy.src_rec.cort_mdl.Cort_mdl
               Container populated with the following items:
               
               lh_vert : numpy.ndarray, shape(lh_vtx_cnt, 3)
                         White matter surface model vertices (left hemisphere).
               lh_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                          White matter surface model faces (left hemisphere).
               lh_valid_vert : numpy.ndarray, shape(lh_vtx_cnt,)
                               Valid flags for white matter surface model vertices (left hemisphere).
               rh_vert : numpy.ndarray, shape(rh_vtx_cnt, 3)
                         White matter surface model vertices (right hemisphere).
               rh_faces : numpy.ndarray, shape(rh_face_cnt, 3)
                          White matter surface model faces (right hemisphere).
               lh_valid_vert : numpy.ndarray, shape(rh_vtx_cnt,)
                               Valid flags for white matter surface model vertices (right hemisphere).
               octa_model_vert : numpy.ndarray, shape(octa_mdl_vtx_cnt, 3)
                                 Octamodel vertices (left hemisphere).
               octa_model_faces : numpy.ndarray, shape(octa_mdl_face_cnt, 3)
                                  Octamodel faces (right hemisphere).
    """
    #Read model
    (lh_white_vert, lh_white_faces,
     rh_white_vert, rh_white_faces,
     lh_sphere_vert,
     rh_sphere_vert) = _read(anatomy_path, subj_name, coreg)
     
    #Cleanse model
    (octa_model_vert, octa_model_faces) = _create_source_mesh_model()
    (lh_valid_vert, rh_valid_vert) = _match_source_mesh_model(lh_sphere_vert, rh_sphere_vert, octa_model_vert)
    (lh_valid_vert, rh_valid_vert) = _rm_out_of_in_skull_vert(bem_reduced_vert,
                                                             lh_white_vert, lh_valid_vert,
                                                             rh_white_vert, rh_valid_vert)
    
    return Cort_mdl(lh_white_vert, lh_white_faces, lh_valid_vert,
                    rh_white_vert, rh_white_faces, rh_valid_vert,
                    octa_model_vert, octa_model_faces)

def _read(anatomy_path, subj_name, coreg):
    """
    Reads cortical freesurfer data.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    coreg : finnpy.src_rec.coregistration_meg_mri.Coreg
            Container with different transformation matrices
    
    Returns
    -------
    lh_white_vert : numpy.ndarray, shape(lh_white_vtx_cnt, 3)
                    White matter surface model vertices (left hemisphere).
    lh_white_faces : numpy.ndarray, shape(lh_white_face_cnt, 3)
                     White matter surface model faces (left hemisphere).
    rh_white_vert : numpy.ndarray, shape(rh_white_vtx_cnt, 3)
                    White matter surface model vertices (right hemisphere).
    rh_white_faces : numpy.ndarray, shape(rh_white_face_cnt, 3)
                     White matter surface model faces (right hemisphere).
    lh_sphere_vert : numpy.ndarray, shape(lh_sphere_vtx_cnt, 3)
                     Spherical freesurfer head model vertices (left hemisphere).
    rh_sphere_vert : numpy.ndarray, shape(rh_sphere_vtx_cnt, 3)
                     Spherical freesurfer head model vertices (right hemisphere).
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (lh_white_vert, lh_white_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.white")
    (rh_white_vert, rh_white_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/rh.white")
    
    (lh_sphere_vert, _) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.sphere")
    (rh_sphere_vert, _) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/rh.sphere")
    
    if (coreg is not None):
        lh_white_vert *= coreg.rotors[6:9]
        rh_white_vert *= coreg.rotors[6:9]
        
    #Scale from m to mm
    lh_white_vert /= 1000
    rh_white_vert /= 1000
        
    #Normalize geometry
    lh_sphere_vert = finnpy.src_rec.utils.norm_vert(lh_sphere_vert)
    rh_sphere_vert = finnpy.src_rec.utils.norm_vert(rh_sphere_vert)
    
    return (lh_white_vert, lh_white_faces,
            rh_white_vert, rh_white_faces,
            lh_sphere_vert,
            rh_sphere_vert)

def _create_source_mesh_model():
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
    (vert, faces) = finnpy.src_rec.sphere_mdl.calculate_sphere_from_octahedron(octahedron_level)
    (vert, faces) = finnpy.src_rec.sphere_mdl.prune_closeby_vert(vert, faces)
    
    return (vert, faces)

def _match_source_mesh_model(lh_sphere_vert, rh_sphere_vert,
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
    lh_valid_vert = finnpy.src_rec.utils.find_valid_vertices(lh_sphere_vert, octa_model_vert)
    rh_valid_vert = finnpy.src_rec.utils.find_valid_vertices(rh_sphere_vert, octa_model_vert)
    
    return (lh_valid_vert, rh_valid_vert)

def _rm_out_of_in_skull_vert(reduced_in_skull_vert, 
                            lh_white_vert_mri, lh_valid_vert,
                            rh_white_vert_mri, rh_valid_vert):
    approx_surface = scipy.spatial.Delaunay(reduced_in_skull_vert)
    lh_valid_vert = _update_invalid_vertices(approx_surface, lh_white_vert_mri, lh_valid_vert)
    rh_valid_vert = _update_invalid_vertices(approx_surface, rh_white_vert_mri, rh_valid_vert)
    
    return (lh_valid_vert, rh_valid_vert)

def _update_invalid_vertices(approx_surface, white_vert, valid_vert):
    """
    Updates invalid vertices
    
    Parameters
    ----------
    approx_surface : scipy.spatial._qhull.Delaunay
                     Approximate in/out skull/skin surface.
    white_vert : numpy.ndarray, shape(white_vtx_cnt, 3)
                 White matter surface vertices.
    valid_vert : numpy.ndarray, shape(white_vtx_cnt, 3)
                 Binary list of valid vertices.
               
    Returns
    -------
    trans_white_vert : numpy.ndarray, shape(white_vtx_cnt, 3)
                       Transformed vertices.
    """
    data = white_vert[np.asarray(valid_vert, dtype=bool)]
    inside_check = (approx_surface.find_simplex(data) != -1)
    valid_vert[np.where(valid_vert)[0][~inside_check]] = False
    
    return valid_vert


