"""
Created on Oct 21, 2022.

@author: voodoocode
"""

import nibabel.freesurfer
import numpy as np
import scipy.spatial

import finnpy.src_rec.utils  # @UnresolvedImport


class Cort_mdl():
    """
    Container populated with the following cortical surface (MRI) based items.
    
    Parameters
    ----------
    lh_vert : np.ndarray, shape(lh_vtx_cnt, 3)
              White matter surface model vertices (left hemisphere).
    lh_faces : np.ndarray, shape(lh_face_cnt, 3)
               White matter surface model faces (left hemisphere).
    lh_valid_vert : np.ndarray, shape(lh_vtx_cnt,)
                    Valid flags for white matter surface model vertices (left hemisphere).
    rh_vert : np.ndarray, shape(rh_vtx_cnt, 3)
              White matter surface model vertices (right hemisphere).
    rh_faces : np.ndarray, shape(rh_face_cnt, 3)
               White matter surface model faces (right hemisphere).
    lh_valid_vert : np.ndarray, shape(rh_vtx_cnt,)
                    Valid flags for white matter surface model vertices (right hemisphere).
    octa_model_vert : np.ndarray, shape(octa_mdl_vtx_cnt, 3)
                      Octamodel vertices (left hemisphere).
    octa_model_faces : np.ndarray, shape(octa_mdl_face_cnt, 3)
                       Octamodel faces (right hemisphere).
    """
    
    def __init__(self,  # noqa: DOC103
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

def get(anatomy_path, subj_name, signal_type, coreg, bem_mdl):
    """
    Read and filter cortical freesurfer data.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    signal_type : string
                  Mode is either "EEG" or "MEG". 
    coreg : finnpy.src_rec.coregistration_meg_mri.Coreg
            Container with different transformation matrices
    bem_mdl : finnpy.src_rec.bem_mdl.BEM_mdl
              Container class, populed with the following items:
              
              vert : list() of np.ndarray, [shape(scaled_vtx_cnt, 3), ...]
                     Remaining vertices of a skin/skull model vertices.
              faces : list() of np.ndarray, [shape(scaled_face_cnt, 3), ...]
                      Remaining faces of a skin/skull model faces.
              faces_normal : np.ndarray, shape(scaled_face_cnt, 3)
                             Normals of the individual remaining inner skull faces.
              faces_area : np.ndarray, shape(scaled_face_cnt)
                           Surface area of the remaining faces.
              bem_solution : np.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                             BEM solution (preliminary step for the calculation of the forward model).
               
    Returns
    -------
    cort_mdl : finnpy.src_rec.cort_mdl.Cort_mdl
               Container populated with the following items:
               
               - lh_vert : np.ndarray, shape(lh_vtx_cnt, 3)
                           White matter surface model vertices (left hemisphere).
               - lh_faces : np.ndarray, shape(lh_face_cnt, 3)
                            White matter surface model faces (left hemisphere).
               - lh_valid_vert : np.ndarray, shape(lh_vtx_cnt,)
                                 Valid flags for white matter surface model vertices (left hemisphere).
               - rh_vert : np.ndarray, shape(rh_vtx_cnt, 3)
                           White matter surface model vertices (right hemisphere).
               - rh_faces : np.ndarray, shape(rh_face_cnt, 3)
                            White matter surface model faces (right hemisphere).
               - rh_valid_vert : np.ndarray, shape(rh_vtx_cnt,)
                                 Valid flags for white matter surface model vertices (right hemisphere).
               - octa_model_vert : np.ndarray, shape(octa_mdl_vtx_cnt, 3)
                                   Octamodel vertices (left hemisphere).
               - octa_model_faces : np.ndarray, shape(octa_mdl_face_cnt, 3)
                                    Octamodel faces (right hemisphere).
    """
    # Read model
    (lh_white_vert, lh_white_faces,
     rh_white_vert, rh_white_faces,
     lh_sphere_vert,
     rh_sphere_vert) = _read(anatomy_path, subj_name, signal_type, coreg)
     
    # Checks whether the inflated spherical freesurfer estimate is similar to an octahedron.
    (octa_model_vert, octa_model_faces) = _create_mesh()
    (lh_valid_vert, rh_valid_vert) = _check_structural_shape_integrity(lh_sphere_vert, rh_sphere_vert, octa_model_vert)
    
    # Checks whether cortex vertices are outside the inner skull delimiter.
    (lh_valid_vert, rh_valid_vert) = _rm_outside_ref_vertices(bem_mdl.vert[bem_mdl.INNER_SKULL_IDX], 
                                                              lh_white_vert, lh_valid_vert,
                                                              rh_white_vert, rh_valid_vert)
    
    return Cort_mdl(lh_white_vert, lh_white_faces, lh_valid_vert,
                    rh_white_vert, rh_white_faces, rh_valid_vert,
                    octa_model_vert, octa_model_faces)

def _read(anatomy_path, subj_name, signal_type, coreg):
    """
    Read cortical freesurfer data.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    signal_type : string
                  Either "EEG" or "MEG".
    coreg : finnpy.src_rec.coregistration_meg_mri.Coreg
            Container with different transformation matrices
    
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
             - lh_white_vert : np.ndarray, shape(lh_white_vtx_cnt, 3)
                               White matter surface model vertices (left hemisphere).
             - lh_white_faces : np.ndarray, shape(lh_white_face_cnt, 3)
                               White matter surface model faces (left hemisphere).
             - rh_white_vert : np.ndarray, shape(rh_white_vtx_cnt, 3)
                               White matter surface model vertices (right hemisphere).
             - rh_white_faces : np.ndarray, shape(rh_white_face_cnt, 3)
                               White matter surface model faces (right hemisphere).
             - lh_sphere_vert : np.ndarray, shape(lh_sphere_vtx_cnt, 3)
                               Spherical freesurfer head model vertices (left hemisphere).
             - rh_sphere_vert : np.ndarray, shape(rh_sphere_vtx_cnt, 3)
                               Spherical freesurfer head model vertices (right hemisphere).
    """
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (lh_white_vert, lh_white_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.white")
    (rh_white_vert, rh_white_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/rh.white")
    
    (lh_sphere_vert, _) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/lh.sphere")
    (rh_sphere_vert, _) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/surf/rh.sphere")
    
    if (signal_type == "MEG"):
        lh_white_vert *= coreg.rotors[6:9]
        rh_white_vert *= coreg.rotors[6:9]
        
    # Scale from m to mm
    lh_white_vert /= 1000
    rh_white_vert /= 1000
        
    # Normalize geometry
    lh_sphere_vert = finnpy.src_rec.utils.fast_3D_normalize_vec_multi(lh_sphere_vert)
    rh_sphere_vert = finnpy.src_rec.utils.fast_3D_normalize_vec_multi(rh_sphere_vert)
    
    return (lh_white_vert, lh_white_faces,
            rh_white_vert, rh_white_faces,
            lh_sphere_vert,
            rh_sphere_vert)

def _create_mesh(octahedron_level = 6):
    """
    Create a sphere from an octahedron and prunes duplicate vertices/faces.
    
    Parameters
    ----------
    octahedron_level : int
                       Level of the octahedron used as the mesh model.
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - vert : np.ndarray, shape(octa_vtx_cnt, 3)
                      Vertices of the spherical model (octahedron).
             - faces : np.ndarray, shape(octa_face_cnt, 3)
                      Faces of the spherical model (octahedron).
    """
    (vert, faces) = finnpy.src_rec.sphere_mdl.calculate_sphere_from_octahedron(octahedron_level)
    (vert, faces) = finnpy.src_rec.sphere_mdl.prune_closeby_vert(vert, faces)
    
    return (vert, faces)

def _check_structural_shape_integrity(lh_sphere_vert, rh_sphere_vert, octa_model_vert):
    """
    Create a source mesh model by warping an octahedron towards the surface sphere created by freesurfer.
    
    Parameters
    ----------
    lh_sphere_vert : np.ndarray, shape(lh_sphere_vtx_cnt, 3)
                     Spherical freesurfer head model vertices (left hemisphere).
    rh_sphere_vert : np.ndarray, shape(rh_sphere_vtx_cnt, 3)
                     Spherical freesurfer head model vertices (right hemisphere).
    octa_model_vert : np.ndarray, shape(octa_vtx_cnt, 3)
                      Vertices of the spherical model (octahedron).
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - lh_valid_vert : np.ndarray, shape(lh_white_vtx_cnt,)
                               Vertices with a match in the spherical model (left hemisphere).
             - rh_valid_vert : np.ndarray, shape(rh_white_vtx_cnt,)
                               Vertices with a match in the spherical model (right hemisphere).            
    """
    # Get valid vertices
    lh_valid_vert = finnpy.src_rec.utils.find_valid_vertices(lh_sphere_vert, octa_model_vert)
    rh_valid_vert = finnpy.src_rec.utils.find_valid_vertices(rh_sphere_vert, octa_model_vert)
    
    return (lh_valid_vert, rh_valid_vert)

def _rm_outside_ref_vertices(ref_vertices,
                             lh_white_vert_mri, lh_valid_vert,
                             rh_white_vert_mri, rh_valid_vert):
    """
    Identify vertices outside the structure defined by delaunay triangles of the reference vertices.
    
    Parameters
    ----------
    ref_vertices : np.ndarray, shape(scaled_vtx_cnt, 3)
                   Vertices of the reference structure.
    lh_white_vert_mri : np.ndarray, shape(lh_white_vtx_cnt, 3)
                        White matter surface model vertices (left hemisphere).
    lh_valid_vert : np.ndarray, shape(lh_vtx_cnt,)
                    Valid flags for white matter surface model vertices (left hemisphere).
    rh_white_vert_mri : np.ndarray, shape(rh_white_vtx_cnt, 3)
                        White matter surface model vertices (right hemisphere).
    rh_valid_vert : np.ndarray, shape(rh_vtx_cnt,)
                    Valid flags for white matter surface model vertices (right hemisphere).
    
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - lh_valid_vert : np.ndarray, shape(lh_vtx_cnt,)
                               Valid flags for white matter surface model vertices (left hemisphere).
             - rh_valid_vert : np.ndarray, shape(rh_vtx_cnt,)
                               Valid flags for white matter surface model vertices (right hemisphere).
    """
    approx_surface = scipy.spatial.Delaunay(ref_vertices)  # pylint: disable=no-member
    lh_valid_vert = _update_invalid_vertices(approx_surface, lh_white_vert_mri, lh_valid_vert)
    rh_valid_vert = _update_invalid_vertices(approx_surface, rh_white_vert_mri, rh_valid_vert)
    
    return (lh_valid_vert, rh_valid_vert)

def _update_invalid_vertices(approx_surface, white_vert, valid_vert):
    """
    Update invalid vertices.
    
    Parameters
    ----------
    approx_surface : scipy.spatial._qhull.Delaunay
                     Approximate in/out skull/skin surface.
    white_vert : np.ndarray, shape(white_vtx_cnt, 3)
                 White matter surface vertices.
    valid_vert : np.ndarray, shape(white_vtx_cnt, 3)
                 Binary list of valid vertices.
               
    Returns
    -------
    trans_white_vert : np.ndarray, shape(white_vtx_cnt, 3)
                       Transformed vertices.
    """
    data = white_vert[np.asarray(valid_vert, dtype=bool)]
    inside_check = (approx_surface.find_simplex(data) != -1)
    valid_vert[np.where(valid_vert)[0][~inside_check]] = False
    
    return valid_vert
