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
    
    :param octahedron_level: Order (recursions) of the octahedron. 
    
    :return: Vertices and faces of the octahedron.
    """
    (vert, faces) = finnpy.source_reconstruction.sphere_model.calculate_sphere_from_octahedron(octahedron_level)
    (vert, faces) = finnpy.source_reconstruction.sphere_model.prune_closeby_vert(vert, faces)
    
    return (vert, faces)

def _read_cortical_data(subject_path):
    """
    Reads cortical freesurfer data.
    
    :param subject_path: Subject's freesurfer path.
    
    :return: Vertices and faces of the cortical white matter for left and right hemisphere. Additionally, vertices of freesurfer's sphere model.
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
    
    :param subject_path: Subject's freesurfer path.
    
    :return: Vertices and faces of the freesurfer extracted cortical white matter surface models, a list of matched freesurfer reconstructed mri vertices (sphere) with model vertices (octahedron), and octahedron vertices and faces. 
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


