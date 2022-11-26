'''
Created on Oct 13, 2022

@author: voodoocode
'''

import nibabel.freesurfer

import finnpy.source_reconstruction.utils
import finnpy.source_reconstruction.sphere_model
import warnings

def create_mesh(octahedron_level = 6):
    (vert, faces) = finnpy.source_reconstruction.sphere_model.calculate_sphere_from_octahedron(octahedron_level)
    (vert, faces) = finnpy.source_reconstruction.sphere_model.prune_closeby_vert(vert, faces)
    
    return (vert, faces)

def read_cortical_data(path):
    (lh_geom_white_vert, lh_geom_white_faces) = nibabel.freesurfer.read_geometry(path + "surf/lh.white")
    (rh_geom_white_vert, rh_geom_white_faces) = nibabel.freesurfer.read_geometry(path + "surf/rh.white")
    
    (lh_geom_sphere_vert, _) = nibabel.freesurfer.read_geometry(path + "surf/lh.sphere")
    (rh_geom_sphere_vert, _) = nibabel.freesurfer.read_geometry(path + "surf/rh.sphere")
    
    return (lh_geom_white_vert, lh_geom_white_faces,
            rh_geom_white_vert, rh_geom_white_faces,
            lh_geom_sphere_vert,
            rh_geom_sphere_vert)

def create_source_mesh_model(path):
    #Get reference mesh
    (model_vert, model_faces) = create_mesh()
    
    #Load surface data from freesurfer reconstructions
    (lh_geom_white_vert, lh_geom_white_faces,
     rh_geom_white_vert, rh_geom_white_faces,
     lh_geom_sphere_vert,
     rh_geom_sphere_vert) = read_cortical_data(path)
        
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


