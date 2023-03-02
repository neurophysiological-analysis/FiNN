'''
Created on Oct 17, 2022

@author: voodoocode
'''

import numpy as np
import nibabel.freesurfer
import finnpy.source_reconstruction.utils

import warnings

def read_freesurfer_annotation(path):
    (annot, _, labels) = nibabel.freesurfer.read_annot(path)
    np.where(annot == 1)
    regions = dict()
    
    for (region_idx, region_name) in enumerate(labels):
        regions[region_name.decode("utf-8")] = np.where(annot == region_idx)[0]
    
    return regions

def get_mri_to_model_trans(mri_neigh_faces):
    reshaped_mri_neigh_faces = mri_neigh_faces.reshape(-1)
    (_, unique_reshaped_mri_neigh_face_indices) = np.unique(reshaped_mri_neigh_faces, return_index = True)
    mri_to_model_trans = np.ones((np.max(reshaped_mri_neigh_faces) + 1)) * np.nan
    for (idx, value) in enumerate(unique_reshaped_mri_neigh_face_indices):
        mri_to_model_trans[reshaped_mri_neigh_faces[value]] = idx        
    mri_to_model_trans = np.asarray(mri_to_model_trans, dtype = int)
    
    return mri_to_model_trans

def get_sphere_faces(fs_avg_path, hemisphere, model_vert, model_faces):
    (fs_avg_surf_sph_vert, _) = nibabel.freesurfer.read_geometry(fs_avg_path + "surf/" + hemisphere + ".sphere")
    neigh_indices = finnpy.source_reconstruction.utils.find_nearest_neighbor(fs_avg_surf_sph_vert/100, model_vert)[0]
    neigh_faces = [neigh_indices[face] for face in model_faces]; neigh_faces = np.asarray(neigh_faces, dtype = int)

    return neigh_faces

def apply_source_region_model(fs_avg_src_data, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert,
                              model_vert, model_faces, fs_avg_path):
    
    morphed_epoch_data = list()
    morphed_epoch_channels = list()
    morphed_region_names = list()
    
    for hemisphere in ["lh", "rh"]:
        mri_neigh_faces = get_sphere_faces(fs_avg_path, hemisphere, model_vert, model_faces)
        mri_to_model_trans = get_mri_to_model_trans(mri_neigh_faces)
        
        #### Translates from vertex id to the nth-channel, e.g. vertex ids [0, 11, 24, ..., 163825] to model ids [0, 1, 2, ..., 4097]
        hem_data = fs_avg_src_data[:len(np.where(src_fs_avg_valid_lh_vert)[0]), :] if (hemisphere == "lh") else fs_avg_src_data[len(np.where(src_fs_avg_valid_rh_vert)[0]):, :]
        
        regions = read_freesurfer_annotation(fs_avg_path + "label/" + hemisphere + ".aparc.annot")
        for region_name in regions.keys():
            if (region_name == "unknown"):
                continue
            
            #Gets the vertex ids for a specific region
            mri_region_vertices_ids = np.where(src_fs_avg_valid_lh_vert)[0][np.in1d(np.where(src_fs_avg_valid_lh_vert)[0], regions[region_name])]
            model_region_vertices_ids = mri_to_model_trans[mri_region_vertices_ids]
            if (len(hem_data[model_region_vertices_ids, :]) == 0):
                continue #In case there is no channel within a region, skip it
            
            morphed_epoch_data.append(np.mean(np.abs(hem_data[model_region_vertices_ids, :]), axis = 0))
            morphed_epoch_channels.append(model_region_vertices_ids)
            morphed_region_names.append(hemisphere + "_" + region_name)
    
    return (morphed_epoch_data, morphed_epoch_channels, morphed_region_names)









