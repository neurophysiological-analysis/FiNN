'''
Created on Oct 17, 2022

@author: voodoocode
'''

import numpy as np
import nibabel.freesurfer
import source_reconstruction.utils

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import warnings

def read_freesurfer_annotation(path):
    (annot, _, labels) = nibabel.freesurfer.read_annot(path)
    np.where(annot == 1)
    regions = dict()
    
    for (region_idx, region_name) in enumerate(labels):
        regions[region_name.decode("utf-8")] = np.where(annot == region_idx)[0]
    
    return regions

def get_translation_list(neigh_faces):
    tmp_vertices = neigh_faces.reshape(-1)
    (_, tmp_vertices_unique_idx) = np.unique(tmp_vertices, return_index = True)
    translation_list = np.ones((np.max(tmp_vertices) + 1)) * np.nan
    for (idx, value) in enumerate(tmp_vertices_unique_idx):
        translation_list[tmp_vertices[value]] = idx        
    translation_list = np.asarray(translation_list, dtype = int)
    
    return translation_list

def get_sphere_faces(fs_avg_path, hemisphere, model_vert, model_faces):
    (fs_avg_surf_sph_vert, _) = nibabel.freesurfer.read_geometry(fs_avg_path + "surf/" + hemisphere + ".sphere")
    neigh_indices = source_reconstruction.utils.find_nearest_neighbor(fs_avg_surf_sph_vert/100, model_vert)[0]
    neigh_faces = [neigh_indices[face] for face in model_faces]; neigh_faces = np.asarray(neigh_faces, dtype = int)

    return neigh_faces

def apply_source_region_model(fs_avg_src_data, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert,
                              model_vert, model_faces, fs_avg_path, demo = False):
    morphed_epoch_data = [list()]
    morphed_epoch_channels = [list()]
    morphed_region_names = [list()]
    
    for hemisphere in ["lh", "rh"]:
        neigh_faces = get_sphere_faces(fs_avg_path, hemisphere, model_vert, model_faces)
        translation_list = get_translation_list(neigh_faces)
        
        #### Translates from vertex id to the nth-channel, e.g. vertex ids [0, 11, 24, ..., 163825] to model ids [0, 1, 2, ..., 4097]
        hem_data = fs_avg_src_data[:len(np.where(src_fs_avg_valid_lh_vert)[0]), :] if (hemisphere == "lh") else fs_avg_src_data[len(np.where(src_fs_avg_valid_rh_vert)[0]):, :]
        
        regions = read_freesurfer_annotation(fs_avg_path + "label/" + hemisphere + ".aparc.annot")
        for region_name in regions.keys():
            if (region_name == "unknown"):
                continue
            
            #Gets the vertex ids for a specific region
            region_vertices_ids = np.where(src_fs_avg_valid_lh_vert)[0][np.in1d(np.where(src_fs_avg_valid_lh_vert)[0], regions[region_name])]
            region_channel_ids = translation_list[region_vertices_ids]
            if (len(hem_data[region_channel_ids, :]) == 0):
                continue #In case there is no channel within a region, skip it
            
            morphed_epoch_data[-1].append(np.mean(np.abs(hem_data[region_channel_ids, :]), axis = 0))
            morphed_epoch_channels[-1].append(region_channel_ids)
            morphed_region_names[-1].append(region_name)
            
            if (demo):
                plt.figure()
                plt.plot(morphed_epoch_data[-1][-1] - np.mean(morphed_epoch_data[-1][-1]))
                return (None, None, None)
    
    return (morphed_epoch_data, morphed_epoch_channels, morphed_region_names)









