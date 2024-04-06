'''
Created on Oct 17, 2022

@author: voodoocode
'''

import numpy as np
import nibabel.freesurfer
import finnpy.src_rec.utils

def _read_freesurfer_annotation(path):
    """
    Averages source space regions to areas defined by the Desikan-Killiany atlas.
    
    Parameters
    ----------
    path : string
           Path to the annotation file.
               
    Returns
    -------
    regions : dict, {regions : [vtx_ids]}
              Dictionary of regions containing vortex ids.
    """
    (annot, _, labels) = nibabel.freesurfer.read_annot(path)
    np.where(annot == 1)
    regions = dict()
    
    for (region_idx, region_name) in enumerate(labels):
        regions[region_name.decode("utf-8")] = np.where(annot == region_idx)[0]
    
    return regions

def _get_sphere_faces(fs_avg_path, hemisphere, octa_model_vert, octa_model_faces):
    """
    Mimics the original downscaling as this is later needed for region averaging, 
    yet, this time in reference to fs-average instead of subject specific.
    
    Parameters
    ----------
    fs_avg_path : string
                  Path to fs average's freesurfer  files.
    hemisphere : string
                 hemisphere to be operated on.
    octa_model_vert : numpy.ndarray, shape(octa_vtx_cnt, 3)
                      Octahedron model vertices.
    octa_model_faces : numpy.ndarray, shape(octa_face_cnt, 3)
                       Octahedron model faces.
               
    Returns
    -------
    neigh_faces : numpy.ndarray, shape(octa_face_cnt, 3)
                  Faces most closeby to model faces.
    """
    (fs_avg_surf_sph_vert, _) = nibabel.freesurfer.read_geometry(fs_avg_path + "surf/" + hemisphere + ".sphere")
    neigh_indices = finnpy.src_rec.utils.find_nearest_neighbor(fs_avg_surf_sph_vert/100, octa_model_vert)[0]
    neigh_faces = [neigh_indices[face] for face in octa_model_faces]
    neigh_faces = np.asarray(neigh_faces, dtype = int)

    return neigh_faces

def _get_mri_to_model_trans(mri_neigh_faces):
    """
    Limits MRI vertices by employing the reduced face list to reduce vortex count. 
    
    Parameters
    ----------
    mri_neigh_faces : numpy.ndarray, shape(face_cnt, 3)
                      Faces most closeby to model faces.
               
    Returns
    -------
    mri_to_model_trans : numpy.ndarray, shape(vtx_cnt, )
                         Region names.
    """
    reshaped_mri_neigh_faces = mri_neigh_faces.reshape(-1)
    (_, unique_reshaped_mri_neigh_face_indices) = np.unique(reshaped_mri_neigh_faces, return_index = True)
    mri_to_model_trans = np.ones((np.max(reshaped_mri_neigh_faces) + 1)) * np.nan #Theoretically the size of this is vtx_cnt, however, the highest id of all valid vertices suffices.
    for (idx, value) in enumerate(unique_reshaped_mri_neigh_face_indices):
        mri_to_model_trans[reshaped_mri_neigh_faces[value]] = idx
    mri_to_model_trans = np.nan_to_num(mri_to_model_trans, nan = -1)
    mri_to_model_trans = np.asarray(mri_to_model_trans, dtype = int)
    mri_to_model_trans[np.argwhere(mri_to_model_trans == -1).squeeze(1)] = np.iinfo(int).min #There is no nan for integer typed arrays, however "-int max" should raise an error in every conceivable situation.
    
    return mri_to_model_trans

def run(src_data, subj_to_fsavg_mdl, fs_path):
    """
    Averages source space regions to areas defined by the Desikan-Killiany atlas.
    
    Parameters
    ----------
    src_data : numpy.ndarray, shape(sensor_ch_cnt, samp_cnt)
               Source space data.
    
    subj_to_fsavg_mdl : finnpy.src_rec.subj_to_fsavg.Subj_to_fsavg_mdl
                        Container class, populated with the following items:
    
                        trans : numpy.ndarray, shape(valid_subj_vtx_cnt, valid_subj_vtx_cnt)
                                Transformation matrix
                        lh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                        Valid/supporting vertices for left hemisphere.
                        rh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                        Valid/supporting vertices for right hemisphere.
    fs_path : string
              Path to the freesurfer folder. Should contain the 'bin' folder, your license.txt, and sources.sh.
               
    Returns
    -------
    morphed_epoch_data : numpy.ndarray, shape(src_region_cnt, epoch_cnt, samples)
                         Average source space data.
    morphed_epoch_channels : list of lists, len(source_ch_cnt,)
                             List of channel ids, clustered by source space region.
                             Identifies which channel supports which region.
    morphed_region_names : list, len(source_ch_cnt,)
                           Region names.
    """
    
    fs_avg_path  = (fs_path + "/") if (fs_path[-1] != "/") else fs_path
    fs_avg_path += "subjects/fsaverage/"
    
    morphed_epoch_data = list()
    morphed_epoch_channels = list()
    morphed_region_names = list()
    
    for hemisphere in ["lh", "rh"]:
        mri_neigh_faces = _get_sphere_faces(fs_avg_path, hemisphere, subj_to_fsavg_mdl.octa_mdl_vert, subj_to_fsavg_mdl.octa_mdl_faces)
        mri_to_model_trans = _get_mri_to_model_trans(mri_neigh_faces)
        
        if (hemisphere == "lh"):
            loc_valid_vert = subj_to_fsavg_mdl.lh_valid_vert
            #### Translates from vertex id to the nth-channel, e.g. vertex ids [0, 11, 24, ..., 163825] to model ids [0, 1, 2, ..., 4097]
            hem_data = src_data[:len(np.where(subj_to_fsavg_mdl.lh_valid_vert)[0]), :]
        else:
            loc_valid_vert = subj_to_fsavg_mdl.rh_valid_vert
            hem_data = src_data[len(np.where(subj_to_fsavg_mdl.rh_valid_vert)[0]):, :]
        
        regions = _read_freesurfer_annotation(fs_avg_path + "label/" + hemisphere + ".aparc.annot")
        for region_name in regions.keys():
            if (region_name == "unknown"):
                continue
            
            #Gets the vertex ids for a specific region
            mri_region_vertices_ids = np.where(loc_valid_vert)[0][np.in1d(np.where(loc_valid_vert)[0], regions[region_name])]
            model_region_vertices_ids = mri_to_model_trans[mri_region_vertices_ids]
            if (len(hem_data[model_region_vertices_ids, :]) == 0):
                continue #In case there is no channel within a region, skip it
            
            morphed_epoch_data.append(np.mean(np.abs(hem_data[model_region_vertices_ids, :]), axis = 0))
            #This, hem_data[model_region_vertices_ids, :], works as the original downscaled spherical data is an inflated version of the surface data.
            morphed_epoch_channels.append(model_region_vertices_ids)
            morphed_region_names.append(hemisphere + "_" + region_name)
    
    morphed_epoch_data = np.asarray(morphed_epoch_data)
    return (morphed_epoch_data, morphed_epoch_channels, morphed_region_names)









