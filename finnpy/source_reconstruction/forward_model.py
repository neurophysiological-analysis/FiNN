'''
Created on Oct 12, 2022

@author: voodoocode
'''

import numpy as np
import scipy.spatial
import mpmath

import finnpy.source_reconstruction.utils
import finnpy.source_reconstruction.bem_model

import mne.forward
import mne.io.constants


def get_meg_coil_positions(rec_meta_info, head_to_mri_trans):
    meg_to_head_trans = rec_meta_info['dev_head_t']["trans"]
    coil_types = mne.forward._read_coil_defs(verbose="Error")  # Reads meg coil types from mne library
    meg_chs = list()
    for (_, channel) in enumerate(rec_meta_info["chs"]):
        if (channel["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH):
            
            ch_trans = np.zeros((4, 4)); ch_trans[3, 3] = 1
            ch_trans[:3, 3] = channel["loc"][0:3]
            ch_trans[:3, 0] = channel["loc"][3:6]
            ch_trans[:3, 1] = channel["loc"][6:9]
            ch_trans[:3, 2] = channel["loc"][9:12]
            
            channel["pos"] = ch_trans[0:3]
            channel["pos_rot_scale"] = ch_trans
            
            curr_coil_type = None
            for coil_type in coil_types:
                if ((int(coil_type["coil_type"]) == channel["coil_type"]) and (coil_type["accuracy"] == 2)):
                    curr_coil_type = coil_type
            if (curr_coil_type is None):
                raise AssertionError("Unable to identify meg coil type")
            channel["coil_type_info"] = curr_coil_type
            
            loc_trans = np.dot(meg_to_head_trans, channel["pos_rot_scale"])
                
            meg_ch = {"coil_type":channel["coil_type_info"]["coil_type"],
                      "rmag":np.copy(channel["coil_type_info"]["rmag"]),
                      "cosmag":np.copy(channel["coil_type_info"]["cosmag"]),
                      "w":np.copy(channel["coil_type_info"]["w"]),
                      "ch_trans":ch_trans}
            
            meg_ch["rmag"] = np.dot(meg_ch["rmag"], loc_trans[:3,:3].T) + loc_trans[:3, 3]
            meg_ch["cosmag"] = np.dot(meg_ch["cosmag"], loc_trans[:3,:3].T)
            
            meg_ch["rmag_mri"] = np.dot(meg_ch["rmag"], head_to_mri_trans[:3,:3].T) + head_to_mri_trans[:3, 3]
            meg_ch["cosmag_mri"] = np.dot(meg_ch["cosmag"], head_to_mri_trans[:3,:3].T)
            
            meg_chs.append(meg_ch)
            
    return meg_chs


def apply_coreg_to_vertices(geom_white_vert, coreg_trans_mat):
    geom_white_vert /= 1000
    trans_geom_white_vert = np.dot(geom_white_vert, coreg_trans_mat[:3,:3].T) + coreg_trans_mat[:3, 3]
    
    return trans_geom_white_vert


def update_invalid_vertices(approx_surface, geom_white_vert, valid_geom_vert):
    data = geom_white_vert[np.asarray(valid_geom_vert, dtype=bool)]
    inside_check = (approx_surface.find_simplex(data) != -1)
    valid_geom_vert[np.where(valid_geom_vert)[0][~inside_check]] = False
    
    return valid_geom_vert


def calc_magnetic_fields(white_vertices_mri, bem_trans_ws_in_skull_vert, coreg_trans_mat, pre_fwd_solution):
    fwd_sol = np.zeros((len(white_vertices_mri), 3, len(bem_trans_ws_in_skull_vert)))
    for (white_vortex_mir_idx, white_vortex_mri) in enumerate(white_vertices_mri):
        norm_diff = np.power(np.linalg.norm(bem_trans_ws_in_skull_vert - white_vortex_mri, axis=1), 3)
        norm_diff[norm_diff == 0] = 1
        loc_diff = np.dot((bem_trans_ws_in_skull_vert - white_vortex_mri), coreg_trans_mat[:3,:3])
        loc_diff /= np.expand_dims(norm_diff, axis=1)
        
        fwd_sol[white_vortex_mir_idx] = loc_diff.T
    
    fwd_sol = fwd_sol.reshape(white_vertices_mri.shape[0] * 3, fwd_sol.shape[2])
    fwd_sol = np.dot(fwd_sol, pre_fwd_solution.T)
    
    return fwd_sol


def add_current_distribution(fwd_sol, white_vertices, meg_ch_infos, rmags, cosmags, ws, meg_ch_indices):
    pc = np.empty((len(white_vertices) * 3, len(meg_ch_infos)))
    for (vortex_idx, vortex) in enumerate(white_vertices):
        pp = finnpy.source_reconstruction.bem_model.calc_bem_fields(vortex, rmags, cosmags)
        pp *= ws
        pp = pp.squeeze(0)
        
        tmp = np.zeros((3, len(meg_ch_infos)))
        for idx in range(len(meg_ch_indices)):
            tmp[:, meg_ch_indices[idx]] += pp[:, idx]
        tmp = np.asarray(tmp)
        pc[int(3 * vortex_idx):int(3 * (vortex_idx + 1)),:] = tmp
    
    fwd_sol += pc
    
    return fwd_sol


def calc_forward_model(lh_geom_white_vert, rh_geom_white_vert,
                       head_to_mri_trans, mri_to_head_trans, rec_meta_info,
                       bem_trans_ws_in_skull_vert, bem_ws_in_skull_faces, bem_ws_in_skull_faces_normal, bem_ws_in_skull_faces_area, bem_solution,
                       valid_lh_geom_vert, valid_rh_geom_vert,
                       _mag_factor=1e-7):

    trans_lh_geom_white_vert = apply_coreg_to_vertices(lh_geom_white_vert, mri_to_head_trans)
    trans_rh_geom_white_vert = apply_coreg_to_vertices(rh_geom_white_vert, mri_to_head_trans)
    
    # Flag cortical points outside of the skull as invalid
    approx_surface = scipy.spatial.Delaunay(bem_trans_ws_in_skull_vert)
    valid_lh_geom_vert = update_invalid_vertices(approx_surface, lh_geom_white_vert, valid_lh_geom_vert)
    valid_rh_geom_vert = update_invalid_vertices(approx_surface, rh_geom_white_vert, valid_rh_geom_vert)
    
    white_vertices = np.concatenate((trans_lh_geom_white_vert[np.where(valid_lh_geom_vert)[0]], trans_rh_geom_white_vert[np.where(valid_rh_geom_vert)[0]]))
    
    conductivity = (.3,)
    sigma = conductivity[0]
    source_mult = 2.0 / ((sigma) + 0)  # Conductivity in the first layer (sigma) and outside the skull (0)
    field_mult = sigma - 0
    
    meg_ch_infos = get_meg_coil_positions(rec_meta_info, head_to_mri_trans)
    
    meg_ch_comp = np.zeros((len(rec_meta_info["chs"],)))
    for (meg_ch_idx, meg_ch) in enumerate(meg_ch_infos):
        meg_ch_comp[meg_ch_idx] = int(meg_ch["coil_type"]) >> 16
    if (len(np.unique(meg_ch_comp)) != 1):
        raise AssertionError("Unequal compensation of channels")
    meg_ch_comp = np.unique(meg_ch_comp)[0]
    if (meg_ch_comp != 0):
        raise NotImplementedError("Compensation not yet implemented.")
    
    # cosmag contains the direction of the coils and rmag contains the. position vector
    rmags = list(); rmags_mri = list(); cosmags = list(); cosmags_mri = list(); ws = list(); meg_ch_indices = list() 
    for (meg_ch_info_idx, _) in enumerate(meg_ch_infos):
        rmags.extend(meg_ch_infos[meg_ch_info_idx]["rmag"])
        rmags_mri.extend(meg_ch_infos[meg_ch_info_idx]["rmag_mri"])
        cosmags.extend(meg_ch_infos[meg_ch_info_idx]["cosmag"])
        cosmags_mri.extend(meg_ch_infos[meg_ch_info_idx]["cosmag_mri"])
        ws.extend(meg_ch_infos[meg_ch_info_idx]["w"])
        meg_ch_indices.extend(np.ones((meg_ch_infos[meg_ch_info_idx]["w"].shape[0],)) * meg_ch_info_idx)
    rmags = np.asarray(rmags); rmags_mri = np.asarray(rmags_mri)
    cosmags = np.asarray(cosmags); cosmags_mri = np.asarray(cosmags_mri)
    ws = np.asarray(ws); meg_ch_indices = np.asarray(meg_ch_indices, dtype=int)
    
    #===========================================================================
    # For reference
    # 
    # See Mosher et al. 1999, 
    # equations 3, 42, and Table 1 equation 42
    #
    # and
    # Lewis 1995, NEUROMAGNETIC SOURCE RECONSTRUCTION
    #===========================================================================
    
    meg_coeff = np.zeros((len(meg_ch_infos), bem_trans_ws_in_skull_vert.shape[0]))
    w_cosmags = np.expand_dims(ws, axis=1) * cosmags_mri
    tau = np.expand_dims(rmags_mri, axis=1) - bem_trans_ws_in_skull_vert
    den = np.sum(tau * tau, axis=2)
    den *= np.sqrt(den) * 3
    
    for (face_idx, face) in enumerate(bem_ws_in_skull_faces):
        num = np.cross(tau[:, face], bem_ws_in_skull_faces_normal[face_idx,:]) * np.expand_dims(w_cosmags, axis=1)
        for vert_idx in range(3):
            loc_effects = np.sum(num[:, vert_idx], axis=1) * bem_ws_in_skull_faces_area[face_idx] / den[:, face[vert_idx]]
            effects_on_vert = np.zeros((len(meg_ch_infos),))
            for (effect_idx, effect_value) in enumerate(loc_effects):
                effects_on_vert[meg_ch_indices[effect_idx]] += effect_value
            meg_coeff[:, face[vert_idx]] += effects_on_vert
    meg_coeff *= field_mult
    
    pre_fwd_solution = np.dot(meg_coeff, bem_solution) * source_mult / (4 * np.pi)
    
    white_vertices_mri = np.dot(head_to_mri_trans[:3,:3], white_vertices.T).T + head_to_mri_trans[:3, 3]
    
    # Calculate magnetic potentials/fields
    fwd_sol = calc_magnetic_fields(white_vertices_mri, bem_trans_ws_in_skull_vert, head_to_mri_trans, pre_fwd_solution)
    
    # Calculate primary current distribution
    fwd_sol = add_current_distribution(fwd_sol, white_vertices, meg_ch_infos, rmags, cosmags, ws, meg_ch_indices)
    
    fwd_sol *= _mag_factor
    fwd_sol = fwd_sol.T
    
    return (fwd_sol,
            valid_lh_geom_vert, valid_rh_geom_vert)
        
        
