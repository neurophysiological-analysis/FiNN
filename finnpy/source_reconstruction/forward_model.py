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
    geom_white_vert /= 1000 #Scale from m to mm
    trans_geom_white_vert = np.dot(geom_white_vert, coreg_trans_mat[:3,:3].T) + coreg_trans_mat[:3, 3]
    
    return trans_geom_white_vert

def update_invalid_vertices(approx_surface, geom_white_vert, valid_geom_vert):
    data = geom_white_vert[np.asarray(valid_geom_vert, dtype=bool)]
    inside_check = (approx_surface.find_simplex(data) != -1)
    valid_geom_vert[np.where(valid_geom_vert)[0][~inside_check]] = False
    
    return valid_geom_vert

def calc_magnetic_fields(white_vertices_mri, bem_trans_ws_in_skull_vert, coreg_trans_mat, pre_fwd_solution):
    """
    Computes infinite medium potentials.
    
    :param white_vertices_mri: White matter vertices.
    :param bem_trans_ws_in_skull_vert: Inner skull model vertices.
    :param coreg_trans_mat: Transformatio matrix from the coregistration.
    :param pre_fwd_solution: Fwd solution precursor.
    
    :return: Forward solution with infinite medium potentials.    
    """
    fwd_sol = np.zeros((len(white_vertices_mri), 3, len(bem_trans_ws_in_skull_vert)))
    
    #See Mosher et al, 1999, equation #8 for reference
    for (white_vortex_mri_idx, white_vortex_mri) in enumerate(white_vertices_mri):
        magnitude = np.power(np.linalg.norm(bem_trans_ws_in_skull_vert - white_vortex_mri, axis=1), 3)
        magnitude[magnitude == 0] = 1
        
        #Move diff from MRI space to head space
        diff = np.dot((bem_trans_ws_in_skull_vert - white_vortex_mri), coreg_trans_mat[:3,:3])
        diff /= np.expand_dims(magnitude, axis=1)
        
        fwd_sol[white_vortex_mri_idx] = diff.T
    
    #Multiply with current dipole moments
    fwd_sol = fwd_sol.reshape(white_vertices_mri.shape[0] * 3, fwd_sol.shape[2])
    fwd_sol = np.dot(fwd_sol, pre_fwd_solution.T)
    
    return fwd_sol

def add_current_distribution(fwd_sol, white_vertices, meg_ch_infos, rmags, cosmags, ws, meg_ch_indices):
    """
    Adds current distribution to the MEG forward solution.
    
    :param fwd_sol: Previously calcualted forward solution.
    :param white_vertices: White matter vertices.
    :param meg_ch_infos: MEG recording meta into.
    :param rmags: 3D positions of MEG coil integration points (MEG space).
    :param cosmags: Direction of the MEG coil integration points (MEG space).
    :param ws: Weights for MEG coil integration points.
    :param meg_ch_indices: Indices of MEG channels.
    
    :return: Forward solution updated for primary currend distributions.
    """
    pc = np.empty((len(white_vertices) * 3, len(meg_ch_infos)))
    for (vortex_idx, vortex) in enumerate(white_vertices):
        #Calculates the magnetic field at a vortex from all MEG sensors.
        pp = finnpy.source_reconstruction.bem_model.calc_bem_fields(vortex, rmags, cosmags)
        pp *= ws # Adds MEG coil weight
        pp = pp.squeeze(0)
        
        #Attribute primary current spread to respective MEG channels.
        tmp = np.zeros((3, len(meg_ch_infos)))
        for idx in range(len(meg_ch_indices)):
            tmp[:, meg_ch_indices[idx]] += pp[:, idx]
        tmp = np.asarray(tmp)
        pc[int(3 * vortex_idx):int(3 * (vortex_idx + 1)),:] = tmp
    
    #Add primary current spread to MEG channels.
    fwd_sol += pc
    
    return fwd_sol


def calc_forward_model(lh_geom_white_vert, rh_geom_white_vert,
                       head_to_mri_trans, mri_to_head_trans, rec_meta_info,
                       bem_trans_ws_in_skull_vert, bem_ws_in_skull_faces, bem_ws_in_skull_faces_normal, bem_ws_in_skull_faces_area, bem_solution,
                       valid_lh_geom_vert, valid_rh_geom_vert,
                       _mag_factor=1e-7):
    """
    Calculates the forward model according to Mosher et al, 1999. 
    
    :param lh_geom_white_vert: Lh white matter surface (MRI).
    :param rh_geom_white_vert: Rh white matter surface (MRI).
    :param head_to_mri_trans: Head to MRI transformation.
    :param mri_to_head_trans: MRI to head transformation.
    :param rec_meta_info: MEG recording meta info.
    :param bem_trans_ws_in_skull_vert: Inner skull model vertices.
    :param bem_ws_in_skull_faces: Inner skull model faces.
    :param bem_ws_in_skull_faces_normal: Inner skull model faces' normals.
    :param bem_ws_in_skull_faces_area: Inner skull model faces' areas.
    :param bem_solution: BEM linear basis functions.
    :param valid_lh_geom_vert: Binary list of Freesurfer lh vertices that have a match in the model vertices (octahedron).
    :param valid_rh_geom_vert: Binary list of Freesurfer rh vertices that have a match in the model vertices (octahedron).
    
    :return: Forward model and checked vertices (lh & rh) for being within the skull.
    """

    #Transform anatomy into head space
    trans_lh_geom_white_vert = apply_coreg_to_vertices(lh_geom_white_vert, mri_to_head_trans)
    trans_rh_geom_white_vert = apply_coreg_to_vertices(rh_geom_white_vert, mri_to_head_trans)
    
    #Flag cortical points outside of the skull as invalid and concatenat all vertices
    approx_surface = scipy.spatial.Delaunay(bem_trans_ws_in_skull_vert)
    valid_lh_geom_vert = update_invalid_vertices(approx_surface, lh_geom_white_vert, valid_lh_geom_vert)
    valid_rh_geom_vert = update_invalid_vertices(approx_surface, rh_geom_white_vert, valid_rh_geom_vert)
    white_vertices = np.concatenate((trans_lh_geom_white_vert[np.where(valid_lh_geom_vert)[0]], trans_rh_geom_white_vert[np.where(valid_rh_geom_vert)[0]]))
    
    #Configure constants
    conductivity = (.3,)
    sigma = conductivity[0]
    source_mult = 2.0 / ((sigma) + 0)  # Conductivity in the first layer (sigma) and outside the skull (0)
    field_mult = sigma - 0
    
    #Load MEG coil meta information (e.g. direction)
    meg_ch_infos = get_meg_coil_positions(rec_meta_info, head_to_mri_trans)    
    meg_ch_comp = np.zeros((len(rec_meta_info["chs"],)))
    for (meg_ch_idx, meg_ch) in enumerate(meg_ch_infos):
        meg_ch_comp[meg_ch_idx] = int(meg_ch["coil_type"]) >> 16
    if (len(np.unique(meg_ch_comp)) != 1):
        raise AssertionError("Unequal compensation of channels")
    meg_ch_comp = np.unique(meg_ch_comp)[0]
    if (meg_ch_comp != 0):
        raise NotImplementedError("Compensation not yet implemented.")
    
    #Cosmag contains the direction of the coils and rmag contains the position vector
    rmags = list(); rmags_mri = list(); cosmags = list(); cosmags_mri = list(); ws = list(); meg_ch_indices = list() 
    for (meg_ch_info_idx, _) in enumerate(meg_ch_infos):
        rmags.extend(meg_ch_infos[meg_ch_info_idx]["rmag"]) # 3D positions of MEG coil integration points (MEG space)
        rmags_mri.extend(meg_ch_infos[meg_ch_info_idx]["rmag_mri"]) # 3D positions of MEG coil integration points (MRI space)
        cosmags.extend(meg_ch_infos[meg_ch_info_idx]["cosmag"]) # Direction of the MEG coil integration points (MEG space)
        cosmags_mri.extend(meg_ch_infos[meg_ch_info_idx]["cosmag_mri"]) # Direction of the MEG coil integration points (MRI space)
        ws.extend(meg_ch_infos[meg_ch_info_idx]["w"]) # Weights for MEG coil integration points
        meg_ch_indices.extend(np.ones((meg_ch_infos[meg_ch_info_idx]["w"].shape[0],)) * meg_ch_info_idx)
    rmags = np.asarray(rmags); rmags_mri = np.asarray(rmags_mri)
    cosmags = np.asarray(cosmags); cosmags_mri = np.asarray(cosmags_mri)
    ws = np.asarray(ws); meg_ch_indices = np.asarray(meg_ch_indices, dtype=int)
    
    #Calculates the contribution of each current source (BEM normals) towards to each MEG sensor. 
    #Based on Mosher, 1999, formula #3 µ/(4*pi)(normals x tau)/norm³ ...
    #extended with additional weights for MEG coil direction, MEG coil weight
    #Integration simplified as multiplication with surface area, see "Multiple Interface Brain and Head Models for EEG: A Surface Charge Approach", Solis and Papandreou-Suppappola
    meg_lin_pot_basis = np.zeros((len(meg_ch_infos), bem_trans_ws_in_skull_vert.shape[0]))
    w_cosmags = np.expand_dims(ws, axis=1) * cosmags_mri # Weighted directions
    tau = np.expand_dims(rmags_mri, axis=1) - bem_trans_ws_in_skull_vert
    den = np.sum(tau * tau, axis=2) # This equates to ||x||^3, from sqrt(x¹+x²+...+x^n)³ = z³ to (x¹+x²+...+x^n) * sqrt((x¹+x²+...+x^n)) = z * z^(0.5) = z^(1 + 0.5)  
    den *= np.sqrt(den) * 3
    for (face_idx, face) in enumerate(bem_ws_in_skull_faces):
        num = np.cross(tau[:, face], bem_ws_in_skull_faces_normal[face_idx, :]) * np.expand_dims(w_cosmags, axis=1)
        for vert_idx in range(3):
            loc_effects = np.sum(num[:, vert_idx], axis=1) * bem_ws_in_skull_faces_area[face_idx] / den[:, face[vert_idx]]
            effects_on_vert = np.zeros((len(meg_ch_infos),))
            for (effect_idx, effect_value) in enumerate(loc_effects):
                effects_on_vert[meg_ch_indices[effect_idx]] += effect_value
            meg_lin_pot_basis[:, face[vert_idx]] += effects_on_vert
    meg_lin_pot_basis *= field_mult
    pre_fwd_solution = np.dot(meg_lin_pot_basis, bem_solution) * source_mult / (4 * np.pi)
    
    #Transform vertices into MRI space
    white_vertices_mri = np.dot(head_to_mri_trans[:3,:3], white_vertices.T).T + head_to_mri_trans[:3, 3]
    
    # Calculate magnetic potentials/fields
    fwd_sol = calc_magnetic_fields(white_vertices_mri, bem_trans_ws_in_skull_vert, head_to_mri_trans, pre_fwd_solution)
    
    # Calculate primary current distribution
    fwd_sol = add_current_distribution(fwd_sol, white_vertices, meg_ch_infos, rmags, cosmags, ws, meg_ch_indices)
    
    #Rescaling
    fwd_sol *= _mag_factor
    fwd_sol = fwd_sol.T
    
    return (fwd_sol, valid_lh_geom_vert, valid_rh_geom_vert)
        
        
