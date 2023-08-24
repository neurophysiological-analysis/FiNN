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


def _get_meg_coil_positions(rec_meta_info, head_to_mri_trans):
    """
    Gets MEG coil spatial information,
    such as coils' individual integration points (rmag), 
    their directions (cosmag), and weights for these integration points. 
    
    Parameters
    ----------
    rec_meta_info : mne.io.read_info
                    MEG scan meta info, obtailable via mne.io.read_info
    head_to_mri_trans : numpy.ndarray, shape(4, 4)
                        Transformation from MEG to MRI space.
               
    Returns
    -------
    meg_chs : dict, ('coil_type', 'rmag', 'cosmag', 'w', 'ch_trans', 'rmag_mri', 'cosmag_mri')
              MEG channel specific information.
    """
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

def _apply_coreg_to_vertices(geom_white_vert, coreg_trans_mat):
    """
    Scales and transforms vertices from MRI space to MEG space.
    
    Parameters
    ----------
    geom_white_vert : numpy.ndarray, shape(n, 3)
                      White matter surface vertices in MRI space.
               
    Returns
    -------
    trans_geom_white_vert : numpy.ndarray, shape(n, 3)
                            Transformed vertices.
    """
    geom_white_vert /= 1000 #Scale from m to mm
    trans_geom_white_vert = np.dot(geom_white_vert, coreg_trans_mat[:3,:3].T) + coreg_trans_mat[:3, 3]
    
    return trans_geom_white_vert

def _update_invalid_vertices(approx_surface, geom_white_vert, valid_geom_vert):
    """
    Updates invalid vertices
    """
    data = geom_white_vert[np.asarray(valid_geom_vert, dtype=bool)]
    inside_check = (approx_surface.find_simplex(data) != -1)
    valid_geom_vert[np.where(valid_geom_vert)[0][~inside_check]] = False
    
    return valid_geom_vert

def _calc_magnetic_fields(white_vertices_mri, bem_trans_ws_in_skull_vert, coreg_trans_mat, pre_fwd_solution):
    """
    Computes infinite medium potentials.
    
    Parameters
    ----------
    white_vertices_mri : numpy.ndarray, shape(m, 3)
                         White matter vertices.
    bem_trans_ws_in_skull_vert : numpy.ndarray, shape(n, 3)
                                 Inner skull model vertices.
    coreg_trans_mat : numpy.ndarray, shape(4, 4)
                      Transformatio matrix from the coregistration.
    pre_fwd_solution : numpy.ndarray, shape(meg_ch_cnt, m * 3)
                       Fwd solution precursor.
               
    Returns
    -------
    fwd_sol : numpy.ndarray, shape(4, meg_ch_cnt, n)
              Forward solution with infinite medium potentials.
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

def _add_current_distribution(fwd_sol, white_vertices, meg_ch_infos, rmags, cosmags, ws, meg_ch_indices):
    """
    Adds current distribution to the MEG forward solution.
    
    Parameters
    ----------
    fwd_sol : numpy.ndarray, shape(number_of_valid_vertices * 3, meg_ch_cnt)
              Previously calcualted (preliminary) forward solution.
    white_vertices : numpy.ndarray, shape(number_of_valid_vertices, 3)
                     Valid white matter vertices.
    meg_ch_infos : list of dictionaries, ('coil_type', 'rmag', 'cosmag', 'w', 'ch_trans', 'rmag_mri', 'cosmag_mri')
                   MEG recording meta into.
    rmags : numpy.ndarray, shape(n, 3)
            3D positions of MEG coil integration points (MEG space).
    cosmags : numpy.ndarray, shape(n, 3)
              Direction of the MEG coil integration points (MEG space).
    ws : Var_type, shape(n,)
         Weights for MEG coil integration points.
    meg_ch_indices : Var_type, shape(n,)
                     Indices of MEG channels.
               
    Returns
    -------
    fwd_sol : numpy.ndarray, shape(number_of_valid_vertices * 3, meg_ch_cnt)
              Forward solution with current distribution.
    """
    pc = np.empty((len(white_vertices) * 3, len(meg_ch_infos)))
    for (vortex_idx, vortex) in enumerate(white_vertices):
        #Calculates the magnetic field at a vortex from all MEG sensors.
        pp = _calc_bem_fields(vortex, rmags, cosmags)
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

def _calc_bem_fields(vortex, rmags, cosmags):
    """
    Calculates the magnetic field at a vortex from all MEG sensors.
    
    Parameters
    ----------
    vortex : numpy.ndarray, shape(3,)
             Position of the vortex
    rmags : numpy.ndarray, shape(n, 3)
            3D positions of MEG coil integration points (MRI space)
    cosmags : numpy.ndarray, shape(n, 3)
              Direction of the MEG coil integration points (MEG space)
               
    Returns
    -------
    fields : numpy.ndarray, shape(1, 3, n)
             The magnetic field of all MEG sensors at a vortex.    
    """
    #See Mosher et al, 1999, equation #1.
    diff = np.expand_dims(rmags.T, axis = 0) - np.expand_dims(vortex, axis = [0, 2])
    norm_diff = np.power(np.linalg.norm(diff, axis = 1), 3)
    norm_diff = norm_diff.squeeze(0)
    norm_diff[norm_diff == 0] = 1
    norm_diff = np.expand_dims(norm_diff, axis = [0, 1])
    
    #See Mosher et al, 1999, equation #19.
    fields = np.empty((1, 3, rmags.shape[0]))
    fields[:, 0] = diff[:, 1] * cosmags[:, 2] - diff[:, 2] * cosmags[:, 1]
    fields[:, 1] = diff[:, 2] * cosmags[:, 0] - diff[:, 0] * cosmags[:, 2]
    fields[:, 2] = diff[:, 0] * cosmags[:, 1] - diff[:, 1] * cosmags[:, 0]
    
    #See Mosher et al, 1999, equation #1.
    fields /= norm_diff
    
    return fields

def calc_forward_model(lh_geom_white_vert, rh_geom_white_vert,
                       head_to_mri_trans, mri_to_head_trans, rec_meta_info,
                       bem_trans_ws_in_skull_vert, bem_ws_in_skull_faces, bem_ws_in_skull_faces_normal, bem_ws_in_skull_faces_area, bem_solution,
                       valid_lh_geom_vert, valid_rh_geom_vert,
                       _mag_factor=1e-7):
    """
    Calculates the forward model according to Mosher et al, 1999. 
    
    Parameters
    ----------
    lh_geom_white_vert : numpy.ndarray, shape(m, 3)
                         Lh white matter surface (MRI).
    rh_geom_white_vert : numpy.ndarray, shape(n, 3)
                         Rh white matter surface (MRI).
    head_to_mri_trans : numpy.ndarray, shape(4, 4)
                        Head to MRI transformation.
    mri_to_head_trans : numpy.ndarray, shape(4, 4)
                        MRI to head transformation.
    rec_meta_info : mne.io.read_info
                    MEG scan meta info, obtailable via mne.io.read_info
    bem_trans_ws_in_skull_vert : numpy.ndarray, shape(x, 3)
                                 Inner skull model vertices.
    bem_ws_in_skull_faces : numpy.ndarray, shape(y, 3)
                            Inner skull model faces.
    bem_ws_in_skull_faces_normal : numpy.ndarray, shape(y, 3)
                                   Inner skull model faces' normals.
    bem_ws_in_skull_faces_area : numpy.ndarray, shape(y,)
                                 Inner skull model faces' areas.
    bem_solution : numpy.ndarray, shape(x, x)
                   BEM linear basis functions.
    valid_lh_geom_vert : numpy.ndarray, shape(m, 3)
                         Binary list of Freesurfer lh vertices that have a match in the model vertices (octahedron).
    valid_rh_geom_vert : numpy.ndarray, shape(n, 3)
                         Binary list of Freesurfer rh vertices that have a match in the model vertices (octahedron).
               
    Returns
    -------
    fwd_sol : Var_type, shape(meg_ch_cnt, remaining_vertices * 3)
              Forward model, transforming from MEG space into valid MRI space.
    valid_lh_geom_vert : Var_type, shape
                         Remaining valid lh vertices.
    valid_rh_geom_vert : Var_type, shape
                         Remaining valid rh vertices.
    """

    #Transform anatomy into head space
    trans_lh_geom_white_vert = _apply_coreg_to_vertices(lh_geom_white_vert, mri_to_head_trans)
    trans_rh_geom_white_vert = _apply_coreg_to_vertices(rh_geom_white_vert, mri_to_head_trans)
    
    #Flag cortical points outside of the skull as invalid and concatenate all vertices
    approx_surface = scipy.spatial.Delaunay(bem_trans_ws_in_skull_vert)
    valid_lh_geom_vert = _update_invalid_vertices(approx_surface, lh_geom_white_vert, valid_lh_geom_vert)
    valid_rh_geom_vert = _update_invalid_vertices(approx_surface, rh_geom_white_vert, valid_rh_geom_vert)
    white_vertices = np.concatenate((trans_lh_geom_white_vert[np.where(valid_lh_geom_vert)[0]], trans_rh_geom_white_vert[np.where(valid_rh_geom_vert)[0]]))
    
    #Configure constants
    conductivity = (.3,)
    sigma = conductivity[0]
    source_mult = 2.0 / ((sigma) + 0)  # Conductivity in the first layer (sigma) and outside the skull (0)
    field_mult = sigma - 0
    
    #Load MEG coil meta information (e.g. direction)
    meg_ch_infos = _get_meg_coil_positions(rec_meta_info, head_to_mri_trans)    
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
    fwd_sol = _calc_magnetic_fields(white_vertices_mri, bem_trans_ws_in_skull_vert, head_to_mri_trans, pre_fwd_solution)
    
    # Calculate primary current distribution
    fwd_sol = _add_current_distribution(fwd_sol, white_vertices, meg_ch_infos, rmags, cosmags, ws, meg_ch_indices)
    
    #Rescaling
    fwd_sol *= _mag_factor
    fwd_sol = fwd_sol.T
    
    return (fwd_sol, valid_lh_geom_vert, valid_rh_geom_vert)
        
def optimize_fwd_model(lh_geom_white_vert, lh_geom_white_faces, valid_geom_lh_vert,
                       rh_geom_white_vert, rh_geom_white_faces, valid_geom_rh_vert,
                       fwd_sol, mri_to_head_trans):
    """
    Transforms a fwd model into surface orientation (orthogonal to the respective surface cluster;
    allowing for cortically constrained inverse modeling)
    and shrinks it by making closeby channels project to the same destination.
    This is different from a 'default' 3D transformation. 
    
    Parameters
    ----------
    lh_geom_white_vert : numpy.ndarray, shape(lh_vert_cnt, 3)
                         Lh vertices.
    lh_geom_white_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                          Lh faces.
    valid_geom_lh_vert : numpy.ndarray, shape(lh_vert_cnt,)
                         Valid/supporting lh vertices.
    rh_geom_white_vert : numpy.ndarray, shape(rh_vert_cnt, 3)
                         Rh vertices.
    rh_geom_white_faces : numpy.ndarray, shape(rh_face_cnt, 3)
                          Rh faces.
    valid_geom_rh_vert : numpy.ndarray, shape(rh_vert_cnt,)
                         Valid/supporting rh vertices.
    fwd_sol : TYPE, shape(meg_ch_cnt, valid_vtx_cnt * 3)
              Forward solution with default orientation.
    
    mri_to_head_trans : numpy.ndarray, shape(4, 4)
                        MRI to head transformation matrix.
    
    Returns
    -------
    fwd_sol * rot : numpy.ndarray, shape(meg_ch_cnt, valid_vtx_cnt)
                    Transformed surface model.
    """
     
    #Computes vertex normals
    lh_acc_normals = _calc_acc_hem_normals(lh_geom_white_vert, lh_geom_white_faces)
    rh_acc_normals = _calc_acc_hem_normals(rh_geom_white_vert, rh_geom_white_faces)
     
    # Figures out which real vertices are represented by the same model vortex
    (lh_cluster_grp, lh_cluster_indices) = _find_vertex_clusters(lh_geom_white_vert, valid_geom_lh_vert, lh_geom_white_faces)
    (rh_cluster_grp, rh_cluster_indices) = _find_vertex_clusters(rh_geom_white_vert, valid_geom_rh_vert, rh_geom_white_faces)
    
    #Determines the orientation 
    lh_orient = finnpy.source_reconstruction.utils.get_eigenbasis(lh_acc_normals, valid_geom_lh_vert, lh_cluster_grp, lh_cluster_indices, mri_to_head_trans)
    rh_orient = finnpy.source_reconstruction.utils.get_eigenbasis(rh_acc_normals, valid_geom_rh_vert, rh_cluster_grp, rh_cluster_indices, mri_to_head_trans)
    
    #Concatenates the eigenvector bases 
    orient = np.concatenate((lh_orient, rh_orient), axis = 0)
     
    #Transforms eigenvector matrix into block matrix for ease of use
    rot = finnpy.source_reconstruction.utils.orient_mat_to_block_format(orient[2::3, :])
    
    #Applies rotation matrix to the fwd solution 
    return fwd_sol * rot
    
def _calc_acc_hem_normals(geom_white_vert, geom_white_faces):
    """
    Generatoes vertex normals from accumulated face normals.
    See https://en.wikipedia.org/wiki/Vertex_normal.
    
    Parameters
    ----------
    geom_white_vert : numpy.ndarray, shape(vert_cnt, 3)
                         MRI model vertices.
    
    geom_white_faces : numpy.ndarray, shape(face_cnt, 3)
                          MRI model faces.
               
    Returns
    -------
    acc_normals : numpy.ndarray, shape(vert_cnt, 3)
                  Array of surface normals for the input faces/vertices.
    """
    
    #Calculate surface normals
    vert0 = geom_white_vert[geom_white_faces[:, 0], :]
    vert1 = geom_white_vert[geom_white_faces[:, 1], :]
    vert2 = geom_white_vert[geom_white_faces[:, 2], :]
    vortex_normals = finnpy.source_reconstruction.utils.fast_cross_product(vert1 - vert0, vert2 - vert1)
    len_vortex_normals = np.linalg.norm(vortex_normals, axis = 1)
    vortex_normals[len_vortex_normals > 0] /= np.expand_dims(len_vortex_normals[len_vortex_normals > 0], axis = 1)
    
    #Accumulate face normals
    acc_normals = np.zeros((geom_white_vert.shape[0], 3))
    for outer_idx in range(3):
        for inner_idx in range(3):
            value = np.zeros((geom_white_vert.shape[0],))
            for vertex_idx in range(geom_white_faces.shape[0]):
                value[geom_white_faces[vertex_idx, outer_idx]] += vortex_normals[vertex_idx, inner_idx]
            acc_normals[:, inner_idx] += value
    
    #Normalize face normals
    len_acc_normals = np.linalg.norm(acc_normals, axis = 1)
    acc_normals[len_acc_normals > 0] /= np.expand_dims(len_acc_normals[len_acc_normals > 0], axis = 1)
    
    return acc_normals
   
def _find_vertex_clusters(geom_white_vert, valid_geom_vert, geom_white_faces):
    """
    Identifies which input vertices (geom_white_vert) are presented by which valid vortex.
    
    Parameters
    ----------
    geom_white_vert : numpy.ndarray, shape(lh_vert_cnt, 3)
                      MRI model vertices.
    valid_geom_vert : numpy.ndarray, shape(lh_vert_cnt,)
                      Valid/supporting vertices.
    geom_white_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                       MRI model faces.
    
    Returns
    -------
    cluster_grp : list, len(n,)
                  Transformed surface model.
    cluster_indices : list, len(n,)
                      Transformed surface model.
    """
    
    #Get adjacency matrix to identify nearest valid vortex
    edges = scipy.sparse.coo_matrix((np.ones(3 * geom_white_faces.shape[0]),
                                        (np.concatenate((geom_white_faces[:, 0], geom_white_faces[:, 1], geom_white_faces[:, 2])),
                                         np.concatenate((geom_white_faces[:, 1], geom_white_faces[:, 2], geom_white_faces[:, 0])))),
                                        shape = (geom_white_vert.shape[0], geom_white_vert.shape[0]))
    edges = edges.tocsr()
    edges += edges.T
    edges = edges.tocoo()
    edges_dists = np.linalg.norm(geom_white_vert[edges.row, :] - geom_white_vert[edges.col, :], axis = 1)
    edges_adjacency = scipy.sparse.csr_matrix((edges_dists, (edges.row, edges.col)), shape = edges.shape)
    _, _, min_idx = scipy.sparse.csgraph.dijkstra(edges_adjacency, indices = np.where(valid_geom_vert)[0], min_only = True, return_predecessors = True)
    
    #Accumulates the clusters
    sort_near_idx = np.argsort(min_idx)
    sort_min_idx = min_idx[sort_near_idx]
    breaks = np.where(sort_min_idx[1:] != sort_min_idx[:-1])[0] + 1
    starts = [0] + breaks.tolist()
    ends = breaks.tolist() + [len(min_idx)]
    cluster_grp = list()
    for cluster_idx in range(len(starts)):
        cluster_grp.append(np.sort(sort_near_idx[starts[cluster_idx]:ends[cluster_idx]]))
    pre_cluster_indices = sort_min_idx[breaks - 1]
    cluster_indices = np.searchsorted(pre_cluster_indices, np.where(valid_geom_vert)[0])
    
    return (cluster_grp, cluster_indices)



        
