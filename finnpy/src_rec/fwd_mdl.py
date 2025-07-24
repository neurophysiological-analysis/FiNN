"""
Created on Oct 12, 2022.

@author: voodoocode
"""

import numpy as np
import mne.forward
import mne.io.constants
import scipy.sparse

import finnpy.src_rec.utils  # @UnresolvedImport

def compute(bem_mdl, cort_mdl, coreg, signal_type, rec_info = None, _mag_factor=1e-7, conductivity = None):
    """
    Calculate the forward model according to Mosher et al, 1999.
    
    Parameters
    ----------    
    bem_mdl : finnpy.src_rec.bem_mdl.BEM_mdl
              Container class, populed with the following items:
              
              - vert : numpy.ndarray, shape(scaled_vtx_cnt, 3)
                       Remaining Vertices of a skin/skull model vertices.
              - faces : numpy.ndarray, shape(scaled_face_cnt, 3)
                        Remaining Vertices of a skin/skull model faces.
              - faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                               Normals of the individual remaining inner skull faces.
              - faces_area : numpy.ndarray, shape(scaled_face_cnt)
                             Surface area of the remaining faces.
              - bem_solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                               BEM solution (preliminary step for the calculation of the forward model).
    cort_mdl : finnpy.src_rec.cort_mdl.Cort_mdl
               Container populated with the following items:
               
               - lh_vert : numpy.ndarray, shape(lh_vtx_cnt, 3)
                                 White matter surface model vertices (left hemisphere).
               - lh_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                                  White matter surface model faces (left hemisphere).
               - rh_vert : numpy.ndarray, shape(rh_vtx_cnt, 3)
                                 White matter surface model vertices (right hemisphere).
               - rh_faces : numpy.ndarray, shape(rh_face_cnt, 3)
                                  White matter surface model faces (right hemisphere).
               - lh_sphere_vert : numpy.ndarray, shape(lh_sphere_vtx_cnt, 3)
                                  Spherical freesurfer head model vertices (left hemisphere).
               - rh_sphere_vert : numpy.ndarray, shape(rh_sphere_vtx_cnt, 3)
                                  Spherical freesurfer head model vertices (right hemisphere).
    coreg : finnpy.src_rec.coreg.Coreg
            Container class, populed with the following items:
             
            - Used in EEG:
                - closest_pts_nas : numpy.ndarray, shape(3, )
                                    Closest points to the nas on the MRI.
                - closest_pts_lpa : numpy.ndarray, shape(3, )
                              Closest points to the lpa on the MRI.
                - closest_pts_rpa : numpy.ndarray, shape(3, )
                              Closest points to the rpa on the MRI.
                - closest_pts_eeg : numpy.ndarray, shape(eeg_ch_cnt, )
                              Closest points to the eeg positions on the MRI.
            - Used in MEG:
                - rotors : numpy.ndarray, shape(9,)
                     Sequence of rotors defining rotation (3), translation (3) and scaling (3).
                - mri_to_meeg_trs : numpy.ndarray, shape(4, 4)
                                    Full affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_tr : numpy.ndarray, shape(4, 4)
                                   Rigid affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_rs : numpy.ndarray, shape(4, 4)
                                   Rotation & scaling only affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_r : numpy.ndarray, shape(4, 4)
                                  Rotation only affine transformation matrix (MRI -> MEEG)
                - meeg_to_mri_trs : numpy.ndarray, shape(4, 4)
                                    Full affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_tr : numpy.ndarray, shape(4, 4)
                                   Rigid affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_rs : numpy.ndarray, shape(4, 4)
                                   Rotation & scaling only affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_r : numpy.ndarray, shape(4, 4)
                                  Rotation only affine transformation matrix (MEEG -> MRI)
    signal_type : string
                  Either 'EEG' or 'MEG'.
    rec_info : string if signal_type == "MEG" else tuple(string, list(string))  
               If MEG, path to the fif.file
               If EEG, tuple of EEG setup name (either 1010 or 1020) and a list of valid eeg channel names.
    _mag_factor : float
                  MEG scaling factor.
    conductivity : np.ndarray, shape(3, ) or shape(1, ) or None
                   Conductivity values for the individual surfaces. If None, defaults to (.3) for MEG and (.3, .006, .3) for EEG.
               
    Returns
    -------
    fwd_sol : Var_type, shape(meeg_ch_cnt, valid_vtx_cnt * 3)
              Forward model, projecting from sen into src space.
    """
    if (conductivity is None):
        conductivity = (.3,) if (len(bem_mdl.vert) == 1) else (.3, .006, .3)

    lh_vert_mri = cort_mdl.lh_vert
    rh_vert_mri = cort_mdl.rh_vert
    
    # Transform anatomy into MEEG space
    if (signal_type == "MEG"):
        lh_vert_meeg = np.dot(lh_vert_mri, coreg.mri_to_meeg_tr[:3, :3].T) + coreg.mri_to_meeg_tr[:3, 3]
        rh_vert_meeg = np.dot(rh_vert_mri, coreg.mri_to_meeg_tr[:3, :3].T) + coreg.mri_to_meeg_tr[:3, 3]
    
    # Concatenate all vertices
    vertices_mri = np.concatenate((lh_vert_mri[np.where(cort_mdl.lh_valid_vert)[0]], rh_vert_mri[np.where(cort_mdl.rh_valid_vert)[0]]))
    if (signal_type == "MEG"):
        vertices_meeg = np.concatenate((lh_vert_meeg[np.where(cort_mdl.lh_valid_vert)[0]], rh_vert_meeg[np.where(cort_mdl.rh_valid_vert)[0]]))
    
    if (signal_type == "MEG"):
        # Load MEG coil meta information (e.g. direction)
        (sen_locs_meg, sen_dirs_meg, sen_locs_mri, sen_dirs_mri, ws, meg_ch_indices, meg_ch_cnt) = get_meg_coil_info(rec_info, coreg)
        
        # Calculates the contribution of each current source (BEM normals) towards to each MEG sensor. 
        src_sen_contrib_mri = _comp_meg_src_sen_contrib(bem_mdl, sen_locs_mri, sen_dirs_mri, ws, meg_ch_indices, meg_ch_cnt, conductivity)
            
        # Calculate magnetic potentials/fields
        fwd_sol_meeg = _calc_magnetic_fields(vertices_mri, bem_mdl.vert[0], coreg.meeg_to_mri_tr, src_sen_contrib_mri)
        
        # Calculate primary current distribution
        fwd_sol_meeg = _add_current_distribution(fwd_sol_meeg, vertices_meeg, meg_ch_cnt, sen_locs_meg, sen_dirs_meg, ws, meg_ch_indices)
        
        # Rescaling
        fwd_sol_meeg *= _mag_factor
    elif (signal_type == "EEG"):
        (sen_locs_mri, ws) = get_eeg_sen_info(rec_info, coreg)
        # Calculates the contribution of each current source (BEM normals) towards to each EEG sensor.
        
        src_sen_contrib_mri = _comp_eeg_src_sen_contrib(bem_mdl, sen_locs_mri, ws, conductivity)
            
        # Calculate magnetic potentials/fields
        fwd_sol_meeg = _calc_magnetic_fields(vertices_mri, np.concatenate(bem_mdl.vert), np.eye(4), src_sen_contrib_mri)
            
    fwd_sol_meeg = fwd_sol_meeg.T    
    return fwd_sol_meeg

def get_eeg_sen_info(rec_info, coreg):
    """
    Return EEG points (and weights) rotated into MRI space.
    
    Parameters
    ----------
    rec_info : tuple of (string, list)
               - eeg_system : string
                              The used EEG system, either '1020 or 1005'.
               - ch_names : list, len(eeg_ch_cnt)
                            List of EEG channel names
    coreg : finnpy.src_rec.coreg.Coreg
            Container class, populed with the following items:
             
            - Used in EEG:
                - closest_pts_nas : numpy.ndarray, shape(3, )
                                    Closest points to the nas on the MRI.
                - closest_pts_lpa : numpy.ndarray, shape(3, )
                              Closest points to the lpa on the MRI.
                - closest_pts_rpa : numpy.ndarray, shape(3, )
                              Closest points to the rpa on the MRI.
                - closest_pts_eeg : numpy.ndarray, shape(eeg_ch_cnt, )
                              Closest points to the eeg positions on the MRI.
            - Used in MEG:
                - rotors : numpy.ndarray, shape(9,)
                     Sequence of rotors defining rotation (3), translation (3) and scaling (3).
                - mri_to_meeg_trs : numpy.ndarray, shape(4, 4)
                                    Full affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_tr : numpy.ndarray, shape(4, 4)
                                   Rigid affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_rs : numpy.ndarray, shape(4, 4)
                                   Rotation & scaling only affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_r : numpy.ndarray, shape(4, 4)
                                  Rotation only affine transformation matrix (MRI -> MEEG)
                - meeg_to_mri_trs : numpy.ndarray, shape(4, 4)
                                    Full affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_tr : numpy.ndarray, shape(4, 4)
                                   Rigid affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_rs : numpy.ndarray, shape(4, 4)
                                   Rotation & scaling only affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_r : numpy.ndarray, shape(4, 4)
                                  Rotation only affine transformation matrix (MEEG -> MRI)
    
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - eeg_coords : np.ndarray, shape(eeg_ch_cnt, 3)
                            Coordinates of the EEG sensors.
             - ws : np.ndarray, shape(eeg_ch_cnt, 1)
                    Weights of the EEG sensors.
    
    
    """
    eeg_setup = rec_info[0]
    valid_ch_names = [valid_ch_name.lower() for valid_ch_name in rec_info[1]]
    
    sen_ref_pts = finnpy.src_rec.utils.read_eeg_pts(eeg_setup)
    ref_ch_names = np.asarray([ref_ch_name.lower() for ref_ch_name in sen_ref_pts["labels"]])
    
    ref_ids = np.empty((len(valid_ch_names)), dtype = int)
    for (valid_ch_idx, valid_ch_name) in enumerate(valid_ch_names):
        ref_ids[valid_ch_idx] = np.argwhere(valid_ch_name == ref_ch_names).squeeze(1)
        
    eeg_coords = coreg.closest_pts_eeg[ref_ids, :]
    ws = np.ones((len(eeg_coords)))

    return (eeg_coords, ws)

def get_meg_coil_info(rec_info, coreg):
    """
    Accumulates spatial information on MEG sensor position.
    
    Parameters
    ----------
    rec_info : string
               Path to the fif.file.
    
    coreg : finnpy.src_rec.coreg.Coreg
            Container class, populed with the following items:
             
            - Used in EEG:
                - closest_pts_nas : numpy.ndarray, shape(3, )
                                    Closest points to the nas on the MRI.
                - closest_pts_lpa : numpy.ndarray, shape(3, )
                              Closest points to the lpa on the MRI.
                - closest_pts_rpa : numpy.ndarray, shape(3, )
                              Closest points to the rpa on the MRI.
                - closest_pts_eeg : numpy.ndarray, shape(eeg_ch_cnt, )
                              Closest points to the eeg positions on the MRI.
            - Used in MEG:
                - rotors : numpy.ndarray, shape(9,)
                     Sequence of rotors defining rotation (3), translation (3) and scaling (3).
                - mri_to_meeg_trs : numpy.ndarray, shape(4, 4)
                                    Full affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_tr : numpy.ndarray, shape(4, 4)
                                   Rigid affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_rs : numpy.ndarray, shape(4, 4)
                                   Rotation & scaling only affine transformation matrix (MRI -> MEEG)
                - mri_to_meeg_r : numpy.ndarray, shape(4, 4)
                                  Rotation only affine transformation matrix (MRI -> MEEG)
                - meeg_to_mri_trs : numpy.ndarray, shape(4, 4)
                                    Full affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_tr : numpy.ndarray, shape(4, 4)
                                   Rigid affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_rs : numpy.ndarray, shape(4, 4)
                                   Rotation & scaling only affine transformation matrix (MEEG -> MRI)
                - meeg_to_mri_r : numpy.ndarray, shape(4, 4)
                                  Rotation only affine transformation matrix (MEEG -> MRI)
    
    Returns
    -------
    result : tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int)
             - sen_locs : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
                       3D positions of MEG coil integration points (MEG space).
             - sen_dirs : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
                         Direction of the MEG coil integration points (MEG space).
             - sen_locs_mri : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
                           3D positions of MEG coil integration points (MRI space).
             - sen_dirs_mri : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
                             Direction of the MEG coil integration points (MRI space).
             - ws : np.ndarray, shape(meg_ch_integration_pt_cnt,)
                    Weights for MEG coil integration points.
             - meg_ch_indices : numpy.ndarray, shape(meg_ch_integration_pt_cnt,)
                                Indices of MEG channels.
             - meg_ch_cnt : integer
                            Number of MEG channels.
    
    Raises
    ------
    AssertionError
        Unequal compensation for MEG has not yet been implemented.
    NotImplementedError
        Unequal compensation for MEG has not yet been implemented.
    """
    # Load MEG coil meta information (e.g. direction)
    meg_ch_infos = _get_meg_coil_integration_pts(rec_info, coreg.meeg_to_mri_tr)    
    meg_ch_cnt = len(meg_ch_infos)
    meg_ch_comp = np.zeros((meg_ch_cnt))
    for (meg_ch_idx, meg_ch) in enumerate(meg_ch_infos):
        meg_ch_comp[meg_ch_idx] = int(meg_ch["coil_type"]) >> 16
    if (len(np.unique(meg_ch_comp)) != 1):
        raise AssertionError("Unequal compensation of channels")
    meg_ch_comp = np.unique(meg_ch_comp)[0]
    if (meg_ch_comp != 0):
        raise NotImplementedError("Unequal compensation not yet implemented.")
    
    # sen_dir contains the direction of the coils and sen_loc contains the position vector
    sen_locs_meg = list(); sen_locs_mri = list(); sen_dirs_meg = list(); sen_dirs_mri = list(); ws = list(); meg_ch_indices = list() 
    for (meg_ch_info_idx, _) in enumerate(meg_ch_infos):
        sen_locs_meg.extend(meg_ch_infos[meg_ch_info_idx]["sen_loc_meg"])  # 3D positions of MEG coil integration points (MEG space)
        sen_locs_mri.extend(meg_ch_infos[meg_ch_info_idx]["sen_loc_mri"])  # 3D positions of MEG coil integration points (MRI space)
        sen_dirs_meg.extend(meg_ch_infos[meg_ch_info_idx]["sen_dir_meg"])  # Direction of the MEG coil integration points (MEG space)
        sen_dirs_mri.extend(meg_ch_infos[meg_ch_info_idx]["sen_dir_mri"])  # Direction of the MEG coil integration points (MRI space)
        ws.extend(meg_ch_infos[meg_ch_info_idx]["w"])  # Weights for MEG coil integration points
        meg_ch_indices.extend(np.ones((meg_ch_infos[meg_ch_info_idx]["w"].shape[0],)) * meg_ch_info_idx)
    sen_locs_meg = np.asarray(sen_locs_meg); sen_locs_mri = np.asarray(sen_locs_mri)
    sen_dirs_meg = np.asarray(sen_dirs_meg); sen_dirs_mri = np.asarray(sen_dirs_mri)
    ws = np.asarray(ws); meg_ch_indices = np.asarray(meg_ch_indices, dtype=int)
    
    return (sen_locs_meg, sen_dirs_meg, sen_locs_mri, sen_dirs_mri, ws, meg_ch_indices, meg_ch_cnt)

def _get_meg_coil_integration_pts(rec_info, meg_to_mri_trans):
    """
    Get MEG integration points.
    
    Gets MEG coil spatial information,
    such as coils' individual integration points (sen_loc), 
    their directions (sen_dir), and weights for these integration points. 
    
    Parameters
    ----------
    rec_info : str
               Path to the *.fif file of the recording
    meg_to_mri_trans : numpy.ndarray, shape(4, 4)
                       Transformation from MEG to MRI space.
               
    Returns
    -------
    meg_chs : dict, ('coil_type', 'sen_loc', 'sen_dir', 'w', 'ch_trans', 'sen_loc_mri', 'sen_dir_mri')
              MEG channel specific information.
              
    Raises
    ------
    AssertionError
        Cannot determin MEG coil type.
    """
    rec_info = mne.io.read_info(rec_info, verbose = "ERROR")
    
    dev_to_meg_trans = rec_info['dev_head_t']["trans"]
    coil_types = mne.forward._read_coil_defs(verbose="Error")  # Reads meg coil types from mne library pylint: disable=protected-access
    meg_chs = list()
    for (_, channel) in enumerate(rec_info["chs"]):
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
            
            loc_trans = np.dot(dev_to_meg_trans, channel["pos_rot_scale"])
                
            meg_ch = {"coil_type": channel["coil_type_info"]["coil_type"],
                      "sen_loc": np.copy(channel["coil_type_info"]["rmag"]),
                      "sen_dir": np.copy(channel["coil_type_info"]["cosmag"]),
                      "w": np.copy(channel["coil_type_info"]["w"]),
                      "ch_trans": ch_trans}
            
            meg_ch["sen_loc_meg"] = np.dot(meg_ch["sen_loc"], loc_trans[:3, :3].T) + loc_trans[:3, 3]
            meg_ch["sen_dir_meg"] = np.dot(meg_ch["sen_dir"], loc_trans[:3, :3].T)
            
            meg_ch["sen_loc_mri"] = np.dot(meg_ch["sen_loc_meg"], meg_to_mri_trans[:3, :3].T) + meg_to_mri_trans[:3, 3]
            meg_ch["sen_dir_mri"] = np.dot(meg_ch["sen_dir_meg"], meg_to_mri_trans[:3, :3].T)
            
            meg_chs.append(meg_ch)
            
    return meg_chs

def get_meg_coil_pos(rec_info, meg_to_mri_trans):
    """
    Get MEG coil position.
    
    Gets MEG coil spatial information,
    such as coils' individual integration points (sen_loc), 
    their directions (sen_dir), and weights for these integration points. 
    
    Parameters
    ----------
    rec_info : mne.io.read_info
               MEG scan meta info, obtailable via mne.io.read_info
    meg_to_mri_trans : numpy.ndarray, shape(4, 4)
                       Transformation from MEG to MRI space.
               
    Returns
    -------
    meg_chs : dict, ('coil_type', 'sen_loc', 'sen_dir', 'w', 'ch_trans', 'sen_loc_mri', 'sen_dir_mri')
              MEG channel specific information.
    
    Raises
    ------
    AssertionError
        Cannot determin MEG coil type.
    """
    rec_info = mne.io.read_info(rec_info, verbose = "ERROR")
    
    dev_to_meg_trans = rec_info['dev_head_t']["trans"]
    coil_types = mne.forward._read_coil_defs(verbose="Error")  # Reads meg coil types from mne library pylint: disable=protected-access
    pos_meg = list()
    pos_mri = list()
    for (_, channel) in enumerate(rec_info["chs"]):
        if (channel["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH):
            
            ch_trans = np.zeros((4, 4)); ch_trans[3, 3] = 1
            ch_trans[:3, 3] = channel["loc"][0:3]
            ch_trans[:3, 0] = channel["loc"][3:6]
            ch_trans[:3, 1] = channel["loc"][6:9]
            ch_trans[:3, 2] = channel["loc"][9:12]
             
            channel["pos"] = ch_trans[0:3]
            channel["pos_rot_scale"] = ch_trans
             
            curr_coil_pos = None
            for coil_type in coil_types:
                if ((int(coil_type["coil_type"]) == channel["coil_type"]) and (coil_type["accuracy"] == 2)):
                    curr_coil_pos = coil_type["rmag"]
            if (curr_coil_pos is None):
                raise AssertionError("Unable to identify meg coil type")
            channel["coil_pos"] = curr_coil_pos
             
            loc_trans = np.dot(dev_to_meg_trans, channel["pos_rot_scale"])
            
            pos_meg.append(np.dot(np.mean(channel["coil_pos"], axis = 0), loc_trans[:3, :3].T) + loc_trans[:3, 3])
            pos_mri.append(np.dot(pos_meg[-1], meg_to_mri_trans[:3, :3].T) + meg_to_mri_trans[:3, 3])
    
    pos_meg = np.asarray(pos_meg)
    pos_mri = np.asarray(pos_mri)
    
    return (pos_meg, pos_mri)

def _comp_eeg_src_sen_contrib(bem_mdl, sen_locs_mri, ws, conductivity = (.3, .006, .3)):
    """
    Calculate the contribution of each current source (BEM normals) towards to each MEG sensor.
     
    Based on Mosher, 1999, formula #3 µ/(4*pi)(normals x tau)/norm³ ...
    extended with additional weights for MEG coil direction, MEG coil weight
    Integration simplified as multiplication with surface area, see "Multiple Interface Brain and Head Models for EEG: A Surface Charge Approach", Solis and Papandreou-Suppappola
    
    Parameters
    ----------
    bem_mdl : finnpy.src_rec.bem_mdl.BEM_mdl
              Container class, populed with the following items:
              
              vert : numpy.ndarray, shape(scaled_vtx_cnt, 3)
                     Remaining Vertices of a skin/skull model vertices.
              faces : numpy.ndarray, shape(scaled_face_cnt, 3)
                      Remaining Vertices of a skin/skull model faces.
              faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                             Normals of the individual remaining inner skull faces.
              faces_area : numpy.ndarray, shape(scaled_face_cnt)
                           Surface area of the remaining faces.
              bem_solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                             BEM solution (preliminary step for the calculation of the forward model).
    sen_locs_mri : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
                3D positions of MEG coil integration points (MRI space).
    ws : Var_type, shape(meg_ch_integration_pt_cnt,)
         Weights for MEG coil integration points.
    conductivity : np.ndarray, shape(3, ) or shape(1, )
                   Conductivity values for the individual surfaces.
                 
    Returns
    -------
    src_sen_contrib : numpy.ndarray, 
                      Contribution of each current source towards each sensor.
    """
    src_sen_contrib = np.zeros((len(sen_locs_mri), bem_mdl.solution.shape[0]))
    scaling = np.repeat((2.0 / (conductivity + np.concatenate(([0], conductivity))[:-1])) / (4.0 * np.pi), [len(vertices) for vertices in bem_mdl.vert])[None, :]
        
    src_vertices = sen_locs_mri
    tgt_vertices = bem_mdl.vert[0]; tgt_faces = bem_mdl.faces[0]
    
    (closest_face_ids, _, proj_pts) = finnpy.src_rec.utils.find_closest_faces(src_vertices, tgt_vertices, tgt_faces)
    bary_params = [finnpy.src_rec.utils.compute_baryzentric_params(tgt_vertices, tgt_face) for tgt_face in tgt_faces]
    bary_coords = [None] * len(proj_pts)
    for src_vert_idx in range(len(proj_pts)):
        bary_coords[src_vert_idx] = finnpy.src_rec.utils.compute_baryzentric_coordinates(proj_pts[src_vert_idx], tgt_vertices,
                                                                                         tgt_faces[closest_face_ids[src_vert_idx]],
                                                                                         bary_params[closest_face_ids[src_vert_idx]])
    face_weights = np.asarray(bary_coords)
    face_weights *= ws[:, None]
    weights = np.matmul(face_weights[:, None], bem_mdl.solution[bem_mdl.faces[0][closest_face_ids]])[:, 0]  # is line 256
    
    for sen_idx in np.arange(0, len(sen_locs_mri)):
        src_sen_contrib[sen_idx] = np.sum(weights[sen_idx:(sen_idx + 1)], axis = 0)
    
    src_sen_contrib *= scaling
    
    return src_sen_contrib

def _comp_meg_src_sen_contrib(bem_mdl, sen_locs_mri, sen_dirs_mri, ws, meg_ch_indices, meg_ch_cnt, conductivity = (.3,)):
    """
    Calculate the contribution of each current source (BEM normals) towards to each MEG sensor.
    
    Based on Mosher, 1999, formula #3 µ/(4*pi)(normals x tau)/norm³ ...
    extended with additional weights for MEG coil direction, MEG coil weight
    Integration simplified as multiplication with surface area, see "Multiple Interface Brain and Head Models for EEG: A Surface Charge Approach", Solis and Papandreou-Suppappola
    
    Parameters
    ----------
    bem_mdl : finnpy.src_rec.bem_mdl.BEM_mdl
              Container class, populed with the following items:
              
              vert : numpy.ndarray, shape(scaled_vtx_cnt, 3)
                     Remaining Vertices of a skin/skull model vertices.
              faces : numpy.ndarray, shape(scaled_face_cnt, 3)
                      Remaining Vertices of a skin/skull model faces.
              faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                             Normals of the individual remaining inner skull faces.
              faces_area : numpy.ndarray, shape(scaled_face_cnt)
                           Surface area of the remaining faces.
              bem_solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                             BEM solution (preliminary step for the calculation of the forward model).
    sen_locs_mri : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
                3D positions of MEG coil integration points (MRI space).
    sen_dirs_mri : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
                  Direction of the MEG coil integration points (MRI space).
    ws : Var_type, shape(meg_ch_integration_pt_cnt,)
         Weights for MEG coil integration points.
    meg_ch_indices : Var_type, shape(meg_ch_integration_pt_cnt,)
                     Indices of MEG channels.
    meg_ch_cnt : integer
                 Number of MEG channels.
    conductivity : np.ndarray, shape(3, ) or shape(1, )
                   Conductivity values for the individual surfaces.
                 
    Returns
    -------
    src_sen_contrib : numpy.ndarray, 
                      Contribution of each current source towards each sensor.
    """
    # Configure constants
    sigma = conductivity[0]
    source_mult = 2.0 / ((sigma) + 0)  # Conductivity in the first layer (sigma) and outside the skull (0)
    field_mult = sigma - 0
    
    meg_lin_pot_basis = np.zeros((meg_ch_cnt, len(bem_mdl.vert[0])))
    w_sen_dirs = np.expand_dims(ws, axis = 1) * sen_dirs_mri  # Weighted directions
    tau = np.expand_dims(sen_locs_mri, axis = 1) - bem_mdl.vert[0]
    den = np.sum(tau * tau, axis = 2)  # This equates to ||x||^3, from sqrt(x¹+x²+...+x^n)³ = z³ to (x¹+x²+...+x^n) * sqrt((x¹+x²+...+x^n)) = z * z^(0.5) = z^(1 + 0.5)  
    den *= np.sqrt(den) * 3
    for (face_idx, face) in enumerate(bem_mdl.faces[0]):
        num = np.cross(tau[:, face], bem_mdl.faces_normal[0][face_idx, :]) * np.expand_dims(w_sen_dirs, axis=1)
        for vert_idx in range(3):
            loc_effects = np.sum(num[:, vert_idx], axis=1) * bem_mdl.faces_area[0][face_idx] / den[:, face[vert_idx]]
            effects_on_vert = np.zeros((meg_ch_cnt,))
            for (effect_idx, effect_value) in enumerate(loc_effects):
                effects_on_vert[meg_ch_indices[effect_idx]] += effect_value
            meg_lin_pot_basis[:, face[vert_idx]] += effects_on_vert
    meg_lin_pot_basis *= field_mult
    src_sen_contrib = np.dot(meg_lin_pot_basis, bem_mdl.solution) * source_mult / (4 * np.pi)
    
    return src_sen_contrib

def _calc_magnetic_fields(white_vertices_mri, all_bem_verts, meg_to_mri_r, pre_fwd_solution):
    """
    Compute infinite medium potentials.
    
    Parameters
    ----------
    white_vertices_mri : numpy.ndarray, shape(valid_white_vtx_cnt, 3)
                         White matter vertices.
    all_bem_verts : numpy.ndarray, shape(bem_vtx_cnt, 3)
                    Inner skull model vertices.
    meg_to_mri_r : numpy.ndarray, shape(4, 4)
                   Transformatio matrix from the coregistration, rotation only.
    pre_fwd_solution : numpy.ndarray, shape(meg_ch_cnt, reduced_in_skull_vtx_cnt)
                       Fwd solution precursor.
               
    Returns
    -------
    fwd_sol : numpy.ndarray, shape(valid_white_vtx_cnt * 3, meg_ch_cnt)
              Forward solution with infinite medium potentials.
    """
    fwd_sol = np.zeros((len(white_vertices_mri), 3, len(all_bem_verts)))
    
    # See Mosher et al, 1999, equation #8 for reference
    for (white_vortex_mri_idx, white_vortex_mri) in enumerate(white_vertices_mri):
        magnitude = np.power(np.linalg.norm(all_bem_verts - white_vortex_mri, axis=1), 3)
        magnitude[magnitude == 0] = 1
        
        # Rotate diff from MRI space to MEG space, maintain MRI scaling
        diff = np.dot((all_bem_verts - white_vortex_mri), meg_to_mri_r[:3, :3])
        # Math: np.dot(X, rot) == (np.dot(rot.T, X)).T (if X is a vector and rot a rotation matrix)
        diff /= np.expand_dims(magnitude, axis=1)
        
        fwd_sol[white_vortex_mri_idx] = diff.T
    
    # Multiply with current dipole moments
    fwd_sol = fwd_sol.reshape(white_vertices_mri.shape[0] * 3, fwd_sol.shape[2])
    fwd_sol = np.dot(fwd_sol, pre_fwd_solution.T)
    
    return fwd_sol

def _add_current_distribution(fwd_sol, white_vertices, meg_ch_cnt, sen_locs, sen_dirs, ws, meg_ch_indices):
    """
    Add current distribution to the MEG forward solution.
    
    Parameters
    ----------
    fwd_sol : numpy.ndarray, shape(number_of_valid_vertices * 3, meg_ch_cnt)
              Previously calcualted (preliminary) forward solution.
    white_vertices : numpy.ndarray, shape(number_of_valid_vertices, 3)
                     Valid white matter vertices.
    meg_ch_cnt : int
                 Number of MEG channels.
    sen_locs : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
            3D positions of MEG coil integration points (MEG space).
    sen_dirs : numpy.ndarray, shape(meg_ch_integration_pt_cnt, 3)
              Direction of the MEG coil integration points (MEG space).
    ws : Var_type, shape(meg_ch_integration_pt_cnt,)
         Weights for MEG coil integration points.
    meg_ch_indices : Var_type, shape(meg_ch_integration_pt_cnt,)
                     Indices of MEG channels.
               
    Returns
    -------
    fwd_sol : numpy.ndarray, shape(number_of_valid_vertices * 3, meg_ch_cnt)
              Forward solution with current distribution.
    """
    pc = np.empty((len(white_vertices) * 3, meg_ch_cnt))
    for (vortex_idx, vortex) in enumerate(white_vertices):
        # Calculates the magnetic field at a vortex from all MEG sensors.
        pp = _calc_bem_fields(vortex, sen_locs, sen_dirs)
        pp *= ws  # Adds MEG coil weight
        pp = pp.squeeze(0)
        
        # Attribute primary current spread to respective MEG channels.
        tmp = np.zeros((3, meg_ch_cnt))
        for idx in range(len(meg_ch_indices)):
            tmp[:, meg_ch_indices[idx]] += pp[:, idx]
        tmp = np.asarray(tmp)
        pc[int(3 * vortex_idx):int(3 * (vortex_idx + 1)), :] = tmp
    
    # Add primary current spread to MEG channels.
    fwd_sol += pc
    
    return fwd_sol

def _calc_bem_fields(vortex, sen_locs, sen_dirs):
    """
    Calculate the magnetic field at a vortex from all MEG sensors.
    
    Parameters
    ----------
    vortex : numpy.ndarray, shape(3,)
             Position of the vortex
    sen_locs : numpy.ndarray, shape(meg_chs_integration_pt_cnt, 3)
            3D positions of MEG coil integration points (MEG space)
    sen_dirs : numpy.ndarray, shape(meg_chs_integration_pt_cnt, 3)
              Direction of the MEG coil integration points (MEG space)
               
    Returns
    -------
    fields : numpy.ndarray, shape(1, 3, meg_chs_integration_pt_cnt)
             The magnetic field of all MEG sensors at a vortex.    
    """
    # See Mosher et al, 1999, equation #1.
    diff = np.expand_dims(sen_locs.T, axis = 0) - np.expand_dims(vortex, axis = [0, 2])
    norm_diff = np.power(np.linalg.norm(diff, axis = 1), 3)
    norm_diff = norm_diff.squeeze(0)
    norm_diff[norm_diff == 0] = 1
    norm_diff = np.expand_dims(norm_diff, axis = [0, 1])
    
    # See Mosher et al, 1999, equation #19.
    fields = np.empty((1, 3, sen_locs.shape[0]))
    fields[:, 0] = diff[:, 1] * sen_dirs[:, 2] - diff[:, 2] * sen_dirs[:, 1]
    fields[:, 1] = diff[:, 2] * sen_dirs[:, 0] - diff[:, 0] * sen_dirs[:, 2]
    fields[:, 2] = diff[:, 0] * sen_dirs[:, 1] - diff[:, 1] * sen_dirs[:, 0]
    
    # See Mosher et al, 1999, equation #1.
    fields /= norm_diff
    
    return fields

def restrict(cort_mdl, fwd_sol, coreg):
    """
    Transform a fwd model into surface orientation and shrinks it.
    
    From orthogonal to the respective surface cluster, allowing for cortically constrained inverse modeling.
    Shrinking is achieved by making closeby channels project to the same destination.
    This is different from a 'default' 3D transformation. 
    
    Parameters
    ----------
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
    fwd_sol : numpy.ndarray, shape(meg_ch_cnt, valid_vtx_cnt * 3)
              Forward solution with default orientation.
    
    coreg : finnpy.src_rec.coreg.Coreg
            Container class, populed with the following items:
             
            rotors : numpy.ndarray, shape(9,)
            Sequence of rotors defining rotation (3), translation (3) and scaling (3).
            
            mri_to_meg_trs : numpy.ndarray, shape(4, 4)
            Full affine transformation matrix (MRI -> MEG)
            mri_to_meg_tr : numpy.ndarray, shape(4, 4)
            Rigid affine transformation matrix (MRI -> MEG)
            mri_to_meg_rs : numpy.ndarray, shape(4, 4)
            Rotation & scaling only affine transformation matrix (MRI -> MEG)
            mri_to_meg_r : numpy.ndarray, shape(4, 4)
            Rotation only affine transformation matrix (MRI -> MEG)
            
            meg_to_mri_trs : numpy.ndarray, shape(4, 4)
            Full affine transformation matrix (MEG -> MRI)
            meg_to_mri_tr : numpy.ndarray, shape(4, 4)
            Rigid affine transformation matrix (MEG -> MRI)
            meg_to_mri_rs : numpy.ndarray, shape(4, 4)
            Rotation & scaling only affine transformation matrix (MEG -> MRI)
            meg_to_mri_r : numpy.ndarray, shape(4, 4)
            Rotation only affine transformation matrix (MEG -> MRI)
    
    Returns
    -------
    fwd_sol * rot : numpy.ndarray, shape(meg_ch_cnt, valid_vtx_cnt)
                    Transformed surface model.
    """   
    # Computes vertex normals
    lh_acc_normals = _calc_acc_hem_normals(cort_mdl.lh_vert, cort_mdl.lh_faces)
    rh_acc_normals = _calc_acc_hem_normals(cort_mdl.rh_vert, cort_mdl.rh_faces)
     
    # Figures out which real vertices are represented by the same model vortex
    (lh_cluster_grp, lh_cluster_indices) = _find_vertex_clusters(cort_mdl.lh_vert, cort_mdl.lh_valid_vert, cort_mdl.lh_faces)
    (rh_cluster_grp, rh_cluster_indices) = _find_vertex_clusters(cort_mdl.rh_vert, cort_mdl.rh_valid_vert, cort_mdl.rh_faces)
    
    # Determines the orientation 
    lh_orient = finnpy.src_rec.utils.get_eigenbasis(lh_acc_normals, cort_mdl.lh_valid_vert, lh_cluster_grp, lh_cluster_indices, coreg.mri_to_meeg_tr)
    rh_orient = finnpy.src_rec.utils.get_eigenbasis(rh_acc_normals, cort_mdl.rh_valid_vert, rh_cluster_grp, rh_cluster_indices, coreg.mri_to_meeg_tr)
    
    # Concatenates the eigenvector bases 
    orient = np.concatenate((lh_orient, rh_orient), axis = 0)
     
    # Transforms eigenvector matrix into block matrix for ease of use
    rot = finnpy.src_rec.utils.orient_mat_to_block_format(orient[2::3, :])
    
    # Applies rotation matrix to the fwd solution 
    return fwd_sol * rot

def _calc_acc_hem_normals(white_vert, white_faces):
    """
    Generate vertex normals from accumulated face normals.
    
    See https://en.wikipedia.org/wiki/Vertex_normal.
    
    Parameters
    ----------
    white_vert : numpy.ndarray, shape(vert_cnt, 3)
                 MRI model vertices.
    
    white_faces : numpy.ndarray, shape(face_cnt, 3)
                  MRI model faces.
               
    Returns
    -------
    acc_normals : numpy.ndarray, shape(vert_cnt, 3)
                  Array of surface normals for the input faces/vertices.
    """
    # Calculate surface normals
    vert0 = white_vert[white_faces[:, 0], :]
    vert1 = white_vert[white_faces[:, 1], :]
    vert2 = white_vert[white_faces[:, 2], :]
    vortex_normals = finnpy.src_rec.utils.fast_3D_cross_product_multi(vert1 - vert0, vert2 - vert1)
    len_vortex_normals = np.linalg.norm(vortex_normals, axis = 1)
    vortex_normals[len_vortex_normals > 0] /= np.expand_dims(len_vortex_normals[len_vortex_normals > 0], axis = 1)
    
    # Accumulate face normals
    acc_normals = np.zeros((white_vert.shape[0], 3))
    for outer_idx in range(3):
        for inner_idx in range(3):
            value = np.zeros((white_vert.shape[0],))
            for vertex_idx in range(white_faces.shape[0]):
                value[white_faces[vertex_idx, outer_idx]] += vortex_normals[vertex_idx, inner_idx]
            acc_normals[:, inner_idx] += value
    
    # Normalize face normals
    len_acc_normals = np.linalg.norm(acc_normals, axis = 1)
    acc_normals[len_acc_normals > 0] /= np.expand_dims(len_acc_normals[len_acc_normals > 0], axis = 1)
    
    return acc_normals

def _find_vertex_clusters(white_vert, valid_vert, white_faces):
    """
    Identify which input vertices (white_vert) are presented by which valid vortex.
    
    Parameters
    ----------
    white_vert : numpy.ndarray, shape(lh_vert_cnt, 3)
                 MRI model vertices.
    valid_vert : numpy.ndarray, shape(lh_vert_cnt,)
                       Valid/supporting vertices.
    white_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                  MRI model faces.
    
    Returns
    -------
    result : tuple of (list, list)
             - cluster_grp : list, len(n,)
                             Transformed surface model.
             - cluster_indices : list, len(n,)
                                 Transformed surface model.
    """
    # Get adjacency matrix to identify nearest valid vortex
    edges = scipy.sparse.coo_matrix((np.ones(3 * white_faces.shape[0]), 
                                     (np.concatenate((white_faces[:, 0], white_faces[:, 1], white_faces[:, 2])),
                                      np.concatenate((white_faces[:, 1], white_faces[:, 2], white_faces[:, 0])))),
                                     shape = (white_vert.shape[0], white_vert.shape[0]))  # noqa: E127
    edges = edges.tocsr()
    edges += edges.T
    edges = edges.tocoo()
    edges_dists = np.linalg.norm(white_vert[edges.row, :] - white_vert[edges.col, :], axis = 1)
    edges_adjacency = scipy.sparse.csr_matrix((edges_dists, (edges.row, edges.col)), shape = edges.shape)
    _, _, min_idx = scipy.sparse.csgraph.dijkstra(edges_adjacency, indices = np.where(valid_vert)[0], min_only = True, return_predecessors = True)
    
    # Accumulates the clusters
    sort_near_idx = np.argsort(min_idx)
    sort_min_idx = min_idx[sort_near_idx]
    breaks = np.where(sort_min_idx[1:] != sort_min_idx[:-1])[0] + 1
    starts = [0] + breaks.tolist()
    ends = breaks.tolist() + [len(min_idx)]
    cluster_grp = list()
    for cluster_idx in range(len(starts)):
        cluster_grp.append(np.sort(sort_near_idx[starts[cluster_idx]:ends[cluster_idx]]))
    pre_cluster_indices = sort_min_idx[breaks - 1]
    cluster_indices = np.searchsorted(pre_cluster_indices, np.where(valid_vert)[0])
    
    return (cluster_grp, cluster_indices)
