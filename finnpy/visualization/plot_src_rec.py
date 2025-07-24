"""
Created on May 2, 2025.

@author: voodoocode
"""

import numpy as np
import sklearn
import pyvista
import finnpy.src_rec.fwd_mdl  # @UnresolvedImport
import copy

def plot_reg_avg(subj_to_fsavg_mdl, morphed_channels, color_data,
                 signal_type = None, rec_meta_info = None, coreg = None, ch_names = None):
    """
    Generate a high resolution 3D plot using using Desikan-Killiany parcellation on fs-average projected data.
    
    Parameters
    ----------
    subj_to_fsavg_mdl : finnpy.src_rec.subj_to_fsavg.Subj_to_fsavg_mdl
                        Container class, populated with the following items:
    
                        trans : numpy.ndarray, shape(valid_subj_vtx_cnt, valid_subj_vtx_cnt)
                                Transformation matrix
                        lh_valid_vert : numpy.ndarray, shape(lh_vtx_cnt,)
                                        Valid/supporting vertices for left hemisphere.
                        lh_vert : numpy.ndarray, shape(lh_vtx_cnt, 3)
                                  White matter surface model vertices (left hemisphere).
                        lh_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                                   White matter surface model faces (left hemisphere).
                        rh_vert : numpy.ndarray, shape(rh_vtx_cnt, 3)
                                  White matter surface model vertices (right hemisphere).
                        rh_faces : numpy.ndarray, shape(rh_face_cnt, 3)
                                   White matter surface model faces (right hemisphere).
                        rh_valid_vert : numpy.ndarray, shape(rh_vtx_cnt,)
                                        Valid flags for white matter surface model vertices (right hemisphere).
                        rh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                        Valid/supporting vertices for right hemisphere.
    morphed_channels : list(np.ndarray(variable size)), len = valid_vtx_cnt
                       Projects from 68 cortical regions onto the vertices of the high resolution MRI scans.
    color_data : numpy.ndarray(valid_vtx_cnt, )
                 Scalar values to be plotted at each vortex of the model.
    signal_type : str
                  'EEG' or 'MEG'.
    rec_meta_info : tuple of (np.ndarray, list or np.ndarray)
                    - eeg_coords or pos_meg : np.ndarray, shape(ch_cnt, 3)
                                              Position of the EEG/MEG sensors. 
                    - ws or pos_mri: list or np.ndarray
                                     Either EEG weights or position of the MEG sensor in MRI space.
    coreg : finnpy.src_rec.coreg
            Coregistration between MEEG and MRI.
    ch_names : list
               Channel names.
    """
    ch_ids = [int(ch_id) for ch_ids in morphed_channels for ch_id in ch_ids]
    mod_color_data = np.zeros((np.max(ch_ids) + 1, ))
    for (ch_idx, ch_ids) in enumerate(morphed_channels):
        mod_color_data[np.asarray(ch_ids, dtype = int)] += color_data[ch_idx]
    plot_fsavg_space(subj_to_fsavg_mdl, mod_color_data, signal_type, rec_meta_info, coreg, ch_names)

def plot_fsavg_space(subj_to_fsavg_mdl, color_data,
                     signal_type = None, rec_meta_info = None, coreg = None, ch_names = None):
    """
    Use finnpy's fsavg model to produce a high resolution 3D plot.
    
    Parameters
    ----------
    subj_to_fsavg_mdl : finnpy.src_rec.subj_to_fsavg.Subj_to_fsavg_mdl
                        Container class, populated with the following items:
    
                        trans : numpy.ndarray, shape(valid_subj_vtx_cnt, valid_subj_vtx_cnt)
                                Transformation matrix
                        lh_valid_vert : numpy.ndarray, shape(lh_vtx_cnt,)
                                        Valid/supporting vertices for left hemisphere.
                        lh_vert : numpy.ndarray, shape(lh_vtx_cnt, 3)
                                  White matter surface model vertices (left hemisphere).
                        lh_faces : numpy.ndarray, shape(lh_face_cnt, 3)
                                   White matter surface model faces (left hemisphere).
                        rh_vert : numpy.ndarray, shape(rh_vtx_cnt, 3)
                                  White matter surface model vertices (right hemisphere).
                        rh_faces : numpy.ndarray, shape(rh_face_cnt, 3)
                                   White matter surface model faces (right hemisphere).
                        rh_valid_vert : numpy.ndarray, shape(rh_vtx_cnt,)
                                        Valid flags for white matter surface model vertices (right hemisphere).
                        rh_valid_vert : numpy.ndarray, shape(fs_avg_vtx_cnt,)
                                        Valid/supporting vertices for right hemisphere.
    color_data : numpy.ndarray(valid_vtx_cnt, )
                 Scalar values to be plotted at each vortex of the model.
    signal_type : str
                  'EEG' or 'MEG'.
    rec_meta_info : tuple of (np.ndarray, list or np.ndarray)
                    - eeg_coords or pos_meg : np.ndarray, shape(ch_cnt, 3)
                                              Position of the EEG/MEG sensors. 
                    - ws or pos_mri: list or np.ndarray
                                     Either EEG weights or position of the MEG sensor in MRI space.
    coreg : finnpy.src_rec.coreg
            Coregistration between MEEG and MRI.
    ch_names : list
               Channel names.
    """
    plot_subj_space(subj_to_fsavg_mdl, color_data, signal_type, rec_meta_info, coreg, ch_names)

def plot_subj_space(cort_mdl, color_data,
                    signal_type = None, rec_meta_info = None, coreg = None, ch_names = None,
                    title = None):
    """
    Use finnpy's cortical model to produce a high resolution 3D plot.
    
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
           rh_valid_vert : numpy.ndarray, shape(rh_vtx_cnt,)
                           Valid flags for white matter surface model vertices (right hemisphere).
           octa_model_vert : numpy.ndarray, shape(octa_mdl_vtx_cnt, 3)
                             Octamodel vertices (left hemisphere).
           octa_model_faces : numpy.ndarray, shape(octa_mdl_face_cnt, 3)
                              Octamodel faces (right hemisphere).
    color_data : numpy.ndarray(valid_vtx_cnt, )
                 Scalar values to be plotted at each vortex of the model.
    signal_type : str
                  'EEG' or 'MEG'.
    rec_meta_info : tuple of (np.ndarray, list or np.ndarray)
                    - eeg_coords or pos_meg : np.ndarray, shape(ch_cnt, 3)
                                              Position of the EEG/MEG sensors. 
                    - ws or pos_mri: list or np.ndarray
                                     Either EEG weights or position of the MEG sensor in MRI space.
    coreg : finnpy.src_rec.coreg
            Coregistration between MEEG and MRI.
    ch_names : list
               Channel names.
    title : str
            Title string.
    """
    
    cort_mdl = copy.deepcopy(cort_mdl)
    
    lh_colors = np.empty((cort_mdl.lh_vert.shape[0]))
    lh_vert = cort_mdl.lh_vert
    lh_valid_vert = cort_mdl.lh_vert[np.where(cort_mdl.lh_valid_vert)[0]]
    lh_tree = sklearn.neighbors.KDTree(lh_valid_vert)
    for lh_vert_idx in range(0, lh_vert.shape[0]):
        lh_valid_vert_idx = int(lh_tree.query(lh_vert[[lh_vert_idx]])[1][0, 0])
        lh_colors[lh_vert_idx] = color_data[lh_valid_vert_idx]
      
    lh_offset = lh_valid_vert.shape[0]
    rh_colors = np.empty((cort_mdl.rh_vert.shape[0]))
    rh_vert = cort_mdl.rh_vert
    rh_valid_vert = cort_mdl.rh_vert[np.where(cort_mdl.rh_valid_vert)[0]]
    rh_tree = sklearn.neighbors.KDTree(rh_valid_vert)
    for rh_vert_idx in range(0, rh_vert.shape[0]):
        rh_valid_vert_idx = int(rh_tree.query(rh_vert[[rh_vert_idx]])[1][0, 0])
        rh_colors[rh_vert_idx] = color_data[rh_valid_vert_idx + lh_offset]
         
    pl = pyvista.Plotter(window_size = (800, 600))
    offset = np.abs(np.min(cort_mdl.lh_vert[:, 0]) - np.max(cort_mdl.lh_vert[:, 0])) / 2
    cort_mdl.lh_vert[:, 0] -= offset / 2; lh_facesplit = [(3, face[0], face[1], face[2]) for face in cort_mdl.lh_faces]
    lh_pt_data = pyvista.PolyData(cort_mdl.lh_vert, np.asarray(lh_facesplit, dtype = int).reshape(-1))
    lh_pt_data.point_data["values"] = lh_colors
    pl.add_mesh(lh_pt_data, scalars = "values", color = (.4, .4, .4), opacity = 1)
    cort_mdl.rh_vert[:, 0] += offset / 2; rh_facesplit = [(3, face[0], face[1], face[2]) for face in cort_mdl.rh_faces]
    rh_pt_data = pyvista.PolyData(cort_mdl.rh_vert, np.asarray(rh_facesplit, dtype = int).reshape(-1))
    rh_pt_data.point_data["values"] = rh_colors
    pl.add_mesh(rh_pt_data, scalars = "values", color = (.4, .4, .4), opacity = 1)
    
    if (signal_type == "MEG"):
        (_, pos_mri) = finnpy.src_rec.fwd_mdl.get_meg_coil_pos(rec_meta_info, coreg.meeg_to_mri_tr)
    if (signal_type == "EEG"):
        (pos_mri, _) = finnpy.src_rec.fwd_mdl.get_eeg_sen_info(rec_meta_info, coreg)
        
    pl.add_points(pos_mri, render_points_as_spheres = False, point_size = 16)  # pylint: disable=possibly-used-before-assignment
    pl.add_point_labels(pos_mri, ch_names, shape_opacity = 0, font_size = 12)#, text_color = "white")
    
    #pl.set_background("black")
    
    if (title is not None):
        pl.add_title(title)
        
    #===========================================================================
    # def compute_azimuth_elevation(camera):
    #     # Vector from focal point to camera position
    #     vec = np.array(camera.position) - np.array(camera.focal_point)
    #     x, y, z = vec
    # 
    #     # Azimuth: angle in XY-plane from X-axis (atan2(y, x))
    #     azimuth = np.degrees(np.arctan2(y, x))
    # 
    #     # Elevation: angle from XY-plane upwards (atan2(z, hyp))
    #     hyp = np.sqrt(x**2 + y**2)
    #     elevation = np.degrees(np.arctan2(z, hyp))
    # 
    #     return azimuth, elevation
    # 
    # def print_camera_angles(a, b):
    #     cam = pl.camera
    #     az, el = compute_azimuth_elevation(cam)
    #     print(f"Azimuth: {az:.2f}°, Elevation: {el:.2f}°")
    #     print(f"Position: {cam.position}")
    #     print("---")
    # interactor = pl.iren
    # interactor.add_observer('TimerEvent', print_camera_angles)
    # timer_id = interactor.create_timer(500, True)
    #===========================================================================
    
    #===========================================================================
    # cam = pl.camera
    # cam.Azimuth(145)
    # cam.Elevation(6)
    # cam.SetPosition((-0.6108798273874511, 0.41434351774217093, 0.09811421902995418))
    #===========================================================================
    
    pl.show()
