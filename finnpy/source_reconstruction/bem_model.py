'''
Created on Oct 12, 2022

@author: voodoocode
'''

import os
import subprocess
import shutil
import warnings

import numpy as np

import nibabel.freesurfer

import matplotlib.pyplot as plt

import finnpy.source_reconstruction.utils
import finnpy.source_reconstruction.sphere_model

def calc_skull_and_skin_models(path, patient_id, preflood_height = 25, overwrite = False):
    if (overwrite == True or os.path.exists(path + "bem/watershed/" + patient_id + "_inner_skull_surface") == False):
        
        cmd = ["mri_watershed", "-h", str(preflood_height), "-useSRAS", "-surf", path + "bem/watershed/" + patient_id, path + "mri/T1.mgz", path + "bem/watershed/ws.mgz"]
        finnpy.source_reconstruction.utils.run_subprocess_in_custom_working_directory(patient_id, cmd)
        
        os.remove(path + "bem/watershed/" + patient_id + "_brain_surface")
        os.remove(path + "bem/watershed/ws.mgz")

def read_skull_and_skin_models(path, subj_name):
    (ws_in_skull_vert, ws_in_skull_faces) = nibabel.freesurfer.read_geometry(path + "bem/watershed/" + subj_name + "_inner_skull_surface")
    (ws_out_skull_vert, ws_out_skull_faces) = nibabel.freesurfer.read_geometry(path + "bem/watershed/" + subj_name + "_outer_skull_surface")
    (ws_out_skin_vect, ws_out_skin_faces) = nibabel.freesurfer.read_geometry(path + "bem/watershed/" + subj_name + "_outer_skin_surface")
    
    return (ws_in_skull_vert, ws_in_skull_faces,
            ws_out_skull_vert, ws_out_skull_faces,
            ws_out_skin_vect, ws_out_skin_faces)
    
def plot_skull_and_skin_models(ws_in_skull_vert, ws_in_skull_faces, 
                               ws_out_skull_vert, ws_out_skull_faces,
                               ws_out_skin_vect, ws_out_skin_faces, 
                               path):
    
    (t1_data_trans, ras_to_mri) = load_and_orient_t1(path)
    mri_to_ras = np.linalg.inv(ras_to_mri)
    
    ws_in_skull_vert_trans = np.dot(ws_in_skull_vert, mri_to_ras[:3, :3].transpose()); ws_in_skull_vert_trans += mri_to_ras[:3, 3]
    if (ws_out_skull_vert is not None):
        ws_out_skull_vert_trans = np.dot(ws_out_skull_vert, mri_to_ras[:3, :3].transpose()); ws_out_skull_vert_trans += mri_to_ras[:3, 3]
    if (ws_out_skin_vect is not None):
        ws_out_skin_vect_trans = np.dot(ws_out_skin_vect, mri_to_ras[:3, :3].transpose()); ws_out_skin_vect_trans += mri_to_ras[:3, 3]
    
    if (ws_out_skull_vert is not None and ws_out_skull_vert is not None):
        surfaces = [["inner_skull", "#FF0000", ws_in_skull_vert_trans, ws_in_skull_faces],
                    ["outer_skull", "#FFFF00", ws_out_skull_vert_trans, ws_out_skull_faces],
                    ["outer_skin", "#FFAA80", ws_out_skin_vect_trans, ws_out_skin_faces]]
    else:
        surfaces = [["inner_skull", "#FF0000", ws_in_skull_vert_trans, ws_in_skull_faces],]
    
    (_, axes) = plt.subplots(3, 4, gridspec_kw = {'wspace':0.025, 'hspace':0.025})
    plot_skull_and_skin_subplots(t1_data_trans, axes, surfaces)
    plt.show(block = True)

def load_and_orient_t1(path):
    t1_img = nibabel.load(path + "mri/T1.mgz")
    
    src_orientation = nibabel.orientations.aff2axcodes(t1_img.affine)
    tgt_orientation = ('R', 'A', 'S')
    trans_orientation = nibabel.orientations.ornt_transform(nibabel.orientations.axcodes2ornt(src_orientation),
                                                            nibabel.orientations.axcodes2ornt(tgt_orientation))
    
    aff_trans = nibabel.orientations.inv_ornt_aff(trans_orientation, t1_img.shape)
    t1_image_trans = t1_img.as_reoriented(trans_orientation)
    ras_to_mri = np.dot(t1_img.header.get_vox2ras_tkr(), aff_trans)
    
    return (t1_image_trans.get_fdata(), ras_to_mri)

def plot_skull_and_skin_subplots(data, axes, surfaces):
    (primary_dim, secondary_dim_x, secondary_dim_y) = (1, 0, 2) # Plot in reference to coronal orientation.
    
    slices = np.asarray([[.12, .14, .17, .21],
                         [.26, .32, .39, .47],
                         [.53, .61, .68, .74],
                         [.79, .83, .86, .88]], dtype = np.float32) * data.shape[primary_dim]
    slices = np.asarray(slices, dtype = np.int32)
    
    for row_idx in range(3):
        for col_idx in range(4):
            axes[row_idx, col_idx].imshow(data[:, slices[row_idx, col_idx], :].T, cmap = plt.cm.gray, origin = "lower")
            axes[row_idx, col_idx].set_autoscale_on(False)
            
            axes[row_idx, col_idx].axis('off')
            axes[row_idx, col_idx].set_aspect('equal')
            
            for surface in surfaces:
                warnings.simplefilter('ignore')
                axes[row_idx, col_idx].tricontour(surface[2][:, secondary_dim_x], surface[2][:, secondary_dim_y], 
                                                  surface[3], surface[2][:, primary_dim], 
                                                  levels = [slices[row_idx, col_idx]], colors = surface[1], linewidths = 1.0, 
                                                  zorder = 1)
    warnings.simplefilter('default')
    
    return (slices, primary_dim, secondary_dim_x, secondary_dim_y)

import mne

def calc_bem_model(ws_in_skull_vert, ws_in_skull_faces, tgt_icosahedron_level = 4):
    src_icosahedron_level = int(np.log(ws_in_skull_faces.shape[0]/20)/np.log(2) / 2)
    
    (src_vert, _) = finnpy.source_reconstruction.sphere_model.read_sphere_from_icosahedron_in_fs_order(src_icosahedron_level)
    (tgt_vert, tgt_faces) = finnpy.source_reconstruction.sphere_model.calculate_sphere_from_icosahedron(tgt_icosahedron_level)
    
    trans_ws_in_skill_vert = np.copy(ws_in_skull_vert)[finnpy.source_reconstruction.utils.find_nearest_neighbor(src_vert, tgt_vert)[0]]
    ws_in_skull_faces = tgt_faces
    
    trans_ws_in_skill_vert /= 1000 # whyever??
    
    ###make bem model
    x_pos = trans_ws_in_skill_vert[ws_in_skull_faces[:, 0], :]
    y_pos = trans_ws_in_skill_vert[ws_in_skull_faces[:, 1], :]
    z_pos = trans_ws_in_skill_vert[ws_in_skull_faces[:, 2], :]
    
    faces_normal = finnpy.source_reconstruction.utils.fast_cross_product((y_pos - x_pos), (z_pos - x_pos))
    double_faces_area = finnpy.source_reconstruction.utils.magn_of_vec(faces_normal)
    n_faces_normal = np.linalg.norm(faces_normal, axis = 1)
    faces_normal[n_faces_normal > 0] = faces_normal[n_faces_normal > 0] / np.expand_dims(n_faces_normal[n_faces_normal > 0], axis = 1)
    
    omega = find_non_diag_omega(ws_in_skull_faces, trans_ws_in_skill_vert, double_faces_area, faces_normal)
    omega = find_diag_omega(omega, ws_in_skull_faces)
    
    bem_solution = np.linalg.inv(np.eye(omega.shape[0]) + 1/omega.shape[0] - omega / (2*np.pi))
    
    return (trans_ws_in_skill_vert, ws_in_skull_faces, double_faces_area/2, faces_normal, bem_solution)

def find_non_diag_omega(faces, vert, double_faces_area, faces_normal):
    omega = np.zeros((vert.shape[0], vert.shape[0]))
    
    for (face_idx, face) in enumerate(faces):
        
        tgt_vert = vert[face, :]
        
        r0 = tgt_vert[0, :] - vert
        r1 = tgt_vert[1, :] - vert
        r2 = tgt_vert[2, :] - vert
        
        nr0 = np.linalg.norm(r0, axis = 1)
        nr1 = np.linalg.norm(r1, axis = 1)
        nr2 = np.linalg.norm(r2, axis = 1)
        
        sa0 = nr0 * nr1 * nr2
        sa1 = np.sum(r0 * r1, axis = 1) * nr2
        sa2 = np.sum(r0 * r2, axis = 1) * nr1
        sa3 = np.sum(r1 * r2, axis = 1) * nr0
        
        solid_angles = 2*np.arctan2(np.sum(np.cross(r0, r1) * r2, axis = 1), sa0 + sa1 + sa2 + sa3)
        
        r10 = r1[0] - r0[0]; nr10 = np.linalg.norm(r10)
        r21 = r2[0] - r1[0]; nr21 = np.linalg.norm(r21)
        r02 = r0[0] - r2[0]; nr02 = np.linalg.norm(r02)

        # Correct for the auto-solid angle problem for use in the gamma functions
        nr0[face] = 1; nr1[face] = 1; nr2[face] = 1
                
        gamma0 = np.log((nr0 * nr10 + np.dot(r0, r10))/(nr1 * nr10 + np.dot(r1, r10)))/nr10
        gamma1 = np.log((nr1 * nr21 + np.dot(r1, r21))/(nr2 * nr21 + np.dot(r2, r21)))/nr21
        gamma2 = np.log((nr2 * nr02 + np.dot(r2, r02))/(nr0 * nr02 + np.dot(r0, r02)))/nr02
        gamma = np.expand_dims(gamma2 - gamma0, axis = 1) * r0 + np.expand_dims(gamma0 - gamma1, axis = 1) * r1 + np.expand_dims(gamma1 - gamma2, axis = 1) * r2
        
        omega0 = (double_faces_area[face_idx] * solid_angles * np.sum(np.cross(r2, r1) * faces_normal[face_idx], axis = 1)
                  + np.sum(np.cross(r0, r1) * r2, axis = 1) * np.sum((r2 - r1) * gamma, axis = 1)) / (double_faces_area[face_idx]*double_faces_area[face_idx])
        omega1 = (double_faces_area[face_idx] * solid_angles * np.sum(np.cross(r0, r2) * faces_normal[face_idx], axis = 1)
                  + np.sum(np.cross(r0, r1) * r2, axis = 1) * np.sum((r0 - r2) * gamma, axis = 1)) / (double_faces_area[face_idx]*double_faces_area[face_idx])
        omega2 = (double_faces_area[face_idx] * solid_angles * np.sum(np.cross(r1, r0) * faces_normal[face_idx], axis = 1)
                  + np.sum(np.cross(r0, r1) * r2, axis = 1) * np.sum((r1 - r0) * gamma, axis = 1)) / (double_faces_area[face_idx]*double_faces_area[face_idx])
        
        loc_omega = np.asarray([omega0, omega1, omega2]).T
        loc_omega[face, :] = 0
        omega[:, face] -= loc_omega
    
    return omega

def find_diag_omega(omega, faces):
    half_missing_omega = (((2.0*np.pi) - np.sum(omega, axis = 1)) / 2)
    
    #Half the angle goes to r0, the other half is distributed amongst the respective faces
    omega[np.diag_indices_from(omega)] += half_missing_omega
    
    for ref_vertex_idx in range(omega.shape[0]):
        neigh_vertices = np.unique(faces[np.argwhere(faces == ref_vertex_idx)[:, 0], :].reshape(-1))
        neigh_vertices = neigh_vertices[neigh_vertices != ref_vertex_idx]
        
        omega[ref_vertex_idx, neigh_vertices] += (half_missing_omega[ref_vertex_idx]/len(neigh_vertices))
    
    return omega

def calc_bem_fields(vortex, rmags, cosmags):
    diff = np.expand_dims(rmags.T, axis = 0) - np.expand_dims(vortex, axis = [0, 2])
    norm_diff = np.power(np.linalg.norm(diff, axis = 1), 3)
    norm_diff = norm_diff.squeeze(0)
    norm_diff[norm_diff == 0] = 1
    norm_diff = np.expand_dims(norm_diff, axis = [0, 1])
    
    x = np.empty((1, 3, rmags.shape[0]))
    x[:, 0] = diff[:, 1] * cosmags[:, 2] - diff[:, 2] * cosmags[:, 1]
    x[:, 1] = diff[:, 2] * cosmags[:, 0] - diff[:, 0] * cosmags[:, 2]
    x[:, 2] = diff[:, 0] * cosmags[:, 1] - diff[:, 1] * cosmags[:, 0]
    
    x /= norm_diff
    
    return x

    