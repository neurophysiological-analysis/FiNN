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

def calc_skull_and_skin_models(anatomy_path, subject_name, preflood_height = 25, overwrite = False):
    """
    Employs freesufers watershed algorithm to calculate skull and skin models.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subject_name : string
                   Subject name.
    preflood_height : int
                      Freesurfer parameter. May need adjusting if segmentation doesn't work properly.
    overwrite : boolean
                Flag to overwrite if files are already present. Defaults to False.
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    if (overwrite == True or os.path.exists(anatomy_path + subject_name + "/bem/watershed/" + subject_name + "_inner_skull_surface") == False):
        
        cmd = ["mri_watershed", "-h", str(preflood_height), "-useSRAS", "-surf",
               anatomy_path + subject_name + "/bem/watershed/" + subject_name,
               anatomy_path + subject_name + "/mri/T1.mgz",
               anatomy_path + subject_name + "/bem/watershed/ws.mgz"]
        finnpy.source_reconstruction.utils.run_subprocess_in_custom_working_directory(subject_name, cmd)
        
        #Remove files not needed for source reconstruction
        os.remove(anatomy_path + subject_name + "/bem/watershed/" + subject_name + "_brain_surface")
        os.remove(anatomy_path + subject_name + "/bem/watershed/ws.mgz")

def read_skull_and_skin_models(anatomy_path, subj_name):
    """
    Reads skull and skin models extracted
    via freesurfer's watershed algorithm.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a
                   sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
               
    Returns
    -------
    in_skull_vert : numpy.ndarray, shape(in_skull_vtx_cnt, 3)
                    Vertices of the inner skull model.
    in_skull_faces : numpy.ndarray, shape(in_skull_face_cnt, 3)
                     Faces of the inner skull model.
    out_skull_vert : numpy.ndarray, shape(out_skull_vtx_cnt, 3)
                     Vertices of the outer skull model.
    out_skull_faces : numpy.ndarray, shape(out_skull_face_cnt, 3)
                      Faces of the outer skull model.
    out_skin_vect : numpy.ndarray, shape(out_skin_vtx_cnt, 3)
                    Vertices of the skin model.
    out_skin_faces : numpy.ndarray, shape(out_skull_face_cnt, 3)
                     Faces of the skin model.
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (in_skull_vert, in_skull_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_inner_skull_surface")
    (out_skull_vert, out_skull_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_outer_skull_surface")
    (out_skin_vect, out_skin_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_outer_skin_surface")
    
    return (in_skull_vert, in_skull_faces,
            out_skull_vert, out_skull_faces,
            out_skin_vect, out_skin_faces)
    
def plot_skull_and_skin_models(in_skull_vert, in_skull_faces, 
                               out_skull_vert, out_skull_faces,
                               out_skin_vect, out_skin_faces, 
                               anatomy_path, subj_name):
    """
    Plot skull and skin models for visual confirmation of proper alignment.
               
    Parameters
    ----------
    in_skull_vert : numpy.ndarray, shape(in_skull_vtx_cnt, 3)
                    Vertices of the inner skull model.
    in_skull_faces : numpy.ndarray, shape(in_skull_face_cnt, 3)
                     Faces of the inner skull model.
    out_skull_vert : numpy.ndarray, shape(out_skull_vtx_cnt, 3)
                     Vertices of the outer skull model.
    out_skull_faces : numpy.ndarray, shape(out_skull_face_cnt, 3)
                      Faces of the outer skull model.
    out_skin_vect : numpy.ndarray, shape(out_skin_vtx_cnt, 3)
                    Vertices of the skin model.
    out_skin_faces : numpy.ndarray, shape(out_skull_face_cnt, 3)
                     Faces of the skin model.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a
                   sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (t1_data_trans, ras_to_mri) = _load_and_orient_t1(anatomy_path + subj_name + "/")
    mri_to_ras = np.linalg.inv(ras_to_mri)
    
    in_skull_vert_trans = np.dot(in_skull_vert, mri_to_ras[:3, :3].transpose()); in_skull_vert_trans += mri_to_ras[:3, 3]
    if (out_skull_vert is not None):
        out_skull_vert_trans = np.dot(out_skull_vert, mri_to_ras[:3, :3].transpose()); out_skull_vert_trans += mri_to_ras[:3, 3]
    if (out_skin_vect is not None):
        out_skin_vect_trans = np.dot(out_skin_vect, mri_to_ras[:3, :3].transpose()); out_skin_vect_trans += mri_to_ras[:3, 3]
    
    if (out_skull_vert is not None and out_skull_vert is not None):
        surfaces = [["inner_skull", "#FF0000", in_skull_vert_trans, in_skull_faces],
                    ["outer_skull", "#FFFF00", out_skull_vert_trans, out_skull_faces],
                    ["outer_skin", "#FFAA80", out_skin_vect_trans, out_skin_faces]]
    else:
        surfaces = [["inner_skull", "#FF0000", in_skull_vert_trans, in_skull_faces],]
    
    (_, axes) = plt.subplots(3, 4, gridspec_kw = {'wspace':0.025, 'hspace':0.025})
    _plot_skull_and_skin_subplots(t1_data_trans, axes, surfaces)
    plt.show(block = True)

def _load_and_orient_t1(subject_path):
    """
    Loads and orients an MRI scan.
    
    Parameters
    ----------
    subject_path : string
                   Path to the subject's T1 scan.
               
    Returns
    -------
    t1_image_trans.get_fdata() : numpy.ndarray, shape(a, b, c)
                                 Reoriented T1 scan.
    ras_to_mri : numpy.ndarray, shape(a, b, c)
                 RAS (right, anterior, superior) to MRI transformation.
    """
    
    if (subject_path[-1] != "/"):
        subject_path += "/"
    
    t1_img = nibabel.load(subject_path + "mri/T1.mgz")
    
    src_orientation = nibabel.orientations.aff2axcodes(t1_img.affine)
    tgt_orientation = ('R', 'A', 'S')
    trans_orientation = nibabel.orientations.ornt_transform(nibabel.orientations.axcodes2ornt(src_orientation),
                                                            nibabel.orientations.axcodes2ornt(tgt_orientation))
    
    aff_trans = nibabel.orientations.inv_ornt_aff(trans_orientation, t1_img.shape)
    t1_image_trans = t1_img.as_reoriented(trans_orientation)
    ras_to_mri = np.dot(t1_img.header.get_vox2ras_tkr(), aff_trans)
    
    return (t1_image_trans.get_fdata(), ras_to_mri)

def _plot_skull_and_skin_subplots(data, axes, surfaces):
    """
    Adds splices from the MRI scan onto the plot.
    
    Parameters
    ----------
    data : numpy.ndarray, shape(a, b, c)
           RAS oriented data.
    axes : list of matplotlib.axes
           Axes for plotting.
    surfaces : list of surfaces, ("Name", "Color", vertices, faces)
               List of surfices to draw.
    """
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

def calc_bem_model_linear_basis(vert, faces, tgt_icosahedron_level = 4):
    """
    Calcuates the BEM linear basis coefficients using the linear collocation method.
    
    Parameters
    ----------
    vert : numpy.ndarray, shape(vtx_cnt, 3)
           Vertices of a skin/skull model.
    faces : numpy.ndarray, shape(face_cnt, 3)
            Faces of a skin/skull model.
    tgt_icosahedron_level : int
                            Order of the icosahedron employed herein,
                            defaults to 4.
               
    Returns
    -------
    reduced_vert : numpy.ndarray, shape(scaled_vtx_cnt, 3)
                   Remaining Vertices of a skin/skull model vertices.
    faces : numpy.ndarray, shape(scaled_face_cnt, 3)
            Remaining Vertices of a skin/skull model faces.
    faces_area : numpy.ndarray, shape(scaled_face_cnt)
                 Surface area of the remaining faces.
    faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                   Normals of the individual remaining inner skull faces.
    bem_solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                   BEM solution (preliminary step for the calculation of the forward model).
    """
    src_icosahedron_level = int(np.log(faces.shape[0] / 20) / np.log(2) / 2)
    
    #Creates a model 
    #Icosahedrons are provided by freesurfer and reading those is faster than computing.
    (src_vert, _) = finnpy.source_reconstruction.sphere_model.read_sphere_from_icosahedron_in_fs_order(src_icosahedron_level)
    (tgt_vert, tgt_faces) = finnpy.source_reconstruction.sphere_model.calculate_sphere_from_icosahedron(tgt_icosahedron_level)
    
    #Reduce anatomy vertices/faces to relevant ones
    reduced_vert = np.copy(vert)[finnpy.source_reconstruction.utils.find_nearest_neighbor(src_vert, tgt_vert)[0]]
    faces = tgt_faces
    
    #Calculate matrix omega, containing "the solid angles subtended at the center of each triangle by all other triangles", see "Error Analysis of a New Galerkin Method to
    #Solve the Forward Problem in MEG and EEG Using the Boundary Element Method" by Satu Tissari, Jussi Rahola for more details equation #17.
    #In short, to calculate potentials using a BEM model, its coefficients/weights have to be calculated beforehand. 
    reduced_vert /= 1000 # Scale from m to mm
    x_pos = reduced_vert[faces[:, 0], :]
    y_pos = reduced_vert[faces[:, 1], :]
    z_pos = reduced_vert[faces[:, 2], :]
    faces_normal = finnpy.source_reconstruction.utils.fast_cross_product((y_pos - x_pos), (z_pos - x_pos))
    double_faces_area = finnpy.source_reconstruction.utils.magn_of_vec(faces_normal)
    n_faces_normal = np.linalg.norm(faces_normal, axis = 1)
    faces_normal[n_faces_normal > 0] = faces_normal[n_faces_normal > 0] / np.expand_dims(n_faces_normal[n_faces_normal > 0], axis = 1)
    omega = _find_non_diag_omega(reduced_vert, faces, double_faces_area, faces_normal)
    omega = _find_diag_omega(omega, faces)
    
    #The matrix "bem_solution" contains all linear basis functions
    #A "deflation" factor is added to replace the "zero" eigenvalue within to ensure invertability
    #See "EEG and MEG: Forward Solutions for Inverse Methods" by Mosher, 1999 for reference: "In E/MEG, the Neumann boundary condition used [...]"
    deflation_factor = 1/omega.shape[0]
    
    #Left part of equation #6 by Satu Tissari, Jussi Rahola (see above). 
    bem_solution = np.linalg.inv(np.eye(omega.shape[0]) + deflation_factor - omega / (2*np.pi))
    
    return (reduced_vert, faces, double_faces_area/2, faces_normal, bem_solution)

def _find_non_diag_omega(vert, faces, double_faces_area, faces_normal):
    """
    Calculates the non-diagonal elements of omega (linear basis factors), see function calc_bem_model.
    
    Parameters
    ----------
    vert : numpy.ndarray, shape(reduced_vtx_cnt, 3)
           Vertices used for linear basis function calculation.
    faces : numpy.ndarray, shape(face_cnt, 3)
            Vertices used for linear basis function calculation.
    double_faces_area : numpy.ndarray, shape(face_cnt)
                        Surface area of the remaining skin/skull model faces.
    faces_normal : numpy.ndarray, shape(face_cnt, 3)
                   Normals of the individual remaining skin/skull model faces.
               
    Returns
    -------
    omega : numpy.ndarray, shape(reduced_vtx_cnt, reduced_vtx_cnt)
            Linear basis factor elements of the BEM solution, precurser to the BEM solution proper.
            Warning, diagonal elements are invalid!
    """
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
        
        #=======================================================================
        # For reference
        #
        # See equation #17 of "Error Analysis of a New Galerkin Method to Solve the Forward Problem in MEG and EEG Using the Boundary Element Method"
        # by Satu Tissari, Jussi Rahola, 1998
        #=======================================================================
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

def _find_diag_omega(omega, faces):
    """
    Calculates the diagonal elements of omega as these cannot be calculated as the non-diagonal ones due to the auto solid angle problem.
    
    See "Error Analysis of a New Galerkin Method to Solve the Forward Problem in MEG and EEG Using the Boundary Element Method" by Satu Tissari, Jussi Rahola for reference.
    See function calc_bem_model for a more general description.
    
    Parameters
    ----------
    omega : numpy.ndarray, shape(reduced_vtx_cnt, reduced_vtx_cnt)
            Linear basis factor elements of the BEM solution, precurser to the BEM solution proper.
            Warning, diagonal elements are currently invalid!
    faces : numpy.ndarray, shape(face_cnt, 3)
            Remaining skin/skull model faces.
    
    Returns
    -------
    omega : numpy.ndarray, shape(reduced_vtx_cnt, reduced_vtx_cnt)
            Linear basis factor elements of the BEM solution with proper diagonal elements.
    """
    half_missing_omega = (((2.0*np.pi) - np.sum(omega, axis = 1)) / 2)
    
    #Half the angle goes to r0, the other half is distributed amongst the respective faces
    omega[np.diag_indices_from(omega)] += half_missing_omega
    
    for ref_vertex_idx in range(omega.shape[0]):
        neigh_vertices = np.unique(faces[np.argwhere(faces == ref_vertex_idx)[:, 0], :].reshape(-1))
        neigh_vertices = neigh_vertices[neigh_vertices != ref_vertex_idx]
        
        omega[ref_vertex_idx, neigh_vertices] += (half_missing_omega[ref_vertex_idx]/len(neigh_vertices))
    
    return omega

    