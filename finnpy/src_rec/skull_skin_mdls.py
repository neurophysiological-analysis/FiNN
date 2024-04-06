'''
Created on Feb 22, 2024

@author: voodoocode
'''

import os
import numpy as np
import nibabel.freesurfer
import matplotlib.pyplot as plt
import warnings

import finnpy.src_rec.utils

class Skull_skin_mdls():
    """
    Container class populated with the following elements:
    
    in_skull_vert : numpy.ndarray, shape(in_skull_vtx_cnt, 3)
                    Vertices of the inner skull model.
    in_skull_faces : numpy.ndarray, shape(in_skull_face_cnt, 3)
                     Faces of the inner skull model.
    out_skull_vert : numpy.ndarray, shape(out_skull_vtx_cnt, 3) - optional
                     Vertices of the outer skull model.
    out_skull_faces : numpy.ndarray, shape(out_skull_face_cnt, 3) - optional
                      Faces of the outer skull model.
    out_skin_vect : numpy.ndarray, shape(out_skin_vtx_cnt, 3) - optional
                    Vertices of the skin model.
    out_skin_faces : numpy.ndarray, shape(out_skull_face_cnt, 3) - optional
                     Faces of the skin model.
    """
    def __init__(self, in_skull_vert, in_skull_faces,
                 out_skull_vert = None, out_skull_faces = None,
                 out_skin_vect = None, out_skin_faces = None):
        
        self.in_skull_vert = in_skull_vert
        self.in_skull_faces = in_skull_faces
        
        self.out_skull_vert = out_skull_vert
        self.out_skull_faces = out_skull_faces
        
        self.out_skin_vect = out_skin_vect
        self.out_skin_faces = out_skin_faces

def read(anatomy_path, subj_name, mode, coreg = None):
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
    mode : string
           Mode is either 'EEG', 'MEG', 'MEEG' or 'full. 'MEG' populates only in_skull_vert and in_skull_faces.
           The other modes also poulated outer skull and outer skin models. 
    coreg : finnpy.src_rec.coreg.Coreg
            Container with different transformation matrices
               
    Returns
    -------
    skull_skin_mdl : finnpy.src_rec.bem_mdl.Skull_skin_mdl 
                     Container class.
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (in_skull_vert, in_skull_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_inner_skull_surface")
    if (mode == "EEG" or mode == "MEEG" or mode == "full"):
        (out_skull_vert, out_skull_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_outer_skull_surface")
        (out_skin_vect, out_skin_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_outer_skin_surface")
    
    if (coreg is not None):
        in_skull_vert *= coreg.rotors[6:9]
        if (mode == "EEG" or mode == "MEEG" or mode == "full"):
            out_skull_vert *= coreg.rotors[6:9]
            out_skin_vect *= coreg.rotors[6:9]
    
    if (mode == "MEG"):
        return Skull_skin_mdls(in_skull_vert, in_skull_faces)
    if (mode == "EEG" or mode == "MEEG" or mode == "full"):
        return Skull_skin_mdls(in_skull_vert, in_skull_faces,
                              out_skull_vert, out_skull_faces,
                              out_skin_vect, out_skin_faces)
    
def plot(skull_skin_mdl, anatomy_path, subj_name, block = True):
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
    
    in_skull_vert_trans = np.dot(skull_skin_mdl.in_skull_vert, mri_to_ras[:3, :3].transpose()); in_skull_vert_trans += mri_to_ras[:3, 3]
    if (skull_skin_mdl.out_skull_vert is not None):
        out_skull_vert_trans = np.dot(skull_skin_mdl.out_skull_vert, mri_to_ras[:3, :3].transpose()); out_skull_vert_trans += mri_to_ras[:3, 3]
    if (skull_skin_mdl.out_skin_vect is not None):
        out_skin_vect_trans = np.dot(skull_skin_mdl.out_skin_vect, mri_to_ras[:3, :3].transpose()); out_skin_vect_trans += mri_to_ras[:3, 3]
    
    if (skull_skin_mdl.out_skull_vert is not None and skull_skin_mdl.out_skull_vert is not None):
        surfaces = [["inner_skull", "#FF0000", in_skull_vert_trans, skull_skin_mdl.in_skull_faces],
                    ["outer_skull", "#FFFF00", out_skull_vert_trans, skull_skin_mdl.out_skull_faces],
                    ["outer_skin", "#FFAA80", out_skin_vect_trans, skull_skin_mdl.out_skin_faces]]
    else:
        surfaces = [["inner_skull", "#FF0000", in_skull_vert_trans, skull_skin_mdl.in_skull_faces],]
    
    (fig, axes) = plt.subplots(3, 4, gridspec_kw = {'wspace':0.025, 'hspace':0.025})
    _plot_subplots(t1_data_trans, axes, surfaces)
    fig.suptitle(subj_name)
    
    if (block):
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

def _plot_subplots(data, axes, surfaces):
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
