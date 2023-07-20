'''
Created on Oct 21, 2022

@author: voodoocode
'''

import os
import finnpy.source_reconstruction.utils
import shutil
from finnpy.source_reconstruction.coregistration_meg_mri import calc_coregistration

import nibabel.freesurfer
import numpy as np
import finnpy.source_reconstruction.utils as finnpy_utils

import mne.io

def extract_anatomy_from_mri_using_fs(subj_name, t1_scan_file, fiducials_file = None, fiducials_path = None, 
                                      overwrite = False):
    """
    Extracts anatomical structures from an mri scan using freesurfer.
    
    :param subj_name: Name of the subject.
    :param t1_scan_file: Name of the mri file.
    :param fiducials_file: Name of the fiducials file. If none is present, default fiducials are morphed from fs-average.
    :param overwrite: Flag whether to overwrite the files of the respective subject folder already exists.
    """
    if (subj_name[-1] == "/"):
        patient_id = subj_name[:-1]
    else:
        patient_id = subj_name
    
    old_base_dir = os.environ["SUBJECTS_DIR"] + "/" + subj_name + "/"
    new_base_dir = os.environ["SUBJECTS_DIR"] + "/" + subj_name + "_tmp" + "/"
    
    if (os.path.exists(old_base_dir) and overwrite == False):
        return
    
    cmd = [__file__[:__file__.rindex("/")] + "/fs_extract_anatomy.sh", subj_name, t1_scan_file]
    
    finnpy.source_reconstruction.utils.run_subprocess_in_custom_working_directory(patient_id, cmd)
    
    os.mkdir(new_base_dir)
    
    #Create watershed model folder
    os.mkdir(new_base_dir + "bem")
    os.mkdir(new_base_dir + "bem/watershed")
    if (fiducials_file is None):
        create_fiducials(old_base_dir, new_base_dir, subj_name)
    else:
        shutil.copyfile(fiducials_path + fiducials_file, new_base_dir + fiducials_file)
    
    #Create and populate mri folder
    os.mkdir(new_base_dir + "mri")
    os.mkdir(new_base_dir + "mri/transforms")
    shutil.copyfile(old_base_dir + "mri/" + "orig.mgz", new_base_dir + "mri/" + "orig.mgz")
    shutil.copyfile(old_base_dir + "mri/" + "T1.mgz", new_base_dir + "mri/" + "T1.mgz")
    shutil.copyfile(old_base_dir + "mri/transforms/" + "talairach.xfm", new_base_dir + "mri/transforms/" + "talairach.xfm")
    
    #Create and populate surface folder
    os.mkdir(new_base_dir + "surf")
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere", new_base_dir + "surf/" + "lh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere.reg", new_base_dir + "surf/" + "lh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "lh.white", new_base_dir + "surf/" + "lh.white")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere", new_base_dir + "surf/" + "rh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere.reg", new_base_dir + "surf/" + "rh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "rh.white", new_base_dir + "surf/" + "rh.white")
    
    shutil.rmtree(old_base_dir)
    shutil.move(new_base_dir, old_base_dir)

def create_fiducials(in_path, out_path, subj_name):
    """
    Reads fiducials from fs-average and transforms them from fs-average space into subject space.
    
    :param in_path: Path to the src folder.
    :param out_path: Path to the tgt folder.
    :param subj_name: Name of the subject.
    """
    
    #===========================================================================
    # Read fiducials from fs average
    #===========================================================================
    (pre_mri_ref_pts, coord_system) = mne.io.read_fiducials(mne.__file__[:mne.__file__.rindex("/")] + "/data/fsaverage/fsaverage-fiducials.fif")
    mri_ref_pts = finnpy_utils.format_fiducials(pre_mri_ref_pts)

    #===========================================================================
    # Move fiducials from fs-average space (MNI) into subject space (MRI)
    #===========================================================================
    trans_mat_ras_mni = np.zeros((4, 4))
    fid = open(in_path + "mri/transforms/talairach.xfm", "r")
    for line in fid:
        if (line == "Linear_Transform = \n" or line == "Linear_Transform =\n"):
            break
    trans_mat_ras_mni[0, :] = fid.readline().replace("\n", "").split(" ")[:4]
    trans_mat_ras_mni[1, :] = fid.readline().replace("\n", "").split(" ")[:4]
    trans_mat_ras_mni[2, :] = fid.readline().replace("\n", "").replace(";","").split(" ")[:4]
    fid.close()
    trans_mat_ras_mni[:3, 3] /= 1000 # scale from m to mm
    trans_mat_ras_mni[3, 3] = 1
    
    trans_mat_mri_ras = nibabel.freesurfer.load(in_path + "mri/orig.mgz")
    trans_mat_mri_ras = np.matmul(trans_mat_mri_ras.header.get_vox2ras(), np.linalg.inv(trans_mat_mri_ras.header.get_vox2ras_tkr()))
    trans_mat_mri_ras[:3, 3] /= 1000 # scale from m to mm
    
    trans_mat_mri_mni = np.matmul(trans_mat_ras_mni, trans_mat_mri_ras)
    trans_mat_mni_mri = np.linalg.inv(trans_mat_mri_mni)
    
    for mri_ref_pt_key in mri_ref_pts.keys():
        mri_ref_pts[mri_ref_pt_key] = np.dot(trans_mat_mni_mri[:3, :3], mri_ref_pts[mri_ref_pt_key]) + trans_mat_mni_mri[:3, 3]
    
    #===========================================================================
    # Write transformed fiducials into directory
    #===========================================================================
    formatted_mri_ref_pts = list()
    for mri_ref_pt_key in mri_ref_pts.keys():
        if (mri_ref_pt_key == "LPA"):
            ident = mne.io.constants.FIFF.FIFFV_POINT_LPA
        if (mri_ref_pt_key == "NASION"):
            ident = mne.io.constants.FIFF.FIFFV_POINT_NASION
        if (mri_ref_pt_key == "RPA"):
            ident = mne.io.constants.FIFF.FIFFV_POINT_RPA
        formatted_mri_ref_pts.append({"r" : mri_ref_pts[mri_ref_pt_key], "ident" : ident, "kind" : mne.io.constants.FIFF.FIFFV_POINT_CARDINAL})
    
    mne.io.write_fiducials(out_path + "bem/" + subj_name + "-fiducials.fif", formatted_mri_ref_pts, coord_system, overwrite = False)

def copy_fs_avg_anatomy(fs_path, subj_path, subj_name):
    """
    In case no mri scans are available for this subject, fs-average is used as a reference template.
    
    :param fs_path: Path to the freesurfer files (containing fs average).
    :param subj_path: Folder name of the subject.
    :param subj_name: Name of the subject.
    """
    old_base_dir = fs_path + "fsaverage" + "/"
    new_base_dir = fs_path + subj_name + "/"
    
    if (os.path.exists(new_base_dir)):
        return
    os.mkdir(new_base_dir)
     
    #Create and populate bem folder
    os.mkdir(new_base_dir + "bem")
    #shutil.copyfile(old_base_dir + "bem/" + "fsaverage-fiducials.fif", new_base_dir + "bem/" + subj_name + "-fiducials.fif")
    shutil.copyfile(mne.__file__[:mne.__file__.rindex("/")] + "/data/fsaverage/fsaverage-fiducials.fif", new_base_dir + "bem/" + subj_name + "-fiducials.fif")
    os.mkdir(new_base_dir + "bem/watershed")
     
    #Create and populate mri folder
    os.mkdir(new_base_dir + "mri")
    os.mkdir(new_base_dir + "mri/transforms")
    shutil.copyfile(old_base_dir + "mri/" + "orig.mgz", new_base_dir + "mri/" + "orig.mgz")
    shutil.copyfile(old_base_dir + "mri/" + "T1.mgz", new_base_dir + "mri/" + "T1.mgz")
    shutil.copyfile(old_base_dir + "mri/transforms/" + "talairach.xfm", new_base_dir + "mri/transforms/" + "talairach.xfm")
     
    #Create and populate surface folder
    os.mkdir(new_base_dir + "surf")
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere", new_base_dir + "surf/" + "lh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere.reg", new_base_dir + "surf/" + "lh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "lh.white", new_base_dir + "surf/" + "lh.white")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere", new_base_dir + "surf/" + "rh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere.reg", new_base_dir + "surf/" + "rh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "rh.white", new_base_dir + "surf/" + "rh.white")
    
def scale_anatomy(subj_path, subj_name, scale, overwrite = False):
    """
    Scales anatomy for a perfect fit between fiducials and anatomy. If files are already scaled, an .is_scaled file created. This flag can be ignore via setting overwrite to True. 
    
    :param subj_path: Path to the freesurfer created subject specific files.
    :param subj_name: Name of the subject.
    :param scale: Scaling to be applied (x, y, z). 
    :param overwrite: Flag to apply scaling even if scaling is already applied.
    """
    
    if (os.path.exists(subj_path + ".is_scaled") == True and overwrite == False):
        return
    
    if (os.path.exists(subj_path + "bem/" + subj_name + "-fiducials.fif")):
        load_scale_save_fiducials(subj_path + "bem/" + subj_name + "-fiducials.fif", scale)
    
    load_scale_save_surfaces(subj_path + "surf/" + "lh.white", scale)
    load_scale_save_surfaces(subj_path + "surf/" + "rh.white", scale)
    load_scale_save_surfaces(subj_path + "surf/" + "lh.seghead", scale)
    
    load_scale_save_mri(subj_path + "mri/" + "orig.mgz", scale)
    load_scale_save_mri(subj_path + "mri/" + "T1.mgz", scale)
    
    file = open(subj_path + ".is_scaled", "wb")
    file.close()

def load_scale_save_fiducials(path, scale):
    """
    Loads, scales, and saves fiducials.
    
    :param path: Path to the fiducials file.
    :param scale: Scale to be applied (x, y, z).
    """
    
    if (os.path.exists(path + "_unscaled")):
        path = path + "_unscaled"
     
    shutil.copyfile(path, path + "_unscaled")
    
    (fiducials, coord_system) = mne.io.read_fiducials(path)
    
    for pt_idx in range(len(fiducials)):
        fiducials[pt_idx]["r"] = fiducials[pt_idx]["r"] * scale
        
    mne.io.write_fiducials(path, fiducials, coord_system, overwrite = True)

def load_scale_save_surfaces(path, scale):
    """
    Load, scale, and save surfaces.
    
    :param path: Path to the subject's freesurfer created surface surfaces.
    :param scale: Scale to be applied (x, y, z).
    """
    
    if (os.path.exists(path + "_unscaled")):
        path = path + "_unscaled"
     
    shutil.copyfile(path, path + "_unscaled")
    
    (vert, faces) = nibabel.freesurfer.read_geometry(path)
    vert = vert * scale
    nibabel.freesurfer.write_geometry(path, vert, faces)

def load_scale_save_mri(path, scale):
    """
    Load, scale, and save MRI information.
    
    :param path: Path to the subject's freesurfer created mri files.
    :param scale: Scale to be applied (x, y, z).
    """
    
    if (os.path.exists(path + "_unscaled")):
        path = path + "_unscaled"
     
    shutil.copyfile(path, path + "_unscaled")
    
    mri = nibabel.load(path)
    
    zooms = np.array(mri.header.get_zooms())
    zooms[[0, 2, 1]] *= scale
    mri.header.set_zooms(zooms)
    mri._affine = mri.header.get_affine()
    
    nibabel.save(mri, path)

def visualize_anatomy_from_mri_using_fs(subj_path):
    """
    Employ freesurfer to visualize anatomy results extracted from freesurfer.
    
    :param subj_path: Path to the subject's freesurfer created files.
    """
    if (subj_path[-1] == "/"):
        patient_id = subj_path[:-1]
    else:
        patient_id = subj_path
    
    cmd = ["freeview", 
           "-v", "$SUBJECTS_DIR/$SUBJECT/mri/brainmask.mgz",
           "-v", "$SUBJECTS_DIR/$SUBJECT/mri/aseg.mgz:colormap=lut:opacity=0.2",
           "-f", "$SUBJECTS_DIR/$SUBJECT/surf/lh.white:edgecolor=yellow",
           "-f", "$SUBJECTS_DIR/$SUBJECT/surf/rh.white:edgecolor=yellow",
           "-f", "$SUBJECTS_DIR/$SUBJECT/surf/lh.pial:annot=aparc:edgecolor=red",
           "-f", "$SUBJECTS_DIR/$SUBJECT/surf/rh.pial:annot=aparc:edgecolor=red",
           "-all", "-qcache"]
    
    finnpy_utils.run_subprocess_in_custom_working_directory(patient_id, cmd)










