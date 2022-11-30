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

import mne.io

def extract_anatomy_from_mri_using_fs(subj_name, t1_scan_file):
    if (subj_name[-1] == "/"):
        patient_id = subj_name[:-1]
    else:
        patient_id = subj_name
    
    old_base_dir = os.environ["SUBJECTS_DIR"] + "/" + subj_name + "/"
    new_base_dir = os.environ["SUBJECTS_DIR"] + "/" + subj_name + "_tmp" + "/"
    
    if (os.path.exists(old_base_dir)):
        return
    
    cmd = [__file__[:__file__.rindex("/")] + "/fs_extract_anatomy.sh", subj_name, t1_scan_file]
    #===========================================================================
    # os.environ["FNAME"] = t1_scan_file
    # os.environ["SUBJECT"] = subj_name
    #===========================================================================
    
    finnpy.source_reconstruction.utils.run_subprocess_in_custom_working_directory(patient_id, cmd)
    
    os.mkdir(new_base_dir)
    
    #Create watershed model folder
    os.mkdir(new_base_dir + "bem")
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
    
    shutil.rmtree(old_base_dir)
    shutil.move(new_base_dir, old_base_dir)

def copy_fs_avg_anatomy(fs_path, subj_path, subj_name):
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
    
    if (os.path.exists(path + "_unscaled")):
        path = path + "_unscaled"
     
    shutil.copyfile(path, path + "_unscaled")
    
    (fiducials, coord_system) = mne.io.read_fiducials(path)
    
    for pt_idx in range(len(fiducials)):
        fiducials[pt_idx]["r"] = fiducials[pt_idx]["r"] * scale
        
    mne.io.write_fiducials(path, fiducials, coord_system, overwrite = True)

def load_scale_save_surfaces(path, scale):
    
    if (os.path.exists(path + "_unscaled")):
        path = path + "_unscaled"
     
    shutil.copyfile(path, path + "_unscaled")
    
    (vert, faces) = nibabel.freesurfer.read_geometry(path)
    vert = vert * scale
    nibabel.freesurfer.write_geometry(path, vert, faces)

def load_scale_save_mri(path, scale):
    
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
    
    source_reconstruction.utils.run_subprocess_in_custom_working_directory(patient_id, cmd)    

def cleanup_non_essential_fs_files(subj_path):
    pass










