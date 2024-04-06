'''
Created on Feb 22, 2024

@author: voodoocode
'''

import os
import shutil
import mne
import numpy as np
import nibabel.freesurfer

import finnpy.src_rec.utils

def init_fs_paths(fs_path, anatomy_path):
    """
    Runs freesurfer initialization steps. These are mandatory for successfull freesurfer exection.
    
    Parameters
    ----------
    fs_path : string
              Path to the freesurfer folder. Should contain the 'bin' folder, your license.txt, and sources.sh.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.

    """
    
    if (fs_path[-1] != "/"):
        fs_path += "/"
    
    os.environ["FREESURFER_HOME"]   = fs_path
    os.environ["FSFAST_HOME"]       = fs_path + "fsfast/"
    os.environ["FSF_OUTPUT_FORMAT"] = "nii.gz"
    os.environ["SUBJECTS_DIR"]      = anatomy_path[:-1] if (anatomy_path[-1] == "/") else anatomy_path
    os.environ["MNI_DIR"]           = fs_path + "mni/"
    
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FREESURFER_HOME']+"bin/"
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FSFAST_HOME']+"bin/"

def extract_mri_anatomy(anatomy_path, subj_name, t1_scan_file, fiducials_file = None, fiducials_path = None,
                        overwrite = False):
    """
    Extracts anatomical structures from an mri scan using freesurfer.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    t1_scan_file : string
                   Name of the mri file.
    fiducials_file : string
                     Name of the fiducials file. If none is present, default fiducials are morphed from fs-average, 
                     defaults to None.
    fiducials_path : string
                     Path to the fiducials file. If none is present, default fiducials are morphed from fs-average, 
                     defaults to None. 
    overwrite : boolean
                Flag whether to overwrite the files of the respective subject folder already exists, 
                defaults to False.
    """
    if (subj_name[-1] == "/"):
        patient_id = subj_name[:-1]
    else:
        patient_id = subj_name
    
    old_base_dir = anatomy_path + "/" + subj_name + "/"
    new_base_dir = anatomy_path + "/" + subj_name + "_tmp" + "/"
    
    if (os.path.exists(old_base_dir) and overwrite == False):
        return
    
    cmd = [__file__[:__file__.rindex("/")] + "/fs_extract_anatomy.sh", subj_name, t1_scan_file]
    finnpy.src_rec.utils.run_subprocess_in_custom_working_directory(patient_id, cmd)
    
    os.mkdir(new_base_dir)
    
    #Create watershed model folder
    os.mkdir(new_base_dir + "bem")
    os.mkdir(new_base_dir + "bem/watershed")
    if (fiducials_file is None):
        _create_fiducials(old_base_dir, new_base_dir, subj_name)
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

def _create_fiducials(in_path, out_path, subj_name):
    """
    Reads fiducials from fs-average and transforms them from fs-average space into subject space.
    
    Parameters
    ----------
    in_path : string
              Path to the src folder.
               
    out_path : string
               Path to the tgt folder.
               
    subj_name : string
                Name of the subject.
    """
    
    #===========================================================================
    # Read fiducials from fs average
    #===========================================================================
    (pre_mri_ref_pts, coord_system) = mne.io.read_fiducials(mne.__file__[:mne.__file__.rindex("/")] + "/data/fsaverage/fsaverage-fiducials.fif")
    mri_ref_pts = _format_fiducials(pre_mri_ref_pts)

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

def _format_fiducials(pre_mri_ref_pts):
    """
    Transforms an mne-fiducials object into an dictionary containing the fiducials.
    
    Parameters
    ----------
    pre_mri_ref_pts : list of dict(), obtained via mne.io.read_fiducials
                      MNE-formatted list of MRI fiducials.
                      
    Returns
    -------
    mri_ref_pts : dict(), ('LPA', 'NASION', 'RPA')
                  MRI reference points for coregistration.
    """
    mri_ref_pts = {"LPA" : None, "NASION" : None, "RPA" : None}
        
    for pt_idx in range(len(pre_mri_ref_pts)):
        if (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_LPA):
            mri_ref_pts["LPA"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif(pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION):
            mri_ref_pts["NASION"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif(pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_RPA):
            mri_ref_pts["RPA"] = pre_mri_ref_pts[pt_idx]["r"]
    
    return mri_ref_pts

def copy_fsavg_anatomy(fs_path, anatomy_path, subj_name, overwrite = False):
    """
    In case no mri scans are available for this subject, fs-average is used as a reference template.
    
    Parameters
    ----------
    fs_path : string
              Path to the freesurfer folder. Should contain the 'bin' folder, your license.txt, and sources.sh.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    overwrite : boolean
                Flag whether to overwrite the files of the respective subject folder already exists, 
                defaults to False.
               
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    old_base_dir  = (fs_path + "/") if (fs_path[-1] != "/") else fs_path
    old_base_dir += "subjects/fsaverage/"
    new_base_dir = anatomy_path + subj_name + "/"
    
    if (os.path.exists(new_base_dir) and overwrite == False):
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

def extract_skull_skin(anatomy_path, subject_name, preflood_height = 25, overwrite = False):
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
        finnpy.src_rec.utils.run_subprocess_in_custom_working_directory(subject_name, cmd)
        
        #Remove files not needed for source reconstruction
        os.remove(anatomy_path + subject_name + "/bem/watershed/" + subject_name + "_brain_surface")
        os.remove(anatomy_path + subject_name + "/bem/watershed/ws.mgz")

def calc_head_model(anatomy_path, subj_name):
    """
    Calculate a head model to read hd surface vertices using freesurfer. Removes files not needed by this reconstruction. 
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    """
    
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    cmd = [__file__[:__file__.rindex("/")] + "/fs_get_model.sh", subj_name]
    
    finnpy.src_rec.utils.run_subprocess_in_custom_working_directory(subj_name, cmd)
    
    if (os.path.exists(anatomy_path + subj_name + "/mri/" + "seghead.mgz") == True):
        os.remove(anatomy_path + subj_name + "/mri/" + "seghead.mgz")
    if (os.path.exists(anatomy_path + subj_name + "/scripts") == True):
        shutil.rmtree(anatomy_path + subj_name + "/scripts")
    if (os.path.exists(anatomy_path + subj_name + "/surf/" + "lh.seghead.inflated") == True):
        os.remove(anatomy_path + subj_name + "/surf/" + "lh.seghead.inflated")
