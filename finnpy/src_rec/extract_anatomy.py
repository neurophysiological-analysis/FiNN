"""
Created on Feb 22, 2024.

@author: voodoocode
"""

import os
import shutil
import mne
import numpy as np
import nibabel.freesurfer
import warnings
import matplotlib.pyplot as plt

import finnpy.misc.external_calls as ex_c  # @UnresolvedImport

def init_paths(freesurfer_path, anatomy_path,
               fastsurfer_path = None, fastsurfer_python_path = None, freesurfer_license_path = None):
    """
    Run freesurfer initialization steps. These are mandatory for successfull freesurfer exection.
    
    Parameters
    ----------
    freesurfer_path : string
                      Path to the freesurfer folder. Should contain the 'bin' folder, your license.txt, and sources.sh.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    fastsurfer_path : string
                      (optional) Path to the fastsurfer_path folder. Should contain 'run_fastsurfer.sh'.
    fastsurfer_python_path : string
                             (optional) Path to the python interpreter.
    freesurfer_license_path : string
                              (optional) Path to the freesurfer license file.

    """
    if (freesurfer_path[-1] != "/"):
        freesurfer_path += "/"
    
    os.environ["FREESURFER_HOME"]   = freesurfer_path  # noqa: E221
    os.environ["FSFAST_HOME"]       = freesurfer_path + "fsfast/"  # noqa: E221
    os.environ["FSF_OUTPUT_FORMAT"] = "nii.gz"  # noqa: E221
    os.environ["SUBJECTS_DIR"]      = anatomy_path[:-1] if (anatomy_path[-1] == "/") else anatomy_path  # noqa: E221
    os.environ["MNI_DIR"]           = freesurfer_path + "mni/"  # noqa: E221
    
    if (fastsurfer_path is not None):
        os.environ["FASTSURFER_HOME"] = fastsurfer_path
    if (fastsurfer_python_path is not None):
        os.environ["FASTSURFER_PYTHON_PATH"] = fastsurfer_python_path
    if (freesurfer_license_path is not None):
        os.environ["FREESURFER_LICENSE_PATH"] = freesurfer_license_path
    
    os.environ["PATH"] = os.environ["PATH"] + ":" + os.environ['FREESURFER_HOME'] + "bin/"
    os.environ["PATH"] = os.environ["PATH"] + ":" + os.environ['FSFAST_HOME'] + "bin/"

def extract_mri(anatomy_path, subj_name, t1_scan_file, fiducials_file = None, fiducials_path = None,
                        mode = "FreeSurfer",
                        overwrite = False):
    """
    Extract anatomical structures from an mri scan using freesurfer.
    
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
    mode : string
           mode is either "FreeSurfer" (default) or "FastSurfer".
    overwrite : boolean
                Flag whether to overwrite the files of the respective subject folder already exists, 
                defaults to False.
                
    Returns
    -------
    None
    """
    if (subj_name[-1] == "/"):
        patient_id = subj_name[:-1]
    else:
        patient_id = subj_name
    
    old_base_dir = anatomy_path + "/" + subj_name + "/"
    new_base_dir = anatomy_path + "/" + subj_name + "_tmp" + "/"
    
    if (os.path.exists(old_base_dir) and overwrite is False):
        return
    
    if (mode == "FastSurfer"):
        cmd = [__file__[:__file__.rindex("/")] + "/fastsurfer_extract_anatomy.sh", subj_name, t1_scan_file]
    else:
        cmd = [__file__[:__file__.rindex("/")] + "/freesurfer_extract_anatomy.sh", subj_name, t1_scan_file]
    ex_c.run(patient_id, cmd)
    
    os.mkdir(new_base_dir)
    
    # Create watershed model folder
    os.mkdir(new_base_dir + "bem")
    os.mkdir(new_base_dir + "bem/watershed")
    if (fiducials_file is None):
        _create_fiducials(old_base_dir, new_base_dir, subj_name)
    else:
        shutil.copyfile(fiducials_path + fiducials_file, new_base_dir + fiducials_file)
    
    # Create and populate mri folder
    os.mkdir(new_base_dir + "mri")
    os.mkdir(new_base_dir + "mri/transforms")
    shutil.copyfile(old_base_dir + "mri/" + "orig.mgz", new_base_dir + "mri/" + "orig.mgz")
    shutil.copyfile(old_base_dir + "mri/" + "T1.mgz", new_base_dir + "mri/" + "T1.mgz")
    shutil.copyfile(old_base_dir + "mri/transforms/" + "talairach.xfm", new_base_dir + "mri/transforms/" + "talairach.xfm")
     
    # Create and populate surface folder
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
    Read fiducials from fs-average and transforms them from fs-average space into subject space.
    
    Parameters
    ----------
    in_path : string
              Path to the src folder.
               
    out_path : string
               Path to the tgt folder.
               
    subj_name : string
                Name of the subject.
    """
    # Read fiducials from fs average
    (pre_mri_ref_pts, coord_system) = mne.io.read_fiducials(mne.__file__[:mne.__file__.rindex("/")] + "/data/fsaverage/fsaverage-fiducials.fif")  # @UndefinedVariable
    mri_ref_pts = _format_fiducials(pre_mri_ref_pts)

    # Move fiducials from fs-average space (MNI) into subject space (MRI)
    trans_mat_ras_mni = np.zeros((4, 4))
    fid = open(in_path + "mri/transforms/talairach.xfm", "r")  # pylint: disable=unspecified-encoding
    for line in fid:
        if (line == "Linear_Transform = \n" or line == "Linear_Transform =\n"):
            break
    trans_mat_ras_mni[0, :] = fid.readline().replace("\n", "").split(" ")[:4]
    trans_mat_ras_mni[1, :] = fid.readline().replace("\n", "").split(" ")[:4]
    trans_mat_ras_mni[2, :] = fid.readline().replace("\n", "").replace(";", "").split(" ")[:4]
    fid.close()
    trans_mat_ras_mni[:3, 3] /= 1000  # scale from m to mm
    trans_mat_ras_mni[3, 3] = 1
    
    trans_mat_mri_ras = nibabel.freesurfer.load(in_path + "mri/orig.mgz")
    trans_mat_mri_ras = np.matmul(trans_mat_mri_ras.header.get_vox2ras(), np.linalg.inv(trans_mat_mri_ras.header.get_vox2ras_tkr()))
    trans_mat_mri_ras[:3, 3] /= 1000  # scale from m to mm
    
    trans_mat_mri_mni = np.matmul(trans_mat_ras_mni, trans_mat_mri_ras)
    trans_mat_mni_mri = np.linalg.inv(trans_mat_mri_mni)
    
    for mri_ref_pt_key in mri_ref_pts.keys():
        mri_ref_pts[mri_ref_pt_key] = np.dot(trans_mat_mni_mri[:3, :3], mri_ref_pts[mri_ref_pt_key]) + trans_mat_mni_mri[:3, 3]
    
    # Write transformed fiducials into directory
    formatted_mri_ref_pts = list()
    for mri_ref_pt_key in mri_ref_pts.keys():
        if (mri_ref_pt_key == "LPA"):
            ident = mne.io.constants.FIFF.FIFFV_POINT_LPA
        if (mri_ref_pt_key == "NASION"):
            ident = mne.io.constants.FIFF.FIFFV_POINT_NASION
        if (mri_ref_pt_key == "RPA"):
            ident = mne.io.constants.FIFF.FIFFV_POINT_RPA
        formatted_mri_ref_pts.append({"r": mri_ref_pts[mri_ref_pt_key], "ident": ident, "kind": mne.io.constants.FIFF.FIFFV_POINT_CARDINAL})  # pylint: disable=possibly-used-before-assignment
    
    mne.io.write_fiducials(out_path + "bem/" + subj_name + "-fiducials.fif", formatted_mri_ref_pts, coord_system, overwrite = False)

def _format_fiducials(pre_mri_ref_pts):
    """
    Transform an mne-fiducials object into an dictionary containing the fiducials.
    
    Parameters
    ----------
    pre_mri_ref_pts : list of dict(), obtained via mne.io.read_fiducials
                      MNE-formatted list of MRI fiducials.
                      
    Returns
    -------
    mri_ref_pts : dict(), ('LPA', 'NASION', 'RPA')
                  MRI reference points for coregistration.
    """
    mri_ref_pts = {"LPA": None, "NASION": None, "RPA": None}
        
    for pt_idx in range(len(pre_mri_ref_pts)):
        if (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_LPA):
            mri_ref_pts["LPA"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION):
            mri_ref_pts["NASION"] = pre_mri_ref_pts[pt_idx]["r"] 
        elif (pre_mri_ref_pts[pt_idx]["ident"] == mne.io.constants.FIFF.FIFFV_POINT_RPA):
            mri_ref_pts["RPA"] = pre_mri_ref_pts[pt_idx]["r"]
    
    return mri_ref_pts

def copy_fsavg(freesurfer_path, anatomy_path, subj_name, overwrite = False):
    """
    In case no mri scans are available for this subject, fs-average is used as a reference template.
    
    Parameters
    ----------
    freesurfer_path : string
              Path to the freesurfer folder. Should contain the 'bin' folder, your license.txt, and sources.sh.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Name of the subject.
    overwrite : boolean
                Flag whether to overwrite the files of the respective subject folder already exists, 
                defaults to False.
    
    Returns
    -------
    None
    """
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    old_base_dir  = (freesurfer_path + "/") if (freesurfer_path[-1] != "/") else freesurfer_path  # noqa: E221
    old_base_dir += "subjects/fsaverage/"
    new_base_dir = anatomy_path + subj_name + "/"
    
    if (os.path.exists(new_base_dir) and overwrite is False):
        return
    os.mkdir(new_base_dir)
     
    # Create and populate bem folder
    os.mkdir(new_base_dir + "bem")
    # shutil.copyfile(old_base_dir + "bem/" + "fsaverage-fiducials.fif", new_base_dir + "bem/" + subj_name + "-fiducials.fif")
    shutil.copyfile(mne.__file__[:mne.__file__.rindex("/")] + "/data/fsaverage/fsaverage-fiducials.fif", new_base_dir + "bem/" + subj_name + "-fiducials.fif")  # @UndefinedVariable
    os.mkdir(new_base_dir + "bem/watershed")
     
    # Create and populate mri folder
    os.mkdir(new_base_dir + "mri")
    os.mkdir(new_base_dir + "mri/transforms")
    shutil.copyfile(old_base_dir + "mri/" + "orig.mgz", new_base_dir + "mri/" + "orig.mgz")
    shutil.copyfile(old_base_dir + "mri/" + "T1.mgz", new_base_dir + "mri/" + "T1.mgz")
    shutil.copyfile(old_base_dir + "mri/transforms/" + "talairach.xfm", new_base_dir + "mri/transforms/" + "talairach.xfm")
     
    # Create and populate surface folder
    os.mkdir(new_base_dir + "surf")
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere", new_base_dir + "surf/" + "lh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "lh.sphere.reg", new_base_dir + "surf/" + "lh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "lh.white", new_base_dir + "surf/" + "lh.white")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere", new_base_dir + "surf/" + "rh.sphere")
    shutil.copyfile(old_base_dir + "surf/" + "rh.sphere.reg", new_base_dir + "surf/" + "rh.sphere.reg")
    shutil.copyfile(old_base_dir + "surf/" + "rh.white", new_base_dir + "surf/" + "rh.white")

def get_skull_skin(anatomy_path, subject_name, preflood_height = 25, overwrite = False):
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
    
    if (overwrite is True or os.path.exists(anatomy_path + subject_name + "/bem/watershed/" + subject_name + "_inner_skull_surface") is False):
        
        cmd = ["mri_watershed", "-h", str(preflood_height), "-useSRAS", "-surf",
               anatomy_path + subject_name + "/bem/watershed/" + subject_name,
               anatomy_path + subject_name + "/mri/T1.mgz",
               anatomy_path + subject_name + "/bem/watershed/ws.mgz"]
        ex_c.run(subject_name, cmd)
        
        # Remove files not needed for source reconstruction
        os.remove(anatomy_path + subject_name + "/bem/watershed/" + subject_name + "_brain_surface")
        os.remove(anatomy_path + subject_name + "/bem/watershed/ws.mgz")

def get_head_model(anatomy_path, subj_name):
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
    
    cmd = [__file__[:__file__.rindex("/")] + "/freesurfer_get_model.sh", subj_name]
    
    ex_c.run(subj_name, cmd)
    
    if (os.path.exists(anatomy_path + subj_name + "/mri/" + "seghead.mgz") is True):
        os.remove(anatomy_path + subj_name + "/mri/" + "seghead.mgz")
    if (os.path.exists(anatomy_path + subj_name + "/scripts") is True):
        shutil.rmtree(anatomy_path + subj_name + "/scripts")
    if (os.path.exists(anatomy_path + subj_name + "/surf/" + "lh.seghead.inflated") is True):
        os.remove(anatomy_path + subj_name + "/surf/" + "lh.seghead.inflated")

def read_skin_skull(anatomy_path, subj_name, signal_type, coreg):
    """
    Read skull and skin models extracted via freesurfer's watershed algorithm.
    
    Parameters
    ----------
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a
                   sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    signal_type : string
                  Mode is either "EEG" or "MEG". 
    coreg : finnpy.src_rec.coreg.Coreg
            Container with different transformation matrices
               
    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
             - vert : np.ndarray(vert_cnt, 3) or [np.ndarray(out_skin_vert_cnt, 3), np.ndarray(out_skull_vert_cnt, 3), np.ndarray(in_skull_vert_cnt, 3)]
                      If signal_type is 'MEG': Vertices of the inner skull model. If signal_type is 'EEG': Vertices of the outer skin, outer skull and inner skull models.
             - faces : np.ndarray(face_cnt, 3) or [np.ndarray(out_skin_vert_cnt, 3), np.ndarray(out_skull_vert_cnt, 3), np.ndarray(in_skull_vert_cnt, 3)]
                       If signal_type is 'MEG': Faces of the inner skull model. If signal_type is 'EEG': Faces of the outer skin, outer skull and inner skull models.
    """
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (in_skull_vert, in_skull_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_inner_skull_surface")
    if (signal_type == "EEG"):
        (out_skull_vert, out_skull_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_outer_skull_surface")
        (out_skin_vert, out_skin_faces) = nibabel.freesurfer.read_geometry(anatomy_path + subj_name + "/bem/watershed/" + subj_name + "_outer_skin_surface")
    
    if (signal_type == "MEG"):
        in_skull_vert *= coreg.rotors[6:9]
    
    if (signal_type == "MEG"):
        return ([in_skull_vert,], [in_skull_faces,])
    elif (signal_type == "EEG"):
        return ([out_skin_vert, out_skull_vert, in_skull_vert], 
                [out_skin_faces, out_skull_faces, in_skull_faces])

def plot_skin_skull(vert, faces,
                    anatomy_path, subj_name, block = True):
    """
    Plot skull and skin models for visual confirmation of proper alignment.
               
    Parameters
    ----------
    vert : np.ndarray(vert_cnt, 3) or [np.ndarray(out_skin_vert_cnt, 3), np.ndarray(out_skull_vert_cnt, 3), np.ndarray(in_skull_vert_cnt, 3)]
           If signal_type is 'MEG': Vertices of the inner skull model. If signal_type is 'EEG': Vertices of the outer skin, outer skull and inner skull models.
    faces : np.ndarray(face_cnt, 3) or [np.ndarray(out_skin_vert_cnt, 3), np.ndarray(out_skull_vert_cnt, 3), np.ndarray(in_skull_vert_cnt, 3)]
            If signal_type is 'MEG': Faces of the inner skull model. If signal_type is 'EEG': Faces of the outer skin, outer skull and inner skull models.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a
                   sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    block : bool
            Whether to block the pyplot process and display the figure.
    """
    if (anatomy_path[-1] != "/"):
        anatomy_path += "/"
    
    (t1_data_trans, ras_to_mri) = _load_and_orient_t1(anatomy_path + subj_name + "/")
    mri_to_ras = np.linalg.inv(ras_to_mri)
    
    if len(vert == 1):
        in_skull_vert_trans = np.dot(vert[0].in_skull_vert, mri_to_ras[:3, :3].transpose()); in_skull_vert_trans += mri_to_ras[:3, 3]
    else:
        out_skin_vect_trans = np.dot(vert[0].out_skin_vect, mri_to_ras[:3, :3].transpose()); out_skin_vect_trans += mri_to_ras[:3, 3]
        out_skull_vert_trans = np.dot(vert[1].out_skull_vert, mri_to_ras[:3, :3].transpose()); out_skull_vert_trans += mri_to_ras[:3, 3]
        in_skull_vert_trans = np.dot(vert[2].in_skull_vert, mri_to_ras[:3, :3].transpose()); in_skull_vert_trans += mri_to_ras[:3, 3]
    
    if (len(faces) == 1):
        surfaces = [["inner_skull", "#FF0000", in_skull_vert_trans, faces[0].in_skull_faces],]
    else:
        surfaces = [["inner_skull", "#FF0000", in_skull_vert_trans, faces[2].in_skull_faces],
                    ["outer_skull", "#FFFF00", out_skull_vert_trans, faces[1].out_skull_faces],
                    ["outer_skin", "#FFAA80", out_skin_vect_trans, faces[0].out_skin_faces]]        
    
    (fig, axes) = plt.subplots(3, 4, gridspec_kw = {'wspace': 0.025, 'hspace': 0.025})
    _plot_subplots(t1_data_trans, axes, surfaces)
    fig.suptitle(subj_name)
    
    if (block):
        plt.show(block = True)

def _load_and_orient_t1(subject_path):
    """
    Load and orients an MRI scan.
    
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
    Add splices from the MRI scan onto the plot.
    
    Parameters
    ----------
    data : numpy.ndarray, shape(a, b, c)
           RAS oriented data.
    axes : list of matplotlib.axes
           Axes for plotting.
    surfaces : list of surfaces, ("Name", "Color", vertices, faces)
               List of surfices to draw.
    """
    (primary_dim, secondary_dim_x, secondary_dim_y) = (1, 0, 2)  # Plot in reference to coronal orientation.
    
    slices = np.asarray([[.12, .14, .17, .21],
                         [.26, .32, .39, .47],
                         [.53, .61, .68, .74],
                         [.79, .83, .86, .88]], dtype = np.float32) * data.shape[primary_dim]
    slices = np.asarray(slices, dtype = np.int32)
    
    for row_idx in range(3):
        for col_idx in range(4):
            axes[row_idx, col_idx].imshow(data[:, slices[row_idx, col_idx], :].T, cmap = plt.cm.gray, origin = "lower")  # pylint: disable=no-member, disable=undefined-variable
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
