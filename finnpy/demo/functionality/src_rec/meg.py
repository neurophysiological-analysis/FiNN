"""
Created on Jun 3, 2025.

@author: voodoocode
"""

import threadpoolctl
import os
import numpy as np
import matplotlib.pyplot as plt

import finnpy.file_io.data_manager as dm  # @UnresolvedImport
import finnpy.src_rec.sen_cov  # @UnresolvedImport
import finnpy.src_rec.coreg  # @UnresolvedImport
import finnpy.src_rec.bem_mdl  # @UnresolvedImport
import finnpy.src_rec.cort_mdl  # @UnresolvedImport
import finnpy.src_rec.fwd_mdl  # @UnresolvedImport
import finnpy.src_rec.inv_mdl  # @UnresolvedImport
import finnpy.src_rec.subj_to_fsavg  # @UnresolvedImport
import finnpy.visualization.plot_src_rec as psr  # @UnresolvedImport

ANATOMY_PATH = "./anatomy/"
SUBJ_NAME = "demo_pat"
FS_PATH = "<path_to_freesurfer>"
T1_PATH = "<path_to_T1>"
FIF_FILE = "<path_to_FIF>"

FAST_EIGEN_DECOMP_PATH = "<path_to_lib>"

IS_DEMO = True

OVERWRITE_FS_EXTRACT = False
OVERWRITE_WS_EXTRACT = False

VISUALIZE_COREG = True

def main():
    """Demo pipeline for MEG source reconstruction."""
    # This line is only required if multiple source reconstructions are computed in parallel (recommended)
    threadpoolctl.threadpool_limits(1, user_api='blas')
    
    (sensor_data, fs, ch_names, ch_types) = get_data()  # noqa: F821 @UndefinedVariable
    sensor_data = sensor_data[:, :int(fs * 10)]
    
    if (os.path.exists("meg_sen_cov") is False):
        sen_cov = finnpy.src_rec.sen_cov.run(sensor_data.T, fs, "MEG", np.ones(sensor_data.shape[0]), ch_names, ch_types, 
                                             fast_eigendecomp_path = FAST_EIGEN_DECOMP_PATH,
                                             float_sz = 256)
        dm.save(sen_cov, "meg_sen_cov")
    else:
        sen_cov = dm.load("meg_sen_cov")
    
    # Extract anatomy
    finnpy.src_rec.extract_anatomy.init_paths(FS_PATH, ANATOMY_PATH)
    finnpy.src_rec.extract_anatomy.extract_mri(ANATOMY_PATH, SUBJ_NAME, T1_PATH, overwrite = OVERWRITE_FS_EXTRACT)
    finnpy.src_rec.extract_anatomy.get_skull_skin(ANATOMY_PATH, SUBJ_NAME, preflood_height = 25, overwrite = OVERWRITE_WS_EXTRACT)
    finnpy.src_rec.extract_anatomy.get_head_model(ANATOMY_PATH, SUBJ_NAME)
    finnpy.src_rec.subj_to_fsavg.prepare(FS_PATH, ANATOMY_PATH, SUBJ_NAME)
    
    if (os.path.exists("meg_coreg") is False):
        (coreg, _) = finnpy.src_rec.coreg.run(SUBJ_NAME, ANATOMY_PATH, "MEG", rec_info = FIF_FILE)
        dm.save(coreg, "meg_coreg")
    else:
        coreg = dm.load("meg_coreg")
    
    if (VISUALIZE_COREG):
        finnpy.src_rec.coreg.plot_coregistration(coreg, "MEG", ANATOMY_PATH, SUBJ_NAME, meg_data_path = "/mnt/data/Professional/UHN/projects/data/MEG-AD/recordings/sen/raw/al00102/a/tsss/01_tsss_1_OFF-1.fif")
    
    if (os.path.exists("meg_bem_mdl") is False):
        bem_mdl = finnpy.src_rec.bem_mdl.run(FS_PATH, ANATOMY_PATH, SUBJ_NAME, "MEG", coreg)
        dm.save(bem_mdl, "meg_bem_mdl")
    else:
        bem_mdl = dm.load("meg_bem_mdl")
        
    if (os.path.exists("meg_cort_mdl") is False):
        cort_mdl = finnpy.src_rec.cort_mdl.get(ANATOMY_PATH, SUBJ_NAME, "MEG", coreg, bem_mdl)
        dm.save(cort_mdl, "meg_cort_mdl")
    else:
        cort_mdl = dm.load("meg_cort_mdl")
    
    if (os.path.exists("meg_fwd_sol") is False):
        fwd_sol = finnpy.src_rec.fwd_mdl.compute(bem_mdl, cort_mdl, coreg, "MEG", FIF_FILE)
        dm.save(fwd_sol, "meg_fwd_sol")
    else:
        fwd_sol = dm.load("meg_fwd_sol")

    if (os.path.exists("meg_rest_fwd_sol") is False):
        rest_fwd_sol = finnpy.src_rec.fwd_mdl.restrict(cort_mdl, fwd_sol, coreg)
        dm.save(rest_fwd_sol, "meg_rest_fwd_sol")
    else:
        rest_fwd_sol = dm.load("meg_rest_fwd_sol")
    
    if (os.path.exists("meg_inv_mdl") is False):
        inv_mdl = finnpy.src_rec.inv_mdl.compute(sen_cov, rest_fwd_sol, "MEG", FIF_FILE)
        dm.save(inv_mdl, "meg_inv_mdl")
    else:
        inv_mdl = dm.load("meg_inv_mdl")
    
    if (os.path.exists("meg_subj_to_fsavg_mdl") is False):
        subj_to_fsavg_mdl = finnpy.src_rec.subj_to_fsavg.compute(cort_mdl, ANATOMY_PATH, SUBJ_NAME, FS_PATH, False)
        dm.save(subj_to_fsavg_mdl, "meg_subj_to_fsavg_mdl")
    else:
        subj_to_fsavg_mdl = dm.load("meg_subj_to_fsavg_mdl")
    
    (sensor_data, fs, ch_names) = get_data()  # noqa: F821 @UndefinedVariable
    
    # Preprocess sensor data
    
    if (IS_DEMO):
        sensor_data[ch_names.index("MEG0133"), :] += 10000
        
    src_data = finnpy.src_rec.inv_mdl.apply(sensor_data, inv_mdl)
    src_fsavg_data = finnpy.src_rec.subj_to_fsavg.apply(subj_to_fsavg_mdl, src_data)
    (src_avg_data, morphed_channels, region_names) = finnpy.src_rec.avg_src_reg.run(src_fsavg_data, subj_to_fsavg_mdl, FS_PATH)
    
    print("Plot 1")
    color_data = np.mean(np.abs(src_data), axis = 1)
    psr.plot_subj_space(cort_mdl, color_data, "MEG", FIF_FILE, coreg, ch_names)
    
    print("Plot 2")
    color_data = np.mean(np.abs(src_fsavg_data), axis = 1)
    psr.plot_fsavg_space(subj_to_fsavg_mdl, color_data, "MEG", FIF_FILE, coreg, ch_names)
    
    print("Plot 3")
    color_data = np.mean(np.abs(src_avg_data), axis = 1)
    psr.plot_reg_avg(subj_to_fsavg_mdl, morphed_channels, color_data, "MEG", FIF_FILE, coreg, ch_names)
    
    print("Plot 4")
    plt.bar(region_names, np.sum(np.abs(src_avg_data), axis = 1))
    plt.xticks(rotation=25)
    plt.show()
