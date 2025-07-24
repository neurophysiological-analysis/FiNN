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

FAST_EIGEN_DECOMP_PATH = "<path_to_lib>"

IS_DEMO = True

OVERWRITE_FS_EXTRACT = False
OVERWRITE_WS_EXTRACT = False

VISUALIZE_COREG = True

def get_demo_noise_data(ch_cnt = 64, duration_s = 100, fs = 100):
    ch_names = ["eeg"] * ch_cnt
    unique_data = np.random.normal(size = (ch_cnt, int(duration_s * fs)))
    
    shared_data = np.repeat(np.random.normal(size = (1, int(duration_s * fs))), ch_cnt, axis = 0)
    
    data = unique_data * 0.8 + shared_data * 0.2
    
    return (data, fs, ch_names)
    

def get_demo_data(ch_cnt = 64, duration_s = 100, fs = 100):
    ch_names = ["eeg"] * ch_cnt
    unique_data = np.random.normal(size = (ch_cnt, int(duration_s * fs)))
    
    shared_data = np.repeat(np.random.normal(size = (1, int(duration_s * fs))), ch_cnt, axis = 0)
    
    data = unique_data * 0.9 + shared_data * 0.1
    
    return (data, fs, ch_names)


def main():
    """Demo pipeline for EEG source reconstruction."""
    # This line is only required if multiple source reconstructions are computed in parallel (recommended)
    threadpoolctl.threadpool_limits(1, user_api='blas')
    
    (sensor_cov_data, fs, ch_names) = get_demo_noise_data()
    
    if (os.path.exists("eeg_sen_cov") is False):
        sen_cov = finnpy.src_rec.sen_cov.run(sensor_cov_data.T, fs, "EEG", np.ones(sensor_cov_data.shape[0]), ch_names, ["eeg"] * sensor_cov_data.shape[0],
                                             fast_eigendecomp_path = FAST_EIGEN_DECOMP_PATH,
                                             float_sz = 256)
        dm.save(sen_cov, "eeg_sen_cov")
    else:
        sen_cov = dm.load("eeg_sen_cov")
    
    # Extract anatomy
    finnpy.src_rec.extract_anatomy.init_paths(FS_PATH, ANATOMY_PATH)
    finnpy.src_rec.extract_anatomy.extract_mri(ANATOMY_PATH, SUBJ_NAME, T1_PATH, overwrite = OVERWRITE_FS_EXTRACT)
    finnpy.src_rec.extract_anatomy.get_skull_skin(ANATOMY_PATH, SUBJ_NAME, preflood_height = 25, overwrite = OVERWRITE_WS_EXTRACT)
    finnpy.src_rec.extract_anatomy.get_head_model(ANATOMY_PATH, SUBJ_NAME)
    finnpy.src_rec.subj_to_fsavg.prepare(FS_PATH, ANATOMY_PATH, SUBJ_NAME)
    
    if (os.path.exists("eeg_coreg") is False):
        (coreg, _) = finnpy.src_rec.coreg.run(SUBJ_NAME, ANATOMY_PATH, "EEG", rec_info = "1020")
        dm.save(coreg, "eeg_coreg")
    else:
        coreg = dm.load("eeg_coreg")
    if (VISUALIZE_COREG):
        finnpy.src_rec.coreg.plot_coregistration(coreg, "EEG", ANATOMY_PATH, SUBJ_NAME)    
    
    if (os.path.exists("eeg_bem_mdl") is False):
        bem_mdl = finnpy.src_rec.bem_mdl.run(FS_PATH, ANATOMY_PATH, SUBJ_NAME, "EEG", coreg)
        dm.save(bem_mdl, "eeg_bem_mdl")
    else:
        bem_mdl = dm.load("eeg_bem_mdl")
        
    if (os.path.exists("eeg_cort_mdl") is False):
        cort_mdl = finnpy.src_rec.cort_mdl.get(ANATOMY_PATH, SUBJ_NAME, "EEG", coreg, bem_mdl)
        dm.save(cort_mdl, "eeg_cort_mdl")
    else:
        cort_mdl = dm.load("eeg_cort_mdl")
    
    if (os.path.exists("eeg_fwd_sol") is False):
        fwd_sol = finnpy.src_rec.fwd_mdl.compute(bem_mdl, cort_mdl, coreg, "EEG", ["1020", ch_names])
        dm.save(fwd_sol, "eeg_fwd_sol")
    else:
        fwd_sol = dm.load("eeg_fwd_sol")
    if (os.path.exists("eeg_rest_fwd_sol") is False):
        rest_fwd_sol = finnpy.src_rec.fwd_mdl.restrict(cort_mdl, fwd_sol, coreg)
        dm.save(rest_fwd_sol, "eeg_rest_fwd_sol")
    else:
        rest_fwd_sol = dm.load("eeg_rest_fwd_sol")
    if (os.path.exists("eeg_inv_mdl") is False):
        inv_mdl = finnpy.src_rec.inv_mdl.compute(sen_cov, rest_fwd_sol, "EEG", ["1020", ch_names])
        dm.save(inv_mdl, "eeg_inv_mdl")
    else:
        inv_mdl = dm.load("eeg_inv_mdl")
    
    if (os.path.exists("eeg_subj_to_fsavg_mdl") is False):
        subj_to_fsavg_mdl = finnpy.src_rec.subj_to_fsavg.compute(cort_mdl, ANATOMY_PATH, SUBJ_NAME, FS_PATH, False)
        dm.save(subj_to_fsavg_mdl, "eeg_subj_to_fsavg_mdl")
    else:
        subj_to_fsavg_mdl = dm.load("eeg_subj_to_fsavg_mdl")
    
    (sensor_data, fs, ch_names) = get_demo_data()
    
    # Preprocess sensor data
    
    if (IS_DEMO):
        sensor_data[ch_names.index("C4"), :] += 10000
            
    src_data = finnpy.src_rec.inv_mdl.apply(sensor_data, inv_mdl)
    src_fsavg_data = finnpy.src_rec.subj_to_fsavg.apply(subj_to_fsavg_mdl, src_data)
    (src_avg_data, morphed_channels, region_names) = finnpy.src_rec.avg_src_reg.run(src_fsavg_data, subj_to_fsavg_mdl, FS_PATH)

    print("Plot 1")
    color_data = np.mean(np.abs(src_data), axis = 1)
    psr.plot_subj_space(cort_mdl, color_data, "EEG", ["1020", ch_names], coreg, ch_names)
        
    print("Plot 2")
    color_data = np.mean(np.abs(src_fsavg_data), axis = 1)
    psr.plot_fsavg_space(subj_to_fsavg_mdl, color_data, "EEG", ["1020", ch_names], coreg, ch_names)
    
    print("Plot 3")
    color_data = np.mean(np.abs(src_avg_data), axis = 1)
    psr.plot_reg_avg(subj_to_fsavg_mdl, morphed_channels, color_data, "EEG", ["1020", ch_names], coreg, ch_names)

    print("Plot 4")
    plt.bar(region_names, np.sum(np.abs(src_avg_data), axis = 1))
    plt.xticks(rotation = 25)
    plt.show()
