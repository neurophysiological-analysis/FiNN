'''
Created on Jan 23, 2024

@author: voodoocode
'''

import finnpy.source_reconstruction.utils
import finnpy.source_reconstruction.mri_anatomy
import finnpy.source_reconstruction.bem_model
import finnpy.source_reconstruction.source_mesh_model

import mne
import finnpy.source_reconstruction.utils
import finnpy.source_reconstruction.source_mesh_model
import finnpy.source_reconstruction.coregistration_meg_mri
import finnpy.source_reconstruction.mri_anatomy
import scipy
import finnpy.source_reconstruction.forward_model
import finnpy.source_reconstruction.sensor_covariance
import finnpy.source_reconstruction.inverse_model

import finnpy.source_reconstruction.source_region_model
import numpy as np
import finnpy.file_io.data_manager as dm

import os
import shutil

class Auto_reconstruct():
    
    def __init__(self, fs_path, anatomy_path, subj_name, t1_path, meg_data_path, sensor_cov_path, cov_path):
        
        self.fs_path = fs_path if (fs_path[-1] == "/") else (fs_path + "/")
        self.anatomy_path = anatomy_path if (anatomy_path[-1] == "/") else (anatomy_path + "/")
        self.subj_name = subj_name
        self.t1_path = t1_path
        self.meg_data_path = meg_data_path
        self.sensor_cov_path = sensor_cov_path
        self.cov_path = cov_path
        
        self.progress = {"fs_extract"               : False,

                         "run_coreg"                : False,                    
                         "ws_extract"               : False,
                         "calc_bem_linear_basis"    : False,
                         "refine_cortical_model"    : False,
                         "fwd_model"                : False,
                         "inv_model"                : False,
                    
                         "comp_fs_avg_trans"        : False}
        
        self.visualization = {"coregistration"      : False,
                              "watershed"           : False}
    
    def reset_progress(self, ref_title):
        for progress_key in list(self.progress.keys())[::-1]:
            self.progress[progress_key] = False
            if (progress_key == ref_title):
                break
    
    def fs_extract(self, overwrite_fs_extract):
        if (overwrite_fs_extract == True and os.path.exists(self.anatomy_path + self.subj_name)):
            shutil.rmtree(self.anatomy_path + self.subj_name)
            self.reset_progress("fs_extract")
            
        if (os.path.exists(self.t1_path) == False):
            finnpy.source_reconstruction.mri_anatomy.copy_fs_avg_anatomy(self.fs_path, self.anatomy_path, SUBJ_NAME, overwrite = overwrite_fs_extract)
        else:
            finnpy.source_reconstruction.mri_anatomy.extract_anatomy_from_mri_using_fs(self.subj_name, self.t1_path, overwrite = overwrite_fs_extract)
        self.progress["fs_extract"] = True
    
    def run_coreg(self, visualize_coregistration):
        if (self.progress["run_coreg"] == False):
            self.rec_meta_info = mne.io.read_info(self.meg_data_path)
            (self.coreg_rotors, meg_pts) = finnpy.source_reconstruction.coregistration_meg_mri.calc_coreg(SUBJ_NAME, ANATOMY_PATH, self.rec_meta_info, registration_scale_type = "free")
            finnpy.source_reconstruction.mri_anatomy.scale_anatomy(ANATOMY_PATH, SUBJ_NAME, self.coreg_rotors[6:9])
            (self.coreg_rotors, meg_pts) = finnpy.source_reconstruction.coregistration_meg_mri.calc_coreg(SUBJ_NAME, ANATOMY_PATH, self.rec_meta_info, registration_scale_type = "restricted")
            
            self.progress["run_coreg"] = True
            dm.save(self.coreg_rotors, self.anatomy_path + self.subj_name + "/tmp/" + "coreg")
            dm.save(meg_pts, self.anatomy_path + self.subj_name + "/tmp/" + "meg_pts")
        else:
            self.coreg_rotors = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "coreg")
            meg_pts = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "meg_pts")
        
        if (visualize_coregistration):
            print("Visualizing coregistration")
            rigid_mri_to_meg_trans = finnpy.source_reconstruction.coregistration_meg_mri.get_rigid_transform(self.coreg_rotors)
            finnpy.source_reconstruction.coregistration_meg_mri.plot_coregistration(rigid_mri_to_meg_trans, self.rec_meta_info, meg_pts, self.anatomy_path, self.subj_name)
    
    def ws_extract(self, overwrite_ws_extract, visualize_skull_skin_plots):
        if (overwrite_ws_extract):
            self.reset_progress("ws_extract")
        
        if (self.progress["ws_extract"] == False):
            finnpy.source_reconstruction.bem_model.calc_skull_and_skin_models(self.anatomy_path, self.subj_name, overwrite = overwrite_ws_extract)
            (self.ws_in_skull_vert, self.ws_in_skull_faces, 
             ws_out_skull_vert, ws_out_skull_faces,
             ws_out_skin_vect, ws_out_skin_faces) = finnpy.source_reconstruction.bem_model.read_skull_and_skin_models(self.anatomy_path, self.subj_name)
             
            self.progress["ws_extract"] = True
            dm.save(self.ws_in_skull_vert, self.anatomy_path + self.subj_name + "/tmp/" + "ws_in_skull_vert")
            dm.save(self.ws_in_skull_faces, self.anatomy_path + self.subj_name + "/tmp/" + "ws_in_skull_faces")
            dm.save(ws_out_skull_vert, self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skull_vert")
            dm.save(ws_out_skull_faces, self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skull_faces")
            dm.save(ws_out_skin_vect, self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skin_vect")
            dm.save(ws_out_skin_faces, self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skin_faces")
        else:
            self.ws_in_skull_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "ws_in_skull_vert")
            self.ws_in_skull_faces = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "ws_in_skull_faces")
            ws_out_skull_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skull_vert")
            ws_out_skull_faces = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skull_faces")
            ws_out_skin_vect = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skin_vect")
            ws_out_skin_faces = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "ws_out_skin_faces")
        
        if (visualize_skull_skin_plots):
            finnpy.source_reconstruction.bem_model.plot_skull_and_skin_models(self.ws_in_skull_vert, self.ws_in_skull_faces,
                                                                              ws_out_skull_vert, ws_out_skull_faces,
                                                                              ws_out_skin_vect, ws_out_skin_faces,
                                                                              self.anatomy_path, self.subj_name)
    
    def calc_bem_linear_basis(self):
        if (self.progress["calc_bem_linear_basis"] == False):
            (self.in_skull_reduced_vert, self.in_skull_faces, 
             self.in_skull_faces_area, self.in_skull_faces_normal, 
             self.bem_solution) = finnpy.source_reconstruction.bem_model.calc_bem_model_linear_basis(self.ws_in_skull_vert, self.ws_in_skull_faces)
            
            self.progress["calc_bem_linear_basis"] = True
            dm.save(self.in_skull_reduced_vert, self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_reduced_vert")
            dm.save(self.in_skull_faces, self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_faces")
            dm.save(self.in_skull_faces_area, self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_faces_area")
            dm.save(self.in_skull_faces_normal, self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_faces_normal")
            dm.save(self.bem_solution, self.anatomy_path + self.subj_name + "/tmp/" + "bem_solution")
        else:
            self.in_skull_reduced_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_reduced_vert")
            self.in_skull_faces = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_faces")
            self.in_skull_faces_area = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_faces_area")
            self.in_skull_faces_normal = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "in_skull_faces_normal")
            self.bem_solution = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "bem_solution")
    
    def refine_cortical_model(self):
        if (self.progress["refine_cortical_model"] == False):
            (self.lh_white_vert, self.lh_white_faces,
             self.rh_white_vert, self.rh_white_faces,
             self.lh_sphere_vert,
             self.rh_sphere_vert) = finnpy.source_reconstruction.utils.read_cortical_models(self.anatomy_path, self.subj_name)
            (self.octa_model_vert, self.octa_model_faces) = finnpy.source_reconstruction.source_mesh_model.create_source_mesh_model()
            (self.lh_white_valid_vert, self.rh_white_valid_vert) = finnpy.source_reconstruction.source_mesh_model.match_source_mesh_model(self.lh_sphere_vert,
                                                                                                                                          self.rh_sphere_vert,
                                                                                                                                          self.octa_model_vert)
            
            self.progress["refine_cortical_model"] = True
            dm.save(self.lh_white_vert, self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_vert")
            dm.save(self.lh_white_faces, self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_faces")
            dm.save(self.rh_white_vert, self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_vert")
            dm.save(self.rh_white_faces, self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_faces")
            dm.save(self.lh_sphere_vert, self.anatomy_path + self.subj_name + "/tmp/" + "lh_sphere_vert")
            dm.save(self.rh_sphere_vert, self.anatomy_path + self.subj_name + "/tmp/" + "rh_sphere_vert")
            dm.save(self.octa_model_vert, self.anatomy_path + self.subj_name + "/tmp/" + "octa_model_vert")
            dm.save(self.octa_model_faces, self.anatomy_path + self.subj_name + "/tmp/" + "octa_model_faces")
            dm.save(self.lh_white_valid_vert, self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_valid_vert")
            dm.save(self.rh_white_valid_vert, self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_valid_vert")
        else:
            self.lh_white_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_vert")
            self.lh_white_faces = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_faces")
            self.rh_white_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_vert")
            self.rh_white_faces = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_faces")
            self.lh_sphere_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "lh_sphere_vert")
            self.rh_sphere_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "rh_sphere_vert")
            self.octa_model_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "octa_model_vert")
            self.octa_model_faces = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "octa_model_faces")
            self.lh_white_valid_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_valid_vert")
            self.rh_white_valid_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_valid_vert")
    
    def fwd_model(self):
        if (self.progress["fwd_model"] == False):
            rigid_mri_to_meg_trans = finnpy.source_reconstruction.coregistration_meg_mri.get_rigid_transform(self.coreg_rotors)
            rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
            (fwd_sol,
             self.lh_white_valid_vert, self.rh_white_valid_vert) = finnpy.source_reconstruction.forward_model.calc_forward_model(self.lh_white_vert, self.rh_white_vert,
                                                                                                                                 rigid_meg_to_mri_trans, rigid_mri_to_meg_trans,
                                                                                                                                 self.rec_meta_info, self.in_skull_reduced_vert,
                                                                                                                                 self.in_skull_faces, self.in_skull_faces_normal,
                                                                                                                                 self.in_skull_faces_area, self.bem_solution,
                                                                                                                                 self.lh_white_valid_vert, self.rh_white_valid_vert)
            self.optimized_fwd_sol = finnpy.source_reconstruction.forward_model.optimize_fwd_model(self.lh_white_vert, self.lh_white_faces, self.lh_white_valid_vert,
                                                                                                   self.rh_white_vert, self.rh_white_faces, self.rh_white_valid_vert,
                                                                                                   fwd_sol, rigid_mri_to_meg_trans)
            
            self.progress["fwd_model"] = True
            dm.save(self.lh_white_valid_vert, self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_valid_vert")
            dm.save(self.rh_white_valid_vert, self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_valid_vert")
            dm.save(self.optimized_fwd_sol, self.anatomy_path + self.subj_name + "/tmp/" + "optimized_fwd_sol")
        else:
            self.lh_white_valid_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "lh_white_valid_vert")
            self.rh_white_valid_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "rh_white_valid_vert")
            self.optimized_fwd_sol = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "optimized_fwd_sol")
        
    def sensor_cov(self, overwrite_sensor_cov):
        (self.sensor_cov_eigen_val,
         self.sensor_cov_eigen_vec,
         self.sensor_cov_names) = finnpy.source_reconstruction.sensor_covariance.get_sensor_covariance(self.sensor_cov_path,
                                                                                                   cov_path = self.cov_path,
                                                                                                   overwrite = overwrite_sensor_cov)

    def inv_model(self, overwrite_mri_trans):
        if (self.progress["inv_model"] == False):
            (self.inv_trans, self.noise_norm) = finnpy.source_reconstruction.inverse_model.calc_inverse_model(self.sensor_cov_eigen_val,
                                                                                                              self.sensor_cov_eigen_vec,
                                                                                                              self.sensor_cov_names,
                                                                                                              self.optimized_fwd_sol,
                                                                                                              self.rec_meta_info)
            (self.fs_avg_trans_mat,
             self.src_fs_avg_valid_lh_vert,
             self.src_fs_avg_valid_rh_vert) = finnpy.source_reconstruction.utils.get_mri_subj_to_fs_avg_trans_mat(self.lh_white_valid_vert,
                                                                                                                  self.rh_white_valid_vert,
                                                                                                                  self.octa_model_vert,
                                                                                                                  self.anatomy_path,
                                                                                                                  self.subj_name,
                                                                                                                  self.fs_path, overwrite = overwrite_mri_trans)    
             
            self.progress["inv_model"] = True
            dm.save(self.inv_trans, self.anatomy_path + self.subj_name + "/tmp/" + "inv_trans")
            dm.save(self.noise_norm, self.anatomy_path + self.subj_name + "/tmp/" + "noise_norm")
            dm.save(self.fs_avg_trans_mat, self.anatomy_path + self.subj_name + "/tmp/" + "fs_avg_trans_mat")
            dm.save(self.src_fs_avg_valid_lh_vert, self.anatomy_path + self.subj_name + "/tmp/" + "src_fs_avg_valid_lh_vert")
            dm.save(self.src_fs_avg_valid_rh_vert, self.anatomy_path + self.subj_name + "/tmp/" + "src_fs_avg_valid_rh_vert")
        else:
            self.inv_trans = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "inv_trans")
            self.noise_norm = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "noise_norm")
            self.fs_avg_trans_mat = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "fs_avg_trans_mat")
            self.src_fs_avg_valid_lh_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "src_fs_avg_valid_lh_vert")
            self.src_fs_avg_valid_rh_vert = dm.load(self.anatomy_path + self.subj_name + "/tmp/" + "src_fs_avg_valid_rh_vert")
            
    def calc_model(self, mode,
                   
                   overwrite_fs_extract = False,
                   overwrite_ws_extract = False,
                   overwrite_sensor_cov = False,
                   overwrite_mri_trans = False):
        
        finnpy.source_reconstruction.utils.init_fs_paths(self.anatomy_path, self.fs_path)
    
        if (mode in ["full", "fs_extract"]):
            self.fs_extract(overwrite_fs_extract)
        self.sensor_cov(overwrite_sensor_cov)
                
        if (mode in ["full", "recording"]):
            self.run_coreg(self.visualization["coregistration"])
            self.ws_extract(overwrite_ws_extract, self.visualization["watershed"])
            self.calc_bem_linear_basis()
            self.refine_cortical_model()
            self.fwd_model()
            self.inv_model(overwrite_mri_trans)
    
    def apply_model(self, sen_data):
        if (self.progress["inv_model"] == False):
            raise AssertionError("Model needs to be computed first")
        
        src_data = finnpy.source_reconstruction.inverse_model.apply_inverse_model(sen_data, self.inv_trans, self.noise_norm)
        fs_avg_src_data = finnpy.source_reconstruction.utils.apply_mri_subj_to_fs_avg_trans_mat(self.fs_avg_trans_mat, src_data)
        (morphed_epoch_data, morphed_epoch_channels, morphed_region_names) = finnpy.source_reconstruction.source_region_model.avg_source_regions(fs_avg_src_data,
                                                                                                                                                 self.src_fs_avg_valid_lh_vert,
                                                                                                                                                 self.src_fs_avg_valid_rh_vert,
                                                                                                                                                 self.octa_model_vert,
                                                                                                                                                 self.octa_model_faces,
                                                                                                                                                 self.fs_path)
        morphed_epoch_data = np.asarray(morphed_epoch_data)
        
        return (morphed_epoch_data, morphed_epoch_channels, morphed_region_names)

def calc_model(fs_path, anatomy_path, subj_name, sensor_cov_path, cov_path, data_path,
               
               visualize_coregistration = False, visualize_skull_skin_plots = False,
               
               overwrite_fs_extract = False, overwrite_ws_extract = False,
               overwrite_sensor_cov = False, overwrite_mri_trans = False):
    
    finnpy.source_reconstruction.utils.init_fs_paths(anatomy_path, fs_path)
    
    finnpy.source_reconstruction.mri_anatomy.copy_fs_avg_anatomy(fs_path, anatomy_path, subj_name, overwrite = overwrite_fs_extract)
    (sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names) = finnpy.source_reconstruction.sensor_covariance.get_sensor_covariance(file_path = sensor_cov_path, cov_path = cov_path, overwrite = overwrite_sensor_cov)

    rec_meta_info = mne.io.read_info(data_path)
    (coreg_rotors, meg_pts) = finnpy.source_reconstruction.coregistration_meg_mri.calc_coreg(subj_name, anatomy_path, rec_meta_info, registration_scale_type = "free")
    finnpy.source_reconstruction.mri_anatomy.scale_anatomy(anatomy_path, subj_name, coreg_rotors[6:9])
    (coreg_rotors, meg_pts) = finnpy.source_reconstruction.coregistration_meg_mri.calc_coreg(subj_name, anatomy_path, rec_meta_info, registration_scale_type = "restricted")
    
    if (visualize_coregistration):
        rigid_mri_to_meg_trans = finnpy.source_reconstruction.coregistration_meg_mri.get_rigid_transform(coreg_rotors)
        finnpy.source_reconstruction.coregistration_meg_mri.plot_coregistration(rigid_mri_to_meg_trans, rec_meta_info, meg_pts, anatomy_path, subj_name)

    finnpy.source_reconstruction.bem_model.calc_skull_and_skin_models(anatomy_path, subj_name, overwrite = overwrite_ws_extract)
    (ws_in_skull_vert, ws_in_skull_faces, 
     ws_out_skull_vert, ws_out_skull_faces,
     ws_out_skin_vect, ws_out_skin_faces) = finnpy.source_reconstruction.bem_model.read_skull_and_skin_models(anatomy_path, subj_name)
    
    if (visualize_skull_skin_plots):
        finnpy.source_reconstruction.bem_model.plot_skull_and_skin_models(ws_in_skull_vert, ws_in_skull_faces,
                                                                          ws_out_skull_vert, ws_out_skull_faces,
                                                                          ws_out_skin_vect, ws_out_skin_faces,
                                                                          anatomy_path, subj_name)
    del ws_out_skull_vert; del ws_out_skull_faces; del ws_out_skin_vect; del ws_out_skin_faces
    
    (in_skull_reduced_vert, in_skull_faces, 
    in_skull_faces_area, in_skull_faces_normal, 
    bem_solution) = finnpy.source_reconstruction.bem_model.calc_bem_model_linear_basis(ws_in_skull_vert, ws_in_skull_faces)
    
    (lh_white_vert, lh_white_faces,
    rh_white_vert, rh_white_faces,
    lh_sphere_vert,
    rh_sphere_vert) = finnpy.source_reconstruction.utils.read_cortical_models(anatomy_path, subj_name)
    (octa_model_vert, octa_model_faces) = finnpy.source_reconstruction.source_mesh_model.create_source_mesh_model()
    (lh_white_valid_vert, rh_white_valid_vert) = finnpy.source_reconstruction.source_mesh_model.match_source_mesh_model(lh_sphere_vert, rh_sphere_vert, octa_model_vert)
        
    rigid_mri_to_meg_trans = finnpy.source_reconstruction.coregistration_meg_mri.get_rigid_transform(coreg_rotors)
    rigid_meg_to_mri_trans = scipy.linalg.inv(rigid_mri_to_meg_trans)
    (fwd_sol,
     lh_white_valid_vert, rh_white_valid_vert) = finnpy.source_reconstruction.forward_model.calc_forward_model(lh_white_vert, rh_white_vert,
                                                                                                               rigid_meg_to_mri_trans, rigid_mri_to_meg_trans, rec_meta_info, in_skull_reduced_vert, in_skull_faces, in_skull_faces_normal, in_skull_faces_area, bem_solution, lh_white_valid_vert, rh_white_valid_vert)
    optimized_fwd_sol = finnpy.source_reconstruction.forward_model.optimize_fwd_model(lh_white_vert, lh_white_faces, lh_white_valid_vert, rh_white_vert, rh_white_faces, rh_white_valid_vert, fwd_sol, rigid_mri_to_meg_trans)
    (inv_trans, noise_norm) = finnpy.source_reconstruction.inverse_model.calc_inverse_model(sensor_cov_eigen_val, sensor_cov_eigen_vec, sensor_cov_names, optimized_fwd_sol, rec_meta_info)
    
    
    
    (fs_avg_trans_mat, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert) = finnpy.source_reconstruction.utils.get_mri_subj_to_fs_avg_trans_mat(lh_white_valid_vert, rh_white_valid_vert, octa_model_vert, anatomy_path, subj_name, fs_path, overwrite = overwrite_mri_trans)
    
    return (inv_trans, noise_norm,
            fs_avg_trans_mat, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert, 
            octa_model_vert, octa_model_faces)

def application(sen_data,
                
                inv_trans, noise_norm, 
                
                fs_avg_trans_mat, src_fs_avg_valid_lh_vert, src_fs_avg_valid_rh_vert, 
                octa_model_vert, octa_model_faces,
                
                fs_path):
    
    src_data = finnpy.source_reconstruction.inverse_model.apply_inverse_model(sen_data, inv_trans, noise_norm)
    fs_avg_src_data = finnpy.source_reconstruction.utils.apply_mri_subj_to_fs_avg_trans_mat(fs_avg_trans_mat, src_data)
    (morphed_epoch_data, morphed_epoch_channels, morphed_region_names) = finnpy.source_reconstruction.source_region_model.avg_source_regions(fs_avg_src_data,
                                                                                                                                             src_fs_avg_valid_lh_vert,
                                                                                                                                             src_fs_avg_valid_rh_vert,
                                                                                                                                             octa_model_vert,
                                                                                                                                             octa_model_faces,
                                                                                                                                             fs_path)
    morphed_epoch_data = np.asarray(morphed_epoch_data)
    
    return (morphed_epoch_data, morphed_epoch_channels, morphed_region_names)

def verify_integrity(morphed_epoch_data, source_data_path):
    
    ref_morphed_epoch_data = dm.load(source_data_path + "/morphed_epoch_data")[:, 0, :] #Only the first epoch
    print((np.abs(ref_morphed_epoch_data - morphed_epoch_data) < 1e-10).all())

FS_PATH = ""
ANATOMY_PATH = ""
SUBJ_NAME = ""

DATA_PATH = ""
SENSOR_COV_PATH = ""

SENSOR_DATA_PATH = ""
SOURCE_DATA_PATH = ""

FS_PATH = "/usr/local/freesurfer/7.2.0/"
ANATOMY_PATH = "/mnt/data/Professional/projects/finnpy/latest/finnpy/data/source_reconstruction/anatomical/"
SUBJ_NAME = "demo_patient_without_T1"

DATA_PATH = "/mnt/data/Professional/projects/finnpy/latest/finnpy/data/source_reconstruction/meg/demo_patient_without_T1/recording/raw/meta_data_only.fif"
SENSOR_COV_PATH = "/mnt/data/Professional/projects/finnpy/latest/finnpy/data/source_reconstruction/empty_room/empty_room.fif"

SENSOR_DATA_PATH = "/mnt/data/Professional/projects/finnpy/latest/finnpy/data/source_reconstruction/meg/demo_patient_without_T1/recording/preprocessed_sen/data"
SOURCE_DATA_PATH = "/mnt/data/Professional/projects/finnpy/latest/finnpy/data/source_reconstruction/meg/demo_patient_without_T1/recording/preprocessed_src/data/"

def main():
    
    #===========================================================================
    # ar = Auto_reconstruct(FS_PATH, ANATOMY_PATH, SUBJ_NAME, T1_SCAN_PATH + SUBJ_NAME + "/" + T1_SCAN_FILE, DATA_PATH, SENSOR_COV_PATH, ANATOMY_PATH + "sensor_cov/")
    # ar.calc_model(mode = "full")
    # ar.apply_model(dm.load(SENSOR_DATA_PATH)[:, 0, :]) #Only the first epoch
    #===========================================================================
    
    model = calc_model(FS_PATH, ANATOMY_PATH, SUBJ_NAME, SENSOR_COV_PATH, ANATOMY_PATH + "sensor_cov/", DATA_PATH)
    sensor_data = dm.load(SENSOR_DATA_PATH)[:, 0, :] #Only the first epoch
    morphed_epoch_data = application(sensor_data, *model, FS_PATH)
    
    verify_integrity(morphed_epoch_data, SOURCE_DATA_PATH)

main()




