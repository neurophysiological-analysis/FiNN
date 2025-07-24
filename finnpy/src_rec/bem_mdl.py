"""
Created on Oct 12, 2022.

@author: voodoocode
"""

import numpy as np
import warnings
import nibabel.freesurfer
import matplotlib.pyplot as plt

import finnpy.src_rec.utils as utils  # @UnresolvedImport
import finnpy.src_rec.sphere_mdl  # @UnresolvedImport
import finnpy.src_rec.extract_anatomy  # @ @UnresolvedImport

class BEM_mdl():
    """
    Container class.
    
    Create a container with BEM model elements for convenience.
        
    Parameters
    ----------
    vert : list of numpy.ndarray, shape(scaled_vtx_cnt, 3)
           Remaining vertices of a skin/skull model vertices.
    faces : list of numpy.ndarray, shape(scaled_face_cnt, 3)
            Remaining faces of a skin/skull model faces.
    faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                   Normals of the individual remaining inner skull faces.
    faces_area : numpy.ndarray, shape(scaled_face_cnt)
                 Surface area of the remaining faces.
    solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
               BEM solution (preliminary step for the calculation of the forward model).
    
    Attributes
    ----------
    INNER_SKULL_IDX : int
                      Index of the inner skull.
    OUTER_SKULL_IDX : int
                      Index of the outer skull.
    OUTER_SKIN_IDX : int
                     Index of the skin.
    """
    
    INNER_SKULL_IDX: int = 0
    OUTER_SKULL_IDX: int = 1
    OUTER_SKIN_IDX: int  = 2  # noqa: E221
    
    def __init__(self, vert, faces, faces_normal, faces_area, solution):
        self.vert = vert
        self.faces = faces
        self.faces_normal = faces_normal
        self.faces_area = faces_area
        self.solution = solution

def run(fs_path, anatomy_path, subj_name, signal_type, coreg,
        conductivity = None, tgt_icosahedron_level = 4):
    """
    Calcuates the BEM linear basis coefficients using the linear collocation method.
    
    Parameters
    ----------
    fs_path : string
              Path to the freesurfer directory.
    anatomy_path : string
                   Path to the anatomy folder. This folder should contain a
                   sub-folder for each subject, to be pupulated with the corresponding structural data.
    subj_name : string
                Subject name.
    signal_type : string
                  Mode is either "EEG" or "MEG". 
    coreg : finnpy.src_rec.coreg.Coreg
            Container with different transformation matrices
    conductivity : (float,) or (float, float, float)
                   Conductivity values for a one layer (MEG) or three layer (EEG) model.
    tgt_icosahedron_level : int
                            Order of the icosahedron employed herein,
                            defaults to 4.
               
    Returns
    -------
    bem_mdl : finnpy.src_rec.bem_mdl.BEM_mdl
              Container class, populed with the following items:
              
              vert : list() of numpy.ndarray, [shape(scaled_vtx_cnt, 3), ...]
                     Remaining vertices of a skin/skull model vertices.
              faces : list() of numpy.ndarray, [shape(scaled_face_cnt, 3), ...]
                      Remaining faces of a skin/skull model faces.
              faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                             Normals of the individual remaining inner skull faces.
              faces_area : numpy.ndarray, shape(scaled_face_cnt)
                           Surface area of the remaining faces.
              bem_solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                             BEM solution (preliminary step for the calculation of the forward model).
    
    Raises
    ------
    AssertionError
        Raised if the number of vertices in the three layer model vary.
    
    """
    if (conductivity is None):
        if (signal_type == "MEG"):
            conductivity = (.3,)
        if (signal_type == "EEG"):
            conductivity = (.3, .006, .3)
    
    # Read anatomical data
    (vert, faces) = finnpy.src_rec.extract_anatomy.read_skin_skull(anatomy_path, subj_name, signal_type, coreg)
    
    if (signal_type == "EEG"):
        if (len(vert[0]) != len(vert[1]) or len(vert[0]) != len(vert[2])):
            raise AssertionError("Error, skin/skull models are of different size (before downscaling).")
    
    # Creates a model 
    # Icosahedrons are provided by freesurfer and reading those is faster than computing.
    src_icosahedron_level = int(np.log(faces[0].shape[0] / 20) / np.log(2) / 2)
    (src_vert, _) = finnpy.src_rec.sphere_mdl.read_sphere_from_icosahedron_in_fs_order(fs_path, src_icosahedron_level)
    (tgt_vert, tgt_faces) = finnpy.src_rec.sphere_mdl.calculate_sphere_from_icosahedron(tgt_icosahedron_level)
    
    # Scales the anatomical models down for further processing
    # Reduce anatomy vertices/faces to relevant ones
    for surf_idx in range(len(vert)):
        vert[surf_idx] = np.copy(vert[surf_idx])[utils.find_nearest_neighbor(src_vert, tgt_vert)[0]]
        faces[surf_idx] = tgt_faces  # Can just overwrite faces as vert are closest to tgt_vert and so tgt_faces match...)
    
    if (signal_type == "EEG"):
        if (len(vert[0]) != len(vert[1]) or len(vert[0]) != len(vert[2])):
            raise AssertionError("Error, skin/skull models are of different size after downscaling.")
    
    # Compute the BEM model
    bem_mdl = _compute_bem_solution(vert, faces, conductivity)
    
    return bem_mdl

def _compute_bem_solution(vert, faces, conductivity):
    """
    Control flow to calculate the bem solution.
    
    Calculate matrix omega, containing "the solid angles subtended at the center of each triangle by all other triangles", see "Error Analysis of a New Galerkin Method to
    Solve the Forward Problem in MEG and EEG Using the Boundary Element Method" by Satu Tissari, Jussi Rahola for more details equation #17.
    In short, to calculate potentials using a BEM model, its coefficients/weights have to be calculated beforehand. 
    
    Important: This is only using mri data.
    
    Parameters
    ----------
    vert : np.ndarray(vert_cnt, 3) or [np.ndarray(out_skin_vert_cnt, 3), np.ndarray(out_skull_vert_cnt, 3), np.ndarray(in_skull_vert_cnt, 3)] 
           Single-layer model: Vertices of the inner skull model. Three-layered model: Vertices of the outer skin, outer skull and inner skull models.
    faces : np.ndarray(face_cnt, 3) or [np.ndarray(out_skin_vert_cnt, 3), np.ndarray(out_skull_vert_cnt, 3), np.ndarray(in_skull_vert_cnt, 3)]
            Single-layer model: Faces of the inner skull model. Three-layered model: Faces of the outer skin, outer skull and inner skull models.
    conductivity : np.ndarray, shape(3, ) or shape(1, ) or None
                   Conductivity values for the individual surfaces. If None, defaults to (.3) for MEG and (.3, .006, .3) for EEG.
    
    Returns
    -------
    BEM_mdl: BEM_mdl
             Container class:
             - vert : list of numpy.ndarray, shape(scaled_vtx_cnt, 3)
                      Remaining vertices of a skin/skull model vertices.
             - faces : list of numpy.ndarray, shape(scaled_face_cnt, 3)
                       Remaining faces of a skin/skull model faces.
             - faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                              Normals of the individual remaining inner skull faces.
             - faces_area : numpy.ndarray, shape(scaled_face_cnt)
                            Surface area of the remaining faces.
             - solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                          BEM solution (preliminary step for the calculation of the forward model).
    
    """
    faces_normals = [None] * len(vert)
    double_faces_areas = [None] * len(vert)
    for surf_idx in range(len(vert)):
        vert[surf_idx] /= 1000  # Scale from m to mm
    
        x_pos = vert[surf_idx][faces[surf_idx][:, 0], :]
        y_pos = vert[surf_idx][faces[surf_idx][:, 1], :]
        z_pos = vert[surf_idx][faces[surf_idx][:, 2], :]
        faces_normal = utils.fast_3D_cross_product_multi((y_pos - x_pos), (z_pos - x_pos))
        double_faces_area = utils.fast_3D_norm_vec_multi(faces_normal)
        n_faces_normal = utils.fast_3D_norm_vec_multi(faces_normal)
        faces_normal[n_faces_normal > 0] = faces_normal[n_faces_normal > 0] / np.expand_dims(n_faces_normal[n_faces_normal > 0], axis = 1)
        
        double_faces_areas[surf_idx] = double_faces_area
        faces_normals[surf_idx] = faces_normal 

    vert_cnt = [vort.shape[0] for vort in vert]; vert_total = np.sum(vert_cnt); vert_offset = np.cumsum(np.concatenate(([0], vert_cnt)))
    omega = np.zeros((vert_total, vert_total))
    for out_surf_idx in range(len(vert)):
        for in_surf_idx in range(len(vert)):
            omega_subset = omega[vert_offset[out_surf_idx]:vert_offset[out_surf_idx + 1], 
                                 vert_offset[in_surf_idx]:vert_offset[in_surf_idx + 1]]
            
            _calc_omega_entries(omega_subset, vert[out_surf_idx], vert[in_surf_idx], faces[in_surf_idx],
                                double_faces_areas[in_surf_idx], faces_normals[in_surf_idx])
            
            if (out_surf_idx == in_surf_idx):
                _correct_diag_omega_entries(omega_subset, faces[in_surf_idx])
                
    # The matrix "bem_solution" contains all linear basis functions
    # A "deflation" factor is added to replace the "zero" eigenvalue within to ensure invertability
    # See "EEG and MEG: Forward Solutions for Inverse Methods" by Mosher, 1999 for reference: "In E/MEG, the Neumann boundary condition used [...]"
    deflation_factor = 1 / omega.shape[0]
    if (len(vert) == 1):
        # Left part of equation #6 by Satu Tissari, Jussi Rahola (see above). 
        bem_solution = np.linalg.inv(np.eye(omega.shape[0]) + deflation_factor - omega / (2 * np.pi))
    
    elif (len(vert) == 3):
        bem_solution = np.copy(omega)
        sigma = np.asarray(np.concatenate(([0,], conductivity))); sigma = (sigma[1:] - sigma[:-1])[None, :] / (sigma[1:] + sigma[:-1])[:, None]
        for out_surf_idx in range(len(vert)):
            for in_surf_idx in range(len(vert)):
                start1 = vert_offset[out_surf_idx]; stop1 = vert_offset[out_surf_idx + 1]
                start2 = vert_offset[in_surf_idx]; stop2 = vert_offset[in_surf_idx + 1]
                
                bem_solution[start1:stop1, start2:stop2] = deflation_factor - bem_solution[start1:stop1, start2:stop2] * sigma[out_surf_idx, in_surf_idx] / (2 * np.pi)
        bem_solution = np.linalg.inv(np.eye(bem_solution.shape[0]) + bem_solution)
        
        scaling_factor = conductivity[1] / conductivity[2]
        if (scaling_factor < .1):
            weighing_factor = (1. + scaling_factor) / scaling_factor 
            
            # No need to copy omega here, isn't needed anymore afterward
            omega_subset = omega[vert_offset[2]:vert_offset[3], vert_offset[2]:vert_offset[3]]
            deflation_factor = 1 / omega_subset.shape[0]
            bem_solution_mod = np.linalg.inv(np.eye(omega_subset.shape[0]) + deflation_factor - omega_subset / (2 * np.pi))
            
            for surf_idx in range(len(vert)):
                start1 = vert_offset[surf_idx]; stop1 = vert_offset[surf_idx + 1]; start2 = vert_offset[2]
                bem_solution[start1:stop1, start2:] -= 2 * np.dot(bem_solution[start1:stop1, start2:], bem_solution_mod)
                
            bem_solution[vert_offset[2]:vert_offset[3], vert_offset[2]:vert_offset[3]] += weighing_factor * bem_solution_mod
            
            bem_solution *= scaling_factor
    return BEM_mdl(vert, faces, faces_normals, [double_faces_area / 2 for double_faces_area in double_faces_areas], bem_solution)

def _calc_omega_entries(omega, out_vert, in_vert, faces, double_faces_area, faces_normal):
    """
    Calculate the elements of omega (linear basis factors), see function calc_bem_model.
    
    Parameters
    ----------
    omega : numpy.ndarray, shape(reduced_vtx_cnt, reduced_vtx_cnt)
            Linear basis factor elements of the BEM solution, precurser to the BEM solution proper.
    out_vert : numpy.ndarray, shape(reduced_vtx_cnt, 3)
           Vertices of a layer from the skin skull model.
    in_vert : numpy.ndarray, shape(reduced_vtx_cnt, 3)
           Vertices of a layer from the skin skull model.
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
    for (face_idx, face) in enumerate(faces):
        tgt_in_vert = in_vert[face, :]
        
        r0 = tgt_in_vert[0, :] - out_vert
        r1 = tgt_in_vert[1, :] - out_vert
        r2 = tgt_in_vert[2, :] - out_vert
        
        xr0r1 = utils.fast_3D_cross_product_multi(r0, r1)
        xr2r1 = utils.fast_3D_cross_product_multi(r2, r1)
        xr0r2 = utils.fast_3D_cross_product_multi(r0, r2)
        xr1r0 = -xr0r1
        
        nr0 = utils.fast_3D_norm_vec_multi(r0)
        nr1 = utils.fast_3D_norm_vec_multi(r1)
        nr2 = utils.fast_3D_norm_vec_multi(r2)
        
        sa0 = nr0 * nr1 * nr2
        sa1 = utils.fast_3D_sum_multi(r0 * r1) * nr2
        sa2 = utils.fast_3D_sum_multi(r0 * r2) * nr1
        sa3 = utils.fast_3D_sum_multi(r1 * r2) * nr0
        
        solid_angles = 2 * np.arctan2(utils.fast_3D_sum_multi(xr0r1 * r2), sa0 + sa1 + sa2 + sa3)
        
        r10 = r1[0] - r0[0]; nr10 = utils.fast_3D_norm_vec_single(r10)
        r21 = r2[0] - r1[0]; nr21 = utils.fast_3D_norm_vec_single(r21)
        r02 = r0[0] - r2[0]; nr02 = utils.fast_3D_norm_vec_single(r02)

        # Correct for the auto-solid angle problem for use in the gamma functions
        bads = np.abs(solid_angles) < np.pi * 1e-12  # 1e-100#1e-12
        nr0[bads] = 1; nr1[bads] = 1; nr2[bads] = 1
                
        gamma0 = np.log((nr0 * nr10 + np.dot(r0, r10)) / (nr1 * nr10 + np.dot(r1, r10))) / nr10
        gamma1 = np.log((nr1 * nr21 + np.dot(r1, r21)) / (nr2 * nr21 + np.dot(r2, r21))) / nr21
        gamma2 = np.log((nr2 * nr02 + np.dot(r2, r02)) / (nr0 * nr02 + np.dot(r0, r02))) / nr02
        gamma = np.expand_dims(gamma2 - gamma0, axis = 1) * r0 + np.expand_dims(gamma0 - gamma1, axis = 1) * r1 + np.expand_dims(gamma1 - gamma2, axis = 1) * r2
        
        # For reference
        # See equation #17 of "Error Analysis of a New Galerkin Method to Solve the Forward Problem in MEG and EEG Using the Boundary Element Method"
        # by Satu Tissari, Jussi Rahola, 1998
        omega0 = (double_faces_area[face_idx] * solid_angles * utils.fast_3D_sum_multi(xr2r1 * faces_normal[face_idx])
                  + utils.fast_3D_sum_multi(xr0r1 * r2) * utils.fast_3D_sum_multi((r2 - r1) * gamma)) / (double_faces_area[face_idx] * double_faces_area[face_idx])  # noqa: W503
        omega1 = (double_faces_area[face_idx] * solid_angles * utils.fast_3D_sum_multi(xr0r2 * faces_normal[face_idx])
                  + utils.fast_3D_sum_multi(xr0r1 * r2) * utils.fast_3D_sum_multi((r0 - r2) * gamma)) / (double_faces_area[face_idx] * double_faces_area[face_idx])  # noqa: W503
        omega2 = (double_faces_area[face_idx] * solid_angles * utils.fast_3D_sum_multi(xr1r0 * faces_normal[face_idx])
                  + utils.fast_3D_sum_multi(xr0r1 * r2) * utils.fast_3D_sum_multi((r1 - r0) * gamma)) / (double_faces_area[face_idx] * double_faces_area[face_idx])  # noqa: W503
        
        loc_omega = np.asarray([omega0, omega1, omega2]).T
        loc_omega[bads, :] = 0
        omega[:, face] -= loc_omega
    
    return omega

def _correct_diag_omega_entries(omega, faces):
    """
    Corrects the diagonal elements of omega as these cannot be calculated as the non-diagonal ones due to the auto solid angle problem.
    
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
    half_missing_omega = (((2.0 * np.pi) - np.sum(omega, axis = 1)) / 2)
    
    # Half the angle goes to r0, the other half is distributed amongst the respective faces
    omega[np.diag_indices_from(omega)] += half_missing_omega
    
    for ref_vertex_idx in range(omega.shape[0]):
        neigh_vertices = np.unique(faces[np.argwhere(faces == ref_vertex_idx)[:, 0], :].reshape(-1))
        neigh_vertices = neigh_vertices[neigh_vertices != ref_vertex_idx]
        
        omega[ref_vertex_idx, neigh_vertices] += half_missing_omega[ref_vertex_idx] / len(neigh_vertices)

    return omega
