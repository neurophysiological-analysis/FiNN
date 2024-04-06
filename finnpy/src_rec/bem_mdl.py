'''
Created on Oct 12, 2022

@author: voodoocode
'''

import numpy as np

import finnpy.src_rec.utils
import finnpy.src_rec.sphere_mdl

class BEM_mdl():
    """
    Container class, populed with the following items:
              
    vert : numpy.ndarray, shape(scaled_vtx_cnt, 3)
           Remaining Vertices of a skin/skull model vertices.
    faces : numpy.ndarray, shape(scaled_face_cnt, 3)
            Remaining Vertices of a skin/skull model faces.
    faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                   Normals of the individual remaining inner skull faces.
    faces_area : numpy.ndarray, shape(scaled_face_cnt)
                 Surface area of the remaining faces.
    bem_solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                   BEM solution (preliminary step for the calculation of the forward model).
    """

    
    def __init__(self, vert, faces, faces_normal, faces_area, solution):
        self.vert = vert
        self.faces = faces
        self.faces_normal = faces_normal
        self.faces_area = faces_area
        self.solution = solution

def run(fs_path, vert, faces, tgt_icosahedron_level = 4):
    """
    Calcuates the BEM linear basis coefficients using the linear collocation method.
    
    Parameters
    ----------
    fs_path : string
              Path to the freesurfer directory.
    vert : numpy.ndarray, shape(vtx_cnt, 3)
           Vertices of a skin/skull model.
    faces : numpy.ndarray, shape(face_cnt, 3)
            Faces of a skin/skull model.
    tgt_icosahedron_level : int
                            Order of the icosahedron employed herein,
                            defaults to 4.
               
    Returns
    -------
    bem_mdl : finnpy.src_rec.bem_mdl.BEM_mdl
              Container class, populed with the following items:
              
              vert : numpy.ndarray, shape(scaled_vtx_cnt, 3)
                     Remaining Vertices of a skin/skull model vertices.
              faces : numpy.ndarray, shape(scaled_face_cnt, 3)
                      Remaining Vertices of a skin/skull model faces.
              faces_normal : numpy.ndarray, shape(scaled_face_cnt, 3)
                             Normals of the individual remaining inner skull faces.
              faces_area : numpy.ndarray, shape(scaled_face_cnt)
                           Surface area of the remaining faces.
              bem_solution : numpy.ndarray, shape(scaled_vtx_cnt, scaled_vtx_cnt)
                             BEM solution (preliminary step for the calculation of the forward model).
    """
    src_icosahedron_level = int(np.log(faces.shape[0] / 20) / np.log(2) / 2)
    #Creates a model 
    #Icosahedrons are provided by freesurfer and reading those is faster than computing.
    (src_vert, _) = finnpy.src_rec.sphere_mdl.read_sphere_from_icosahedron_in_fs_order(fs_path, src_icosahedron_level)
    (tgt_vert, tgt_faces) = finnpy.src_rec.sphere_mdl.calculate_sphere_from_icosahedron(tgt_icosahedron_level)
    
    #Reduce anatomy vertices/faces to relevant ones
    reduced_vert = np.copy(vert)[finnpy.src_rec.utils.find_nearest_neighbor(src_vert, tgt_vert)[0]]
    faces = tgt_faces
    
    #Calculate matrix omega, containing "the solid angles subtended at the center of each triangle by all other triangles", see "Error Analysis of a New Galerkin Method to
    #Solve the Forward Problem in MEG and EEG Using the Boundary Element Method" by Satu Tissari, Jussi Rahola for more details equation #17.
    #In short, to calculate potentials using a BEM model, its coefficients/weights have to be calculated beforehand. 
    reduced_vert /= 1000 # Scale from m to mm
    x_pos = reduced_vert[faces[:, 0], :]
    y_pos = reduced_vert[faces[:, 1], :]
    z_pos = reduced_vert[faces[:, 2], :]
    faces_normal = finnpy.src_rec.utils.fast_cross_product((y_pos - x_pos), (z_pos - x_pos))
    double_faces_area = finnpy.src_rec.utils.magn_of_vec(faces_normal)
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
    
    return BEM_mdl(reduced_vert, faces, faces_normal, double_faces_area/2, bem_solution)

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

    
