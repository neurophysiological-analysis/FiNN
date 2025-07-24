'''
Created on Jul 22, 2025

@author: voodoocode
'''

import os
import h5py
import finnpy.misc.external_calls as ex_c
import finnpy.visualization._conv_nifti_obj

def conv_ewert(mat_path, nii_path, obj_path, slicer_path,
               seg_min_thresh = None, seg_max_thresh = None):
    """
    Extracts subcortical components from the Ewert 2017 atlas.
    
    Parameters
    ----------
    mat_path : string
               Path to the atlas_index.mat file.
    nii_path : string
               Path to the to-be-converted nifti files.
    obj_path : String
               Path to place the output files.
    slicer_path : String
                  Path to the 3dslicer binary/executable.
    seg_min_thresh : float
                     Minimum value for automated segmentation
    seg_max_thresh : float
                     Maximum value for automated segmentation
    """
    
    if (seg_min_thresh is None):
        seg_min_thresh = 0.35
    if (seg_max_thresh is None):
        seg_max_thresh = 1.00
    
    file = h5py.File(mat_path, "r")
    for region_idx in range(file["atlases"]["names"].shape[0]):
        f_name = ''.join([chr(char[0]) for char in file[file["atlases"]["names"][region_idx][0]]])
        region = ''.join([chr(char[0]) for char in file[file[file["atlases"]["labels"][0, 0]][0, region_idx]]])
        
        ex_c.run("01", [os.path.dirname(__file__) + "/slicer.sh", slicer_path,
                        finnpy.visualization._conv_nifti_obj.__file__,  # pylint: disable=protected-access
                        (nii_path + f_name), obj_path, region, str(seg_min_thresh), str(seg_max_thresh)])
        
        continue
    
    file.close()
