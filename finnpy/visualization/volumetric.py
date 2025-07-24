"""
Created on Feb 2, 2023.

@author: voodoocode
"""

import os
import shutil
import numpy as np
import scipy.ndimage
import pyvista
import ctypes

def _convert_vti_vdb(path, f_name, vti_vdb_conv_path, tmp_dir = None):
    """
    Convert a vti object (not blender readable) into a vdb object (blender readable).
    
    Parameters
    ----------
    path : str
           Path to the *.vti file.
    f_name : str
             Name of the *.vti file.
    vti_vdb_conv_path : str
                        Path to the *.vti to *.vdb conversion library.
                        Needs to be locally compiled. Code is available at https://github.com/neurophysiological-analysis/FiNN_extensions.
    tmp_dir : str
              Optional. Prefered name of the temporary folder used for the external funcition call.
              This folder will be cleaned up automatically after successful execution.
    """
    if (tmp_dir is None):
        tmp_dir = str(np.random.randint(0, 1e5))
        while (os.path.exists(tmp_dir)):
            tmp_dir = str(np.random.randint(0, 1e5))
    
    ex_conv = ctypes.CDLL(vti_vdb_conv_path)
    args = [b'-path', path.encode(), b'-fname', f_name.encode()]
    argc = len(args)
    argv = (ctypes.c_char_p * argc)(*args)
    ex_conv.convert_file(argc, argv)

def export_to_blender(structs, volumes, vti_vtb_converter = None, outpath = None):
    """
    Export structures and volumes to blender.
    
    Parameters
    ----------
    structs : list, pyvista.core.pointset.PolyData
              List of structures to be exported.
    volumes : list, pyvista.core.pointset.ImageData
              List of volumes to be exported.
    vti_vtb_converter : String
                        Path to the vti to vtb converter. Code is available at
                        https://github.com/neurophysiological-analysis/FiNN_extensions and needs to be compiled.
    outpath : String
              Path to write the exported files to. Default is local directory '.'.
    """
    if (outpath is None):
        outpath = "."
    
    if (len(outpath) > 0 and outpath != "."):
        if (outpath[-1] != "/"):
            outpath += "/"
    
    for struct in structs:
        shutil.copy(struct["path"], outpath + struct["name"] + ".obj")
    
    for vol in volumes:
        vol.save(outpath + vol["name"] + ".vti")
        _convert_vti_vdb(os.path.abspath(outpath) + "/", vol["name"] + ".vti", vti_vtb_converter)
        meta_info = open(outpath + vol["name"] + ".meta", "w")  # pylint: disable=unspecified-encoding
        meta_info.write("Center: %f %f %f\n" % (vol.center[0], vol.center[1], vol.center[2]))
        meta_info.write("Dimensions: %f %f %f\n" % (vol.dimensions[0], vol.dimensions[1], vol.dimensions[2]))
        meta_info.write("Spacings: %f %f %f\n" % (vol.spacing[0], vol.spacing[1], vol.spacing[2]))
        meta_info.close()
        
def add_axes(fig):
    """
    Add axes to plot.
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    """
    fig.add_axes()

def add_pts(fig, pts, color = None, size = None, opacity = None):
    """
    Add points to the figure
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    pts : np.ndarray, shape(3, n)
          To be added points
    color : string
            Color of the points. Defaults to grey.
    size : int
           Size of the points. Defaults to 5.
    opacity : float 
              Opacitiy of the added structure. Defaults to 0.5.
    Returns
    -------
    mesh : pyvista.core.pointset.PolyData
           Pyvista object representing the added structure.
    """
    if (color is None):
        color = "grey"
    
    if (opacity is None):
        opacity = .5
        
    if (size is None):
        size = 3
        
    mesh = fig.add_points(pts, color = color, render_points_as_spheres = True, point_size = size, opacity = opacity)
    
    return mesh

def add_title(fig, title):
    """
    Add a title to the figure
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    title : string
            Color of the points. Defaults to grey.
    Returns
    -------
    mesh : pyvista.core.pointset.PolyData
           Pyvista object representing the added structure.
    """
    
    fig.add_title(title)

def create_figure():
    """
    Create a Plotter/Figure object to draw the visualization.
    
    Returns
    -------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    """
    fig = pyvista.Plotter(window_size = (800, 600))
    
    return fig

def add_structure(fig, path, name, color = None, opacity = None, 
                  invert_x = False, invert_y = False, invert_z = False):
    """
    Add structures to the visualization.
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    path : str
           Path to the to be added structure (*.obj format).
    name : str
           Name of the added structure.
    color : string
            Color of the points. Defaults to grey.
    opacity : float 
              Opacitiy of the added structure. Defaults to 0.5.
              
    Returns
    -------
    mesh : pyvista.core.pointset.PolyData
           Pyvista object representing the added structure.
    """
    if (color is None):
        color = "grey"
        
    if (opacity is None):
        opacity = .5
    
    mesh = pyvista.read(path)
    
    if (invert_x):
        trans = np.eye(4); trans[0, 0] = -1
        mesh.transform(trans, inplace = True)
    if (invert_y):
        trans = np.eye(4); trans[1, 1] = -1
        mesh.transform(trans, inplace = True)
    if (invert_z):
        trans = np.eye(4); trans[2, 2] = -1
        mesh.transform(trans, inplace = True)
    
    mesh = mesh.decimate(.5)
    mesh = mesh.smooth(n_iter = 5, relaxation_factor = 0.1)
    fig.add_mesh(mesh, show_edges = False, smooth_shading = True, color = color, opacity = opacity)
        
    mesh["path"] = path
    mesh["name"] = name
    
    return mesh

def add_volume(fig, pts, name,
               grid_step_cnt = 100, max_filter_sz = 10, gaussian_filter_sigma = 6,
               opacity = None):
    """
    Add a volume to the visulization. This function may be used to plot 3D heatmaps.
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    pts : np.ndarray, shape(n, 4) or shape(n, 6)
          Datapoints used to compute the volume/heatmap. Dimensions are x, y, z, and color.
    name : str
           Name of the added volume.
    grid_step_cnt : int 
                    Resolution of the pointcloud: Defaults to 100.
    max_filter_sz : int
                    Employed to make points bigger, for a more cohesive heatmap. Defaults to 10.
    gaussian_filter_sigma : int
                            Employed to blur the volume, making it appear more cohesive. Defaults to 6.
    opacity : float 
              Opacitiy of the added structure. Default: 0.5.
    
    
    Returns
    -------
    mesh : pyvista.core.pointset.ImageData
           Pyvista object representing the added volume.
    """
    if (opacity is None):
        opacity = np.asarray([0, 0, 0.1, .3, .75]) / 4
    
    min_x = np.min(pts[:, 0]); max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1]); max_y = np.max(pts[:, 1])
    min_z = np.min(pts[:, 2]); max_z = np.max(pts[:, 2])
     
    step_sz = np.min([(max_x - min_x) / grid_step_cnt, (max_y - min_y) / grid_step_cnt, (max_z - min_z) / grid_step_cnt])
    space_x = np.linspace(min_x, max_x, int((max_x - min_x) / step_sz))
    space_y = np.linspace(min_y, max_y, int((max_y - min_y) / step_sz))
    space_z = np.linspace(min_z, max_z, int((max_z - min_z) / step_sz))
    
    def _config_color(space_x, space_y, space_z, pts, color):
        mesh_c = np.zeros((space_x.shape[0], space_y.shape[0], space_z.shape[0]))    
        for (pt_idx, pt) in enumerate(pts):
            x_idx = np.argmin(np.abs(space_x - pt[0]))
            y_idx = np.argmin(np.abs(space_y - pt[1]))
            z_idx = np.argmin(np.abs(space_z - pt[2]))
             
            mesh_c[x_idx - (x_idx == mesh_c.shape[0]),
                   y_idx - (y_idx == mesh_c.shape[1]),
                   z_idx - (y_idx == mesh_c.shape[2])] = color[pt_idx]
        
        mesh_c -= np.min(mesh_c); mesh_c /= np.max(mesh_c)
        if (max_filter_sz != 0):
            mesh_c = scipy.ndimage.maximum_filter(mesh_c, max_filter_sz)
        if (gaussian_filter_sigma != 0):
            mesh_c = scipy.ndimage.gaussian_filter(mesh_c, sigma = gaussian_filter_sigma, order = 0)
        mesh_c -= np.min(mesh_c); mesh_c /= np.max(mesh_c)
        
        return mesh_c
    
    if (pts.shape[1] == 4):
        mesh_c = _config_color(space_x, space_y, space_z, pts[:, :3], pts[:, 3])
    elif (pts.shape[1] == 6):
        mesh_c0 = _config_color(space_x, space_y, space_z, pts[:, :3], pts[:, 3])
        mesh_c1 = _config_color(space_x, space_y, space_z, pts[:, :3], pts[:, 4])
        mesh_c2 = _config_color(space_x, space_y, space_z, pts[:, :3], pts[:, 5])
        
        mesh_c = [mesh_c0, mesh_c1, mesh_c2]
    else:
        raise AssertionError("Each point has to come with either 1 or 3 color dimensions.")
    
    grid = pyvista.ImageData()
    grid.dimensions = np.asarray(mesh_c.shape) + 1
    grid.origin = ((max_x - min_x) / 2, (max_y - min_y) / 2, (max_z - min_z) / 2)
    grid.origin = ((min_x, min_y, min_z))
    grid.spacing = (step_sz, step_sz, step_sz)
    if (pts.shape[1] == 4):
        grid.cell_data["values-I"] = mesh_c.flatten(order = "F")
    if (pts.shape[1] == 6):
        grid.cell_data["values-I"] = np.asarray([mesh_c0.flatten(order = "F"), mesh_c1.flatten(order = "F"), mesh_c2.flatten(order = "F"), np.ones(mesh_c0.shape)]).T
    
    opac_range = np.arange(0, len(opacity)) / len(opacity)
    grid_opac = np.asarray([(np.argmax(opac_range > val)) for val in mesh_c.flatten(order = "F")])
    grid_opac = opacity[grid_opac]
    grid.cell_data["values-II"] = grid_opac 
     
    fig.add_volume(grid, scalars = "values-I", opacity = opacity)    
    grid["name"] = name
    return grid

def add_polygon(fig, pts, faces, name, opacity):
    """
    Add a polygon to the visualization.
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    pts : np.ndarray, shape(n_pts, 3)
          Points/Vertices defining the polygon.
    faces : np.ndarray, shape(face_cnt, vtx_cnt) or shape(face_cnt * (vtx_cnt + 1),)
            Faces connect the individual vertices. These have to passed as either list of lists wherein the
            inner lists are individual faces (e.g. [[1, 2, 3], [1, 4, 3]]) or as a flattened list wherein
            each face starts with a number identifying the amount of vertices used in the next face (e.g. [3, 1, 2, 3, 3, 1, 4, 3]).
    name : str
           Name of the added structure.
    opacity : float 
              Opacitiy of the added structure. Defaults to 0.5.
              
    Returns
    -------
    mesh : pyvista.core.pointset.PolyData
           Pyvista object representing the added structure.
    """
    if (len(faces.shape) == 2):
        loc_faces = np.asarray([[len(face), *face] for face in faces]); faces = faces.reshape(-1)
        polygon = pyvista.PolyData(pts, loc_faces)
    else:
        polygon = pyvista.PolyData(pts, faces)
    
    fig.add_mesh(polygon, opacity = opacity)
    
    polygon["name"] = name
    
    return polygon

def add_cen_grav(fig, pts):
    """
    Add a center of gravity to the visualization.
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    pts : np.ndarray, shape(n, 3)
          Datapoints used to compute the center of gravity from.
    """
    weights = np.copy(pts[:, 3])
    weights -= np.min(weights); weights /= np.max(weights)
    cen_pt = np.average(pts[:, :3], axis = 0, weights = weights)
    fig.add_points(cen_pt, color = "red", point_size = 20, render_points_as_spheres = True)

def show_figure(fig, screenshot_name):
    """
    Show the pyvista plotter.
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    screenshot_name : string
                      Name of the to be printed file.
    """ 
    if (screenshot_name is not None):
        fig.show(screenshot = screenshot_name)
    else:
        fig.show()
    
def set_black_background(fig):
    """
    Set background to black.
    
    Parameters
    ----------
    fig : pyvista.Plotter
         Plot/Figure object used to preview the visualization.
    """
    fig.set_background("black")
