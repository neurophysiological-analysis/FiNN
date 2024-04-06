'''
Created on Feb 2, 2023

@author: voodoocode
'''

import numpy as np
import mayavi.mlab
import mayavi.tools
import scipy.ndimage
import tvtk.util.ctf

def plot_circle(x = 0, y = 0, z = 0, r = .2, c_val = 0, color = [1, 0, 0], colormap = None, scale_mode = 'none'):
    if (colormap is None):
        mayavi.mlab.points3d(x, y, z, scale_factor = r, color = tuple(np.asarray(color) * c_val))
    else:        
        mayavi.mlab.points3d(x, y, z, c_val, scale_factor = r, colormap = colormap, scale_mode = scale_mode)

def plot_contour3d_data(data, magnification = None, contours = 3, 
                        nanmean_filter_sz = 2, gaussian_filter_sz = .75, 
                        color = (.5, 0, 1)):
    pre_x = np.asarray(data[1:, 0], dtype = float); pre_y = np.asarray(data[1:, 1], dtype = float)
    pre_z = np.asarray(data[1:, 2], dtype = float); pre_c = np.asarray(data[1:, 3], dtype = float)

    mean_x = np.mean(pre_x)
    mean_y = np.mean(pre_y)
    mean_z = np.mean(pre_z)
    
    res = np.ceil(np.max([np.max(pre_x) - np.min(pre_x), np.max(pre_y) - np.min(pre_y), np.max(pre_z) - np.min(pre_z)]))
    res = int(np.ceil((res + 4) * magnification))

    (x_pos, y_pos, z_pos) = np.mgrid[:(res + 1), :(res + 1), :(res + 1)]
    c_pos = np.zeros(x_pos.shape)
    
    c_pos_cnt = np.zeros(c_pos.shape)
    for pre_idx in range(len(pre_c)):
        try:
            c_pos[int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5),
                  int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5),
                  int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5)] += pre_c[pre_idx]
            c_pos_cnt[int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5),
                      int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5),
                      int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5)] += 1
        except:
            print("A")                          
    c_pos[c_pos_cnt != 0] /= c_pos_cnt[c_pos_cnt != 0]
    
    x_pos = (x_pos - res/2-.5)/magnification+mean_x
    y_pos = (y_pos - res/2-.5)/magnification+mean_y
    z_pos = (z_pos - res/2-.5)/magnification+mean_z

    c_pos[c_pos == 0] = np.nan
    c_pos = scipy.ndimage.generic_filter(c_pos, np.nanmean, nanmean_filter_sz)
    c_pos[np.isnan(c_pos)] = 0
    c_pos = scipy.ndimage.gaussian_filter(c_pos, sigma = gaussian_filter_sz)
    
    c_pos -= np.min(c_pos)
    c_pos /= np.max(c_pos)
    
    mayavi.mlab.contour3d(x_pos, y_pos, z_pos, c_pos, color = color, opacity = .3, contours = contours)
    
def plot_contour3d_data2(data, magnification = None, contours = 3, 
                         nanmean_filter_sz = 2, gaussian_filter_sz = .75, 
                         color = (.5, 0, 1)):
    pre_x = np.asarray(data[1:, 0], dtype = float); pre_y = np.asarray(data[1:, 1], dtype = float)
    pre_z = np.asarray(data[1:, 2], dtype = float); pre_c = np.asarray(data[1:, 3], dtype = float)

    pre_x += np.min(pre_x); pre_x /= np.max(pre_x); 
    
    res = np.ceil(np.max([np.max(pre_x) - np.min(pre_x), np.max(pre_y) - np.min(pre_y), np.max(pre_z) - np.min(pre_z)]))
    res = int(np.ceil((res + 2) * magnification))

    (x_pos, y_pos, z_pos) = np.mgrid[:(res + 1), :(res + 1), :(res + 1)]
    c_pos = np.zeros(x_pos.shape)
    c_pos_cnt = np.zeros(c_pos.shape)
    for pre_idx in range(len(pre_c)):
        c_pos[int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5),
              int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5),
              int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5)] += pre_c[pre_idx]
        c_pos_cnt[int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5),
                  int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5),
                  int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5)] += 1
    print(int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5), 
          int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5), 
          int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5))
    c_pos[c_pos_cnt != 0] /= c_pos_cnt[c_pos_cnt != 0]
    
    x_pos = (x_pos - res/2-.5)/magnification+mean_x
    y_pos = (y_pos - res/2-.5)/magnification+mean_y
    z_pos = (z_pos - res/2-.5)/magnification+mean_z

    c_pos[c_pos == 0] = np.nan
    c_pos = scipy.ndimage.generic_filter(c_pos, np.nanmean, nanmean_filter_sz)
    c_pos[np.isnan(c_pos)] = 0
    c_pos = scipy.ndimage.gaussian_filter(c_pos, sigma = gaussian_filter_sz)
    
    c_pos -= np.min(c_pos)
    c_pos /= np.max(c_pos)
    
    mayavi.mlab.contour3d(x_pos, y_pos, z_pos, c_pos, color = color, opacity = .3, contours = contours)

def plot_volumetric_data(data, magnification = None,
                         nanmean_filter_sz = None, gaussian_filter_sz = .75,
                         ctf_pts = None, otf_pts = None, vmax = .8, 
                         mode = "mean"):
    pre_x = np.asarray(data[:, 0], dtype = float); pre_y = np.asarray(data[:, 1], dtype = float)
    pre_z = np.asarray(data[:, 2], dtype = float); pre_c = np.asarray(data[:, 3], dtype = float)
    
    res = np.ceil(np.max([np.max(pre_x) - np.min(pre_x), np.max(pre_y) - np.min(pre_y), np.max(pre_z) - np.min(pre_z)]))
    res = int(np.ceil(res * magnification))
    
    mean_x = np.mean([np.max(pre_x), np.min(pre_x)])
    mean_y = np.mean([np.max(pre_y), np.min(pre_y)])
    mean_z = np.mean([np.max(pre_z), np.min(pre_z)])

    (x_pos, y_pos, z_pos) = np.mgrid[:(res+1), :(res+1), :(res+1)]
    
    if (mode == "mean" or mode == "log_mean"):
        c_pos = np.zeros(x_pos.shape)
        c_pos_cnt = np.zeros(c_pos.shape)
        for pre_idx in range(len(pre_c)):
            c_pos[int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5),
                  int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5),
                  int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5)] += pre_c[pre_idx]
            c_pos_cnt[int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5),
                      int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5),
                      int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5)] += 1
        c_pos[c_pos_cnt != 0] /= c_pos_cnt[c_pos_cnt != 0]
    
    elif(mode == "median"):
        c_pos = [[[list() for _ in range(51)] for _ in range(51)] for _ in range(51)]
        for pre_idx in range(len(pre_c)):
            c_pos[int((pre_x[pre_idx] - mean_x) * magnification + res/2 + .5)][int((pre_y[pre_idx] - mean_y) * magnification + res/2 + .5)][int((pre_z[pre_idx] - mean_z) * magnification + res/2 + .5)].append(pre_c[pre_idx])
        for idx_x in range(51):
            for idx_y in range(51):
                for idx_z in range(51):
                     c_pos[idx_x][idx_y][idx_z] = np.median(c_pos[idx_x][idx_y][idx_z]) if (len(c_pos[idx_x][idx_y][idx_z]) != 0) else 0
        c_pos = np.asarray(c_pos, dtype = float)
    
    #print(np.min(c_pos[c_pos>0]), np.max(c_pos))
    
    x_pos = (x_pos - res/2-.5)/magnification+mean_x
    y_pos = (y_pos - res/2-.5)/magnification+mean_y
    z_pos = (z_pos - res/2-.5)/magnification+mean_z
    
    if (mode == "log_mean"):
        c_pos[c_pos != 0] = np.log(c_pos[c_pos != 0])
        c_pos += 1
    
    if (nanmean_filter_sz is not None):
        c_pos[c_pos == 0] = np.nan
        c_pos = scipy.ndimage.generic_filter(c_pos, np.nanmean, nanmean_filter_sz)
        c_pos[np.isnan(c_pos)] = 0
        c_pos = scipy.ndimage.gaussian_filter(c_pos, sigma = gaussian_filter_sz)
    
    #print(np.min(c_pos[c_pos>0]), np.max(c_pos))
    
    c_pos -= np.min(c_pos)
    c_pos /= np.max(c_pos)
    volume = mayavi.mlab.pipeline.volume(mayavi.tools.pipeline.scalar_field(x_pos, y_pos, z_pos, c_pos), vmin = np.min(c_pos), vmax = vmax)
    
    if (otf_pts is not None):
        otf = tvtk.util.ctf.PiecewiseFunction()
        for otf_pt in otf_pts:
            otf.add_point(otf_pt[0], otf_pt[1])
        volume._otf = otf
        volume._volume_property.set_scalar_opacity(otf)
        
    if (ctf_pts is not None):
        ctf = tvtk.util.ctf.ColorTransferFunction()
        for ctf_pt in ctf_pts:
            ctf.add_rgb_point(ctf_pt[0], ctf_pt[1], ctf_pt[2], ctf_pt[3])
        volume._ctf = ctf
        volume._volume_property.set_color(ctf)
        
    return (volume, c_pos, ctf_pts, otf_pts)

def create_figure(width = 800, height = 800, 
                  bgcolor = (1, 1, 1)):
    return mayavi.mlab.figure(size = (width, height), bgcolor = bgcolor)
    
def show_figure():
    mayavi.mlab.show()








