'''
Created on Jun 12, 2018

Creates a topoplot from provided data, indicating either size of change or size of change and significance.

:author: voodoocode
'''

import numpy as np
import scipy.interpolate
import skimage.filters

import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.markers
import matplotlib.pyplot as plt

import pathlib
import pyexcel_ods

class Topoplot():
    """
    topoplot generation class. Initialition costs a couple of seconds due to mask generation.
    Performance advice: if possible, only generate a single topoplot object.
    
    :param mode: Mode is either "EEG" or "MEG"
    """
    
    #Mask for the topoplot color value data
    topoplot_mask_data = None
    default_v_diff = 50
    win_sz = 1.3
    get_coords = None
    mode = None

    def __init__(self, mode):
        """
        Constructor. Currently supports: The extended 10-20 system for 64 channels - ext_10_20_64_ch
        
        :param mode: Mode is either "EEG" or "MEG"
        """
        
        if (mode not in ["EEG", "MEG"]):
            raise NotImplementedError("This setup has not yet been implemented")
        else:
            self.mode = mode
            self.get_coords = self._read_map
        
        self._generate_topoplot_mask()

    def _read_map(self, mode):
        """
        Reads the coordinate map.
        
        :param mode: Identifies which coordinate map to read.
        """
        map_path = str(pathlib.Path(__file__).parent.absolute()) + "/coord_map.ods"
        #map_file = pyexcel_ods.read_data("methods/visualization_map.ods")[mode]
        map_file = pyexcel_ods.read_data(map_path)[mode]
        while (len(map_file[-1]) == 0): #Remove trailing empty rows
            map_file = map_file[:-1]
        map_file = np.asarray(map_file)
        
        ch_pos = dict()
        for (line_idx, _) in enumerate(map_file):
            ch_pos[str(map_file[line_idx][2])] = (map_file[line_idx][0], map_file[line_idx][1])
        
        return ch_pos

    def run(self, values, ch_name_list, 
            omit_channels = None, substitute_channels = None, 
            v_min = None, v_max = None, v_border_values = [], v_border_labels = [""],
            file_path = None,
            screen_channels = False, annotate_ch_names = False):
        """
        Plots a 2D topomap
        
        :param values: May either be a N x 3 or N x 1 matrix. Dimensions #2 (boolean only) may be used to indicate significance before multiple comparison correction and dimensions #3 (boolean only) may be used to indicate significance after multiple comparison correction.
        :param ch_names: Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        :param omit_channels: A list which channels are to be omitted. Channels are identified via names matching the ones specified in ch_names.
        :param substitute_channels: A list of dictionaries. Each dictionary contains a 'tgt' section with a single string defining the channel to be substituted and a second 'src' section which contains a list of strings, defining channel names which are used to substitute the 'tgt' channel.
        :param v_min: Minimal value on the color bar. If None, v_min is chosen as the minimum value within the data.
        :param v_max: Maximuim value on the color bar. If None, v_max is chosen as the maximum value within the data.
        :param v_border_values: Where to put new ticks onto the color bar. v_min and v_max are always added as values. The number of labels defined in v_border_labels must be exactly one element larger than the number of elements in v_border_values.
        :param v_border_labels: Labels for the ticks on the color bar. The number of labels defined in v_border_labels must be exactly one element larger than the number of elements in v_border_values.
        :param file_path: Path (including file name and file ending) were the file is stored. In case of None, the file is not saved.
        :param screen_channels: If true, channels are not drawn as a smoothed 2D plane, but a voroni diagram easening the identification of individual unexpected results.
        :param annotate_ch_names: If true, channels get annotate with their individual names.
        
        :return: The figure and the axes object to easen the inclusion of a plot into a larger picture.
        """
        
        coords = list()
        coord_ref_list = self.get_coords(self.mode)
        filt_values = list()
        filt_ch_names = list()
        for (ch_name_idx, ch_name) in enumerate(ch_name_list):
            if (ch_name not in coord_ref_list.keys()):
                continue
            
            coords.append(coord_ref_list[ch_name])
            filt_values.append(values[ch_name_idx])
            filt_ch_names.append(ch_name)
        
        values = np.asarray(filt_values)
        ch_name_list = filt_ch_names 
        coords = np.asarray(coords, dtype = np.float32)
        coords = coords.transpose()
        
        (fig, ax) = plt.subplots(1, 1)
        
        if (type(values) != np.ndarray):
            values = np.asarray(values)
        if (len(values.shape) == 1):
            values = np.expand_dims(values, axis = 1)
        
        values[:, 0] = self._mask_data(values[:, 0], ch_name_list, substitute_channels, omit_channels)
        (data, X, Y) = self._interpolate_data(coords, values[:, 0], screen_channels)

        (norm_data, v_min, v_max, v_diff) = self._normalize_data(data, X, Y, v_min, v_max)
    
        self._draw_figure(ax, X, Y, norm_data, v_min, v_max, v_diff)
        
        if (len(values.shape) > 1):
            self._annotate_ch_sig(coords, ch_name_list, ax, values[:, :], omit_channels, substitute_channels)
        
        if (annotate_ch_names):
            self._add_ch_names(coords, ch_name_list, ax)
            
        self._refine_image(ax)
        self._add_color_bar(v_min, v_max, v_border_values, v_border_labels)
        
        if ((file_path is None) == False):
            fig.savefig(file_path)
        
        return (fig, ax)
    
    def _add_color_bar(self, v_min, v_max, v_border_values, v_border_labels):
        """
        Adds a color bar to the topoplot.
        
        :param v_min: The minimal value on the color bar.
        :param v_max: The maximimal value on the color bar.
        :param v_border_values: Where to put new ticks onto the color bar. v_min and v_max are always added as values. The number of labels defined in v_border_labels must be exactly 
        one element larger than the number of elements in v_border_values.
        :param v_border_labels: Labels for the ticks on the color bar. The number of labels defined in v_border_labels must be exactly 
        one element larger than the number of elements in v_border_values. 
        """
        sm      = plt.cm.ScalarMappable(cmap = plt.get_cmap("jet"), norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max))
        sm.set_array([])
        cbar    = plt.colorbar(sm)
    
        assert((len(v_border_values) + 1) == len(v_border_labels))
    
        y_tick_list = [v_min] + v_border_values + [v_max]
        for y_ticksBorders in y_tick_list:
            cbar.ax.plot([cbar.ax.get_xlim()[0], cbar.ax.get_xlim()[1]], [y_ticksBorders, y_ticksBorders], linewidth = 1, color = "black")
        y_ticks = list()
        y_tick_labels = list()
        for y_tick_idx in np.arange(0, len(y_tick_list) - 1):
            if (v_border_labels[y_tick_idx] is not None and len(v_border_labels[y_tick_idx]) > 0):
                y_ticks.append((y_tick_list[y_tick_idx] + y_tick_list[y_tick_idx + 1])/2)
                y_tick_labels.append(v_border_labels[y_tick_idx])
    
        cbar.ax.get_yaxis().set_ticks(y_ticks)
        cbar.ax.get_yaxis().set_ticklabels(y_tick_labels, rotation = -90, va = 'center')
    
    def _refine_image(self, ax):
        """
        Adds additional elements to the topoplot to make it visually more appealing.
        
        :param ax: The axes object of the topoplot
        """
        #Add border of face
        circ = plt.Circle((0, 0), 1, color = "black", zorder = 11, linewidth = 1, fill = False)
        ax.add_artist(circ)
        
        line = plt.Line2D([-0.309, 0, 0.309], [0.9511, 1.2 , 0.9511], color = "black", zorder = 11, linewidth = 1)
        ax.add_artist(line)
        
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        ax.set_xlim(-self.win_sz, self.win_sz)
        ax.set_ylim(-self.win_sz, self.win_sz)
    
    def _mask_data(self, values, ch_name_list, substitute_channels, omit_channels):
        """
        Substitutes and omits channels which are marked respectively.
        
        :param values: The original values to be plotted
        :param ch_name_list: Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        :param omit_channels: A list which channels are to be omitted. Channels are identified via names matching the ones specified in ch_names.
        :param substitute_channels: A list of dictionaries. Each dictionary contains a 'tgt' section with a single string defining the channel to be substituted and 
        a second 'src' section which contains a list of strings, defining channel names which are used to substitute the 'tgt' channel.
        
        :return: The corrected values
        """    
        
        if ((substitute_channels is None or substitute_channels == False) == False):
            values = self._substitute_channels(values, ch_name_list, substitute_channels)
        if ((omit_channels is None or omit_channels == False) == False):
            values = self._omit_channels(values, ch_name_list, omit_channels)
            
        return values
    
    def _substitute_channels(self, values, ch_name_list, substitute_channels):
        """
        Substitutes channels by overwriting each 'tgt' channel with the average of the respective 'src' channels
        
        :param values: The original values to be plotted
        :param ch_name_list: Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        :param substitute_channels: A list of dictionaries. Each dictionary contains a 'tgt' section with a single string defining the channel to be substituted and 
        a second 'src' section which contains a list of strings, defining channel names which are used to substitute the 'tgt' channel.
        
        :return: The corrected values
        """
        mod_ch_name_list = [ch_name for ch_name in ch_name_list]
        
        for sub_list in substitute_channels:
            tgt_ch_name = sub_list["tgt"]
            
            if (tgt_ch_name not in mod_ch_name_list):
                continue
            tgt_idx = mod_ch_name_list.index(tgt_ch_name)
            
            src_idx = []
            for src_ch_name in sub_list["src"]:
                tmp = mod_ch_name_list.index(src_ch_name)
                src_idx.append(tmp)
                
            values[tgt_idx] = np.mean(values[np.asarray(src_idx)])
        
        return values
    
    def _omit_channels(self, values, ch_name_list, omit_channels):
        """
        Omits channels by setting them to zero.
        
        :param values: The original values to be plotted
        :param ch_name_list: Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        :param omit_channels: A list which channels are to be omitted. Channels are identified via names matching the ones specified in ch_names.
        
        :return: The corrected values    
        """
        
        mod_ch_name_list = [ch_name for ch_name in ch_name_list]
        
        for ch_name in omit_channels:
            if (ch_name not in mod_ch_name_list):
                continue
            idx = mod_ch_name_list.index(ch_name)
            
            if (idx <= len(values)):
                values[idx] = 0
               
        return values
    
    def _generate_topoplot_mask(self):
        """
        Generates a mask to hide areas of the topoplot to make it circular
        """
        self.topoplot_mask_data = np.ones((1000, 1000))
        for x in range(self.topoplot_mask_data.shape[0]):
            xPos = -self.win_sz + self.win_sz*2/1000 * x
            for y in range(self.topoplot_mask_data.shape[1]):
                yPos = -self.win_sz + self.win_sz*2/1000 * y
                
                if ((np.power(xPos - 0, 2) + np.power(yPos - 0, 2)) >= (self.win_sz - 0.1)):
                    self.topoplot_mask_data[x, y] = np.nan
    
    def _interpolate_data(self, coords, values, screen_channels = False):
        """
        Interpolates the individual data points and hides anything 'outside' the head
        
        :param coords: Coordinates of the individual points
        :param values: Color values of the individual points
        :param screen_channels: If true, channels are not drawn as a smoothed 2D plane, but a voroni diagram easening the identification of individual unexpected results.
        
        :return mesh grid values, x coordinates and y coordinates
        """
        x = np.linspace(-self.win_sz, self.win_sz, 1000)
        y = np.linspace(-self.win_sz, self.win_sz, 1000)
        X, Y = np.meshgrid(x,y)
        
        if (screen_channels):
            data = scipy.interpolate.griddata((coords[0], coords[1]), values, (X, Y), method = "nearest")
        else:
            data = scipy.interpolate.griddata((coords[0], coords[1]), values, (X, Y), method = "cubic", fill_value = 0)
        
        data = data * self.topoplot_mask_data            
        
        return data, X, Y
    
    def _normalize_data(self, data, X, Y, v_min, v_max):
        """
        Normalizes the topoplot data
        
        :param data: Color values
        :param X: x coordinates
        :param Y: y coordinates
        :param v_min: Minimal color value. If None, v_min is chosen as the minimum value within the data.
        :param v_max: Maximimal color value. If None, v_max is chosen as the maximum value within the data.
        
        :return: normalized data, v_min, v_max and v_diff
        """
        
        if (type(data) is not np.ndarray):
            norm_data = data((X, Y))
        else:
            norm_data = data
        norm_data = np.nan_to_num(norm_data)    
        norm_data = skimage.filters.gaussian(norm_data, sigma = 7)
        if (type(data) is not np.ndarray):
            mask = data((X, Y))
        else:
            mask = data
        mask[np.isnan(mask) == False] = 1
        norm_data = norm_data * mask
        
        if (v_min is None and v_max is None):
            tmp = np.max((np.abs(np.nanmin(norm_data)), np.abs(np.nanmax(norm_data))))
            v_max = tmp
            v_min = -tmp
        else:
            if (v_min is None):
                v_min = np.nanmin(norm_data)
            if (v_max is None):
                v_max = np.nanmax(norm_data)

        v_diff = (v_max - v_min) / self.default_v_diff 
                
        if (v_min is not None):
            norm_data[norm_data <= v_min + v_diff * 2] = v_min + v_diff * 2
        if (v_max is not None):
            norm_data[norm_data >= v_max - v_diff * 2] = v_max - v_diff * 2
                
        return (norm_data, v_min, v_max, v_diff)
    
    def _draw_figure(self, ax, X, Y, norm_data, v_min, v_max, v_diff):
        """
        Draws the contour of the topoplot.
        
        :param ax: Axes object of the topoplot
        :param X: x coordinates
        :param Y: y coordinates
        :param norm_data: normalized color values.
        :param v_min: Minimal color value. If None, v_min is chosen as the minimum value within the data.
        :param v_max: Maximimal color value. If None, v_max is chosen as the maximum value within the data.
        :param v_diff: Step size between individual color steps
        """
        
        levels = np.arange(v_min, v_max, v_diff)
        
        ax.contourf(X, Y, norm_data, cmap = plt.get_cmap("jet"), levels = levels, antialiased = False, zorder = 1)
    
    def _annotate_ch_sig(self, coords, ch_name_list, ax, signValues, omit_channels = None, substitute_channels = None):
        """
        Adds channel positions and respective significance (if supplied)
        
        :param coords: Coordinates of the individual points.
        :param ch_name_list: Names of the individual channels.
        :param ax: axes object to draw onto.
        :param signValues: significance values.
        :param omit_channels: Channels omitted from visualization
        :param substitute_channels: Channels substituted in the visualization
        """
        
        if (type(signValues) is not np.ndarray):
            signValues = np.asarray(signValues)
        
        halfMarker = matplotlib.markers.MarkerStyle(marker = "o", fillstyle = "bottom")
        
        for chIdx in range(0, len(ch_name_list)):
            
            # In case a channel is either substituted or not omitted, the corresponding significance is also not displayed
            if ((ch_name_list[chIdx] in [subName["tgt"] for subName in substitute_channels]) 
                or (ch_name_list[chIdx] in omit_channels)):
                continue
            
            if (len(signValues.shape) == 2 and len(signValues[0, :]) == 3):
                if (signValues[chIdx, 1] == 1 and signValues[chIdx, 2] == 0):
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = 'black', s = 24, marker="o", zorder = 2)
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = "white", s = 14, marker=halfMarker, zorder = 3)
                elif (signValues[chIdx, 2] == 1 and signValues[chIdx, 2] == 1):
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = 'black', s = 24, marker="o", zorder = 2)
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = "white", s = 14, marker="o", zorder = 3)
                else:
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = 'black', s = 24, marker="o", zorder = 2)
                    
            elif (len(signValues.shape) == 2 and len(signValues[0, :]) == 2):
                if (signValues[chIdx, 1] == 1):
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = 'black', s = 24, marker="o", zorder = 2)
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = "white", s = 14, marker=halfMarker, zorder = 3)
                else:
                    ax.scatter(coords[0, chIdx], coords[1, chIdx], color = 'black', s = 24, marker="o", zorder = 2)
            else:
                ax.scatter(coords[0, chIdx], coords[1, chIdx], color = 'black', s = 24, marker="o", zorder = 2)
    
    def _add_ch_names(self, coords, ch_name_list, ax):
        """
        Annotates the individual channels with their names
        
        :param coords: Coordinates of the individual channels.
        :param ch_name_list: Names of the individual channels.
        :param ax: axis object to be annotated
        """
        
        for chIdx in range(0, len(ch_name_list)):
            text = ch_name_list[chIdx]
            ax.annotate(text, [coords[0, chIdx], coords[1, chIdx]], zorder = 3)
            
    def _get_eeg_ch_coords(self):
        """
        
        To be removed into a separate csv file at a later point.
        
        """
        coords = {
            "Cz" : [0, 0],
            "C1" : [-0.201729106628242, 0],
            "C3" : [-0.400576368876081, 0],
            "C5" : [-0.602305475504323, 0],
            "T7" : [-0.801152737752161, 0],
            "T9" : [-1, 0],
            "C2" : [0.201729106628242, 0],
            "C4" : [0.400576368876081, 0],
            "C6" : [0.602305475504323, 0],
            "T8" : [0.801152737752161, 0],
            "T10" : [1, 0],
            "CPz" : [0, -0.25089605734767],
            "CP1" : [-0.195965417867435, -0.247311827956989],
            "CP3" : [-0.386167146974063, -0.254480286738351],
            "CP5" : [-0.585014409221902, -0.279569892473118],
            "TP7" : [-0.763688760806916, -0.308243727598566],
            "TP9" : [-0.953890489913545, -0.3584229390681],
            "CP2" : [0.195965417867435, -0.247311827956989],
            "CP4" : [0.386167146974063, -0.254480286738351],
            "CP6" : [0.585014409221902, -0.279569892473118],
            "TP8" : [0.763688760806916, -0.308243727598566],
            "TP10" : [0.953890489913545, -0.3584229390681],
            "Pz" : [0, -0.501792114695341],
            "P1" : [-0.164265129682997, -0.505376344086022],
            "P3" : [-0.328530259365994, -0.508960573476702],
            "P5" : [-0.492795389048991, -0.53405017921147],
            "P7" : [-0.648414985590778, -0.587813620071685],
            "P9" : [-0.812680115273775, -0.734767025089606],
            "P2" : [0.164265129682997, -0.505376344086022],
            "P4" : [0.328530259365994, -0.508960573476702],
            "P6" : [0.492795389048991, -0.53405017921147],
            "P8" : [0.648414985590778, -0.587813620071685],
            "P10" : [0.812680115273775, -0.734767025089606],
            "POz" : [0, -0.74910394265233],
            "PO3" : [-0.273775216138329, -0.727598566308244],
            "PO7" : [-0.472622478386167, -0.810035842293907],
            "PO9" : [-0.636887608069164, -0.949820788530466],
            "PO4" : [0.273775216138328, -0.727598566308244],
            "PO8" : [0.472622478386167, -0.810035842293907],
            "PO10" : [0.636887608069164, -0.949820788530466],
            "Oz" : [0, -1],
            "O1" : [-0.247838616714697, -0.949820788530466],
            "O2" : [0.247838616714697, -0.949820788530466],
            "Iz" : [0, -1.25089605734767],
            "FCz" : [0, 0.25089605734767],
            "FC1" : [-0.195965417867435, 0.247311827956989],
            "FC3" : [-0.386167146974063, 0.254480286738351],
            "FC5" : [-0.585014409221902, 0.279569892473118],
            "FT7" : [-0.763688760806916, 0.308243727598566],
            "FT9" : [-0.953890489913545, 0.3584229390681],
            "FC2" : [0.195965417867435, 0.247311827956989],
            "FC4" : [0.386167146974063, 0.254480286738351],
            "FC6" : [0.585014409221902, 0.279569892473118],
            "FT8" : [0.763688760806916, 0.308243727598566],
            "FT10" : [0.953890489913545, 0.3584229390681],
            "Fz" : [0, 0.501792114695341],
            "F1" : [-0.164265129682997, 0.505376344086022],
            "F3" : [-0.328530259365994, 0.508960573476702],
            "F5" : [-0.492795389048991, 0.53405017921147],
            "F7" : [-0.648414985590778, 0.587813620071685],
            "F9" : [-0.812680115273775, 0.734767025089606],
            "F2" : [0.164265129682997, 0.505376344086022],
            "F4" : [0.328530259365994, 0.508960573476702],
            "F6" : [0.492795389048991, 0.53405017921147],
            "F8" : [0.648414985590778, 0.587813620071685],
            "F10" : [0.812680115273775, 0.734767025089606],
            "AFz" : [0, 0.74910394265233],
            "AF3" : [-0.273775216138329, 0.727598566308244],
            "AF7" : [-0.472622478386167, 0.810035842293907],
            "AF9" : [-0.636887608069164, 0.949820788530466],
            "AF4" : [0.273775216138328, 0.727598566308244],
            "AF8" : [0.472622478386167, 0.810035842293907],
            "AF10" : [0.636887608069164, 0.949820788530466],
            "Fpz" : [0, 1],
            "Fp1" : [-0.247838616714697, 0.949820788530466],
            "Fp2" : [0.247838616714697, 0.949820788530466],
            "Nz" : [0, 1.25089605734767],
        }
        return coords
    
    
    
    
    
    
    
