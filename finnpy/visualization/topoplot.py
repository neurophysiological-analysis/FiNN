"""
Created on Jun 12, 2018.

Creates a topoplot from provided data, indicating either size of change or size of change and significance.

:author: voodoocode
"""

import numpy as np
import scipy.interpolate
import skimage.filters

import matplotlib  # @UnusedImport
import matplotlib.markers
matplotlib.use("QtAgg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import pathlib  # noqa: E402
import pyexcel_ods  # noqa: E402

class Topoplot():
    """
    Topoplot generation class.
    
    Initialization costs a couple of seconds due to mask generation.
    Performance advice: if possible, only generate a single topoplot object.
    
    Constructor. Currently supports: The extended 10-20 system for 64 channels - ext_10_20_64_ch
    
    Attributes
    ----------
    topoplot_mask_data : np.ndarray
                         Whether to mask channels.
    default_v_diff : float
                     default value for v_diff.
    win_sz : float
             Default window size.
    get_coords : callable
                 Function to reat the coordinates.
    signal_type : str
                  Signal type is either "EEG" or "MEG"
    
    Parameters
    ----------
    signal_type : str
                  Signal type is either "EEG" or "MEG"
                  
    Raises
    ------
    NotImplementedError
        If signal_type is neither 'EEG' nor 'MEG'.
    """
    
    # Mask for the topoplot color value data
    topoplot_mask_data: np.ndarray = None
    default_v_diff: float = 50
    win_sz: float = 1.3
    get_coords: callable = None
    signal_type: str = None

    def __init__(self, signal_type):
        if (signal_type not in ["EEG", "MEG"]):
            raise NotImplementedError("This setup has not yet been implemented")
        else:
            self.signal_type = signal_type
            self.get_coords = self._read_map
        
        self._generate_topoplot_mask()

    def _read_map(self, signal_type):
        """
        Read the coordinate map.
        
        Parameters
        ----------
        signal_type : str
                      Signal type is either "EEG" or "MEG"   
                      
        Returns
        -------
        dict
            Position of the individual channels.
        """
        map_path = str(pathlib.Path(__file__).parent.absolute()) + "/coord_map.ods"
        map_file = pyexcel_ods.read_data(map_path)[signal_type]
        while (len(map_file[-1]) == 0):  # Remove trailing empty rows
            map_file = map_file[:-1]
        map_file = np.asarray(map_file)
        
        ch_pos = dict()
        for (line_idx, _) in enumerate(map_file):
            ch_pos[str(map_file[line_idx][2])] = (map_file[line_idx][0], map_file[line_idx][1])
        
        return ch_pos

    def run(self, values, ch_name_list, 
            omit_channels = None, substitute_channels = None, 
            v_min = None, v_max = None, v_border_values = None, v_border_labels = None,
            file_path = None,
            screen_channels = False, annotate_ch_names = False, 
            ax = None):
        """
        Plot a 2D topomap.
        
        Parameters
        ----------
        values : np.ndarray, shape(ch_cnt, 1 or 3)
                 May either be a N x 3 or N x 1 matrix.
                 Dimensions #2 (boolean only) may be used to indicate significance before multiple comparison
                 correction and dimensions #3 (boolean only) may be used to indicate significance after multiple comparison correction.
        ch_name_list : list
                       Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        omit_channels : list
                        A list which channels are to be omitted. Channels are identified via names matching the ones specified in ch_names.
        substitute_channels : dict
                              A list of dictionaries. Each dictionary contains a 'tgt' section with a single string
                              defining the channel to be substituted and a second 'src' section which contains a list of strings,
                              defining channel names which are used to substitute the 'tgt' channel.
        v_min : float
                Minimal value on the color bar. If None, v_min is chosen as the minimum value within the data.
        v_max : float
                Maximuim value on the color bar. If None, v_max is chosen as the maximum value within the data.
        v_border_values : list
                          Where to put new ticks onto the color bar. v_min and v_max are always added as values.
                          The number of labels defined in v_border_labels must be exactly one element larger than
                          the number of elements in v_border_values.
        v_border_labels : list
                          Labels for the ticks on the color bar. The number of labels defined in v_border_labels
                          must be exactly one element larger than the number of elements in v_border_values.
        file_path : str
                    Path (including file name and file ending) were the file is stored. In case of None, the file is not saved.
        screen_channels : boolean
                          If true, channels are not drawn as a smoothed 2D plane, but a voroni diagram easening the identification of individual unexpected results.
        annotate_ch_names : boolean
                            If true, channels get annotate with their individual names.
        ax : matplotlib.axes.Axes
             Provide an axis object to embed the topoplot into.
        
        
        Returns
        -------
        matplotlib.pyplot.figure or tuple of (matplotlib.axes.Axes, matplotlib.pyplot.figure)
            - axes : matplotlib.axes.Axes
                     The axes object to easen the inclusion of a plot into a larger picture.
            - fig : matplotlib.axes.Axes
                    The figure object.
        """
        if (omit_channels is None):
            omit_channels = []
        if (substitute_channels is None):
            substitute_channels = []
        if (v_border_values is None):
            v_border_values = []
        if (v_border_labels is None):
            v_border_labels = [""]
        
        coords = list()
        coord_ref_list = self.get_coords(self.signal_type)
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
        
        created_ax = False
        if (ax is None):
            (fig, ax) = plt.subplots(1, 1)
            created_ax = True
        
        if (type(values) is not np.ndarray):
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
        self._add_color_bar(v_min, v_max, v_border_values, v_border_labels, ax)
        
        if ((file_path is None) is False and created_ax is True):
            fig.savefig(file_path)  # pylint: disable=possibly-used-before-assignment
        
        if (created_ax):
            return (fig, ax)
        else:
            return ax
    
    def _add_color_bar(self, v_min, v_max, v_border_values, v_border_labels, ax):
        """
        Add a color bar to the topoplot.
        
        Parameters
        ----------
        v_min : float
                Minimal value on the color bar. If None, v_min is chosen as the minimum value within the data.
        v_max : float
                Maximuim value on the color bar. If None, v_max is chosen as the maximum value within the data.
        v_border_values : list
                          Where to put new ticks onto the color bar. v_min and v_max are always added as values.
                          The number of labels defined in v_border_labels must be exactly one element larger than
                          the number of elements in v_border_values.
        v_border_labels : list
                          Labels for the ticks on the color bar. The number of labels defined in v_border_labels
        ax : matplotlib.axes.Axes
               The axes object to easen the inclusion of a plot into a larger picture.
        """
        sm      = plt.cm.ScalarMappable(cmap = plt.get_cmap("jet"), norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max))  # noqa: E221
        sm.set_array([])
        cbar    = plt.colorbar(sm, ax = ax)  # noqa: E221
    
        assert ((len(v_border_values) + 1) == len(v_border_labels))
    
        y_tick_list = [v_min] + v_border_values + [v_max]
        for y_ticksBorders in y_tick_list:
            cbar.ax.plot([cbar.ax.get_xlim()[0], cbar.ax.get_xlim()[1]], [y_ticksBorders, y_ticksBorders], linewidth = 1, color = "black")
        y_ticks = list()
        y_tick_labels = list()
        for y_tick_idx in np.arange(0, len(y_tick_list) - 1):
            if (v_border_labels[y_tick_idx] is not None and len(v_border_labels[y_tick_idx]) > 0):
                y_ticks.append((y_tick_list[y_tick_idx] + y_tick_list[y_tick_idx + 1]) / 2)
                y_tick_labels.append(v_border_labels[y_tick_idx])
    
        cbar.ax.get_yaxis().set_ticks(y_ticks)
        cbar.ax.get_yaxis().set_ticklabels(y_tick_labels, rotation = -90, va = 'center')
    
    def _refine_image(self, ax):
        """
        Add additional elements to the topoplot to make it visually more appealing.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
               The axes object to easen the inclusion of a plot into a larger picture.
        """
        # Add border of face
        circ = plt.Circle((0, 0), 1, color = "black", zorder = 11, linewidth = 1, fill = False)
        ax.add_artist(circ)
        
        line = plt.Line2D([-0.309, 0, 0.309], [0.9511, 1.2, 0.9511], color = "black", zorder = 11, linewidth = 1)
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
        Substitute and omits channels which are marked respectively.
        
        Parameters
        ----------
        values : np.ndarray, shape(ch_cnt, 1 or 3)
                 May either be a N x 3 or N x 1 matrix.
                 Dimensions #2 (boolean only) may be used to indicate significance before multiple comparison
                 correction and dimensions #3 (boolean only) may be used to indicate significance after multiple comparison correction.
        ch_name_list : list
                       Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        substitute_channels : dict
                              A list of dictionaries. Each dictionary contains a 'tgt' section with a single string
                              defining the channel to be substituted and a second 'src' section which contains a list of strings,
                              defining channel names which are used to substitute the 'tgt' channel.
        omit_channels : list
                        A list which channels are to be omitted. Channels are identified via names matching the ones specified in ch_names.
        
        Returns
        -------
        values : np.ndarray, shape(ch_cnt, 1 or 3)
                 The corrected values
        """
        if (len(substitute_channels) > 0):
            values = self._substitute_channels(values, ch_name_list, substitute_channels)
        if (len(omit_channels) > 0):
            values = self._omit_channels(values, ch_name_list, omit_channels)
            
        return values
    
    def _substitute_channels(self, values, ch_name_list, substitute_channels):
        """
        Substitute channels by overwriting each 'tgt' channel with the average of the respective 'src' channels.
        
        Parameters
        ----------
        values : np.ndarray, shape(ch_cnt, 1 or 3)
                 May either be a N x 3 or N x 1 matrix.
                 Dimensions #2 (boolean only) may be used to indicate significance before multiple comparison
                 correction and dimensions #3 (boolean only) may be used to indicate significance after multiple comparison correction.
        ch_name_list : list
                       Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        substitute_channels : dict
                              A list of dictionaries. Each dictionary contains a 'tgt' section with a single string
                              defining the channel to be substituted and a second 'src' section which contains a list of strings,
        
        Returns
        -------
        values : np.ndarray, shape(ch_cnt, 1 or 3)
                 The corrected values
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
        Omit channels by setting them to zero.
        
        Parameters
        ----------
        values : np.ndarray, shape(ch_cnt, 1 or 3)
                 May either be a N x 3 or N x 1 matrix.
                 Dimensions #2 (boolean only) may be used to indicate significance before multiple comparison
                 correction and dimensions #3 (boolean only) may be used to indicate significance after multiple comparison correction.
        ch_name_list : list
                       Names of the individual channels. Used for the spatial positioning of channels and the annotation of channels.
        omit_channels : list
                        A list which channels are to be omitted. Channels are identified via names matching the ones specified in ch_names.
        
        Returns
        -------
        values : np.ndarray, shape(ch_cnt, 1 or 3)
                 The corrected values
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
        """Generate a mask to hide areas of the topoplot to make it circular."""
        self.topoplot_mask_data = np.ones((1000, 1000))
        for x in range(self.topoplot_mask_data.shape[0]):
            xPos = -self.win_sz + self.win_sz * 2 / 1000 * x
            for y in range(self.topoplot_mask_data.shape[1]):
                yPos = -self.win_sz + self.win_sz * 2 / 1000 * y
                
                if ((np.power(xPos - 0, 2) + np.power(yPos - 0, 2)) >= (self.win_sz - 0.1)):
                    self.topoplot_mask_data[x, y] = np.nan
    
    def _interpolate_data(self, coords, values, screen_channels = False):
        """
        Interpolate the individual data points and hides anything 'outside' the head.
        
        Parameters
        ----------
        coords : np.ndarray, shape(ch_cnt, 3)
                 Coordinates of the individual points
        values : np.ndarray, shape(ch_cnt, 1)
                 Color values of the individual points
        screen_channels : boolean
                          If true, channels are not drawn as a smoothed 2D plane,
                          but a voroni diagram easening the identification of individual unexpected results.
        
        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            - mesh grid values : np.ndarray
                                 Color values.
            - x coordinates : np.ndarray
                              X-coordinates.
            - y coordinates : np.ndarray
                              Y-coordinates.
        """
        x = np.linspace(-self.win_sz, self.win_sz, 1000)
        y = np.linspace(-self.win_sz, self.win_sz, 1000)
        X, Y = np.meshgrid(x, y)
        
        if (screen_channels):
            data = scipy.interpolate.griddata((coords[0], coords[1]), values, (X, Y), method = "nearest")
        else:
            data = scipy.interpolate.griddata((coords[0], coords[1]), values, (X, Y), method = "cubic", fill_value = 0)
        
        data = data * self.topoplot_mask_data            
        
        return data, X, Y
    
    def _normalize_data(self, data, X, Y, v_min, v_max):
        """
        Normalize the topoplot data.
        
        Parameters
        ----------
        data : np.ndarray
               Color values.
        X : np.ndarray
            X-coordinates.
        Y : np.ndarray
            Y-coordinates.
        v_min : float
                Minimal color value. If None, v_min is chosen as the minimum value within the data.
        v_max : float
                Maximimal color value. If None, v_max is chosen as the maximum value within the data.
        
        Returns
        -------
        - normalized data : np.ndarray
                            Color-range normalized data.
        - v_min : float
                  Minimum color value
        - v_max : float
                  Maximum color value
         - v_diff : float
                    Color value range.
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
        mask[np.isnan(mask) is False] = 1
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
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
               The axes object to easen the inclusion of a plot into a larger picture.
        X : np.ndarray
            X-coordinates.
        Y : np.ndarray
            Y-coordinates.
        norm_data : np.ndarray
                    Color-range normalized data.
        v_min : float
                Minimal color value. If None, v_min is chosen as the minimum value within the data.
        v_max : float
                Maximimal color value. If None, v_max is chosen as the maximum value within the data.
        v_diff : float
                 Step size between individual color steps
        """
        levels = np.arange(v_min, v_max, v_diff)
        
        ax.contourf(X, Y, norm_data, cmap = plt.get_cmap("jet"), levels = levels, antialiased = False, zorder = 1)
    
    def _annotate_ch_sig(self, coords, ch_name_list, ax, signValues, omit_channels = None, substitute_channels = None):
        """
        Add channel positions and respective significance (if supplied).
        
        Parameters
        ----------
        coords : np.ndarray, shape(ch_cnt, 3)
                 Coordinates of the individual points
        ch_name_list : list
                       Names of the individual channels.
        ax : matplotlib.axes.Axes
               The axes object to easen the inclusion of a plot into a larger picture.
        signValues : list
                     significance values.
        omit_channels : list
                        A list which channels are to be omitted. Channels are identified via names matching the ones specified in ch_names.
        substitute_channels : dict
                              A list of dictionaries. Each dictionary contains a 'tgt' section with a single string
                              defining the channel to be substituted and a second 'src' section which contains a list of strings,
                              defining channel names which are used to substitute the 'tgt' channel.
        """
        if (type(signValues) is not np.ndarray):
            signValues = np.asarray(signValues)
        
        halfMarker = matplotlib.markers.MarkerStyle(marker = "o", fillstyle = "bottom")
        
        for chIdx in range(0, len(ch_name_list)):
            
            # In case a channel is either substituted or not omitted, the corresponding significance is also not displayed
            if ((ch_name_list[chIdx] in [subName["tgt"] for subName in substitute_channels])
                or (ch_name_list[chIdx] in omit_channels)):  # noqa: E129, W503
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
        Annotate the individual channels with their names.
        
        Parameters
        ----------
        coords : np.ndarray, shape(ch_cnt, 3)
                 Coordinates of the individual points
        ch_name_list : list
                       Names of the individual channels.
        ax : matplotlib.axes.Axes
               The axes object to easen the inclusion of a plot into a larger picture.
        """
        for chIdx in range(0, len(ch_name_list)):
            text = ch_name_list[chIdx]
            ax.annotate(text, [coords[0, chIdx], coords[1, chIdx]], zorder = 3)
