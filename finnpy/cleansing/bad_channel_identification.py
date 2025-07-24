"""
Created on Jun 2, 2020.

This module implements a function to identify bad channels based on increased/decreased power within a channel compared to the majority of other channels. 

:author: voodoocode
"""

import numpy as np

import PyQt6.QtWidgets
import PyQt6.QtGui
import PyQt6.QtCore
import functools
import matplotlib.backends.backend_qtagg
import scipy.signal
import multiprocessing
import warnings

import finnpy.cleansing.outlier_removal  # @UnresolvedImport

idenfity_faulty_visual_inspection_lock = multiprocessing.Lock()

def run(data, ch_names, fs, ref_areas = None, broadness = 3, visual_inspection = True):
    """
    Identify bad channels.
    
    Identifies which channels have substantially more or less power in the frequency ranges defined by ref_areas.
    Channels whose power is more different than *broadness* (default: 3) standard deviations will be primed as faulty channels.
    In case visual inspection is *active* (default: True), the automatic results can be further refined via manual selection.
    Attention: Function is parallelized. Sensitive parts are placed within a locked area to avoid unexpected behaviour. 
    
    Parameters
    ----------
    data : np.ndarray, shape(ch_cnt, samples)
                       Input data.
    ch_names : list, len(ch_cnt)
               Names of the channels. Used for visualization purposes only.
               The order has to match the channel order of data.
    fs: list, len(ch_cnt)
        List of sampling frequencies for each channel.
    ref_areas: list of lists
               Frequency bands which may be used to identify channels with substantially more/less power than others.
               Important: Should not contain frequency bands of interest.
    broadness : float
                Number of standard deviations threshold by which channels are automatically categorized as faulty.
                In case visual inspection is enabled (recommended) this only results in priming the channels.
    visual_inspection : boolean
                        Toggles visual inspection on and off.
    
    Returns
    -------
    tuple of (list, list, list)
        - valid_list: list
                      List of valid channels.
        - invalid_list : list
                         List of invalid channels.
        - scores : list
                   Z-scores of the channels.
    """
    if (ref_areas is None):
        ref_areas = [[105, 120], [135, 145], [155, 195]]
    
    score_list = list()
    pow_oi = list()
    for ch_idx in range(0, len(data)):
        (_, power) = scipy.signal.welch(data[ch_idx], fs = fs[ch_idx], nfft = int(fs[ch_idx]), nperseg = int(fs[ch_idx]), noverlap = fs[ch_idx] // 2)
        pow_oi.append(power)
    pow_oi = np.asarray(pow_oi)
    
    # Filter reference power to be within two standard deviations for each separate frequency bin and take the median value
    ref_list = np.zeros((int(np.max(fs))))
    for ref_area in ref_areas:
        for refIdx in range(ref_area[0], ref_area[1]):
            ref_list[refIdx] = np.median(finnpy.cleansing.outlier_removal.run(pow_oi[:, refIdx], pow_oi[:, refIdx]))
            
    pow_oi = np.log10(pow_oi)
    ref_list[ref_list == 0] = np.nan  # Avoid divide by zero warning/error and ignore 'bad' values.
    # These may occure accidentially and are no reason for concern if they appear sparsely. 
    ref_list = np.log10(ref_list)
    
    # Compare the median power of non-outlier (harshly filtered) channels vs the power of each channel and determine distance
    diff = np.zeros((len(data), int(np.max(fs)))) * np.nan
    for ch_idx in range(len(data)):
        for ref_area in ref_areas:
            for ref_idx in range(ref_area[0], ref_area[1]):
                diff[ch_idx, ref_idx] = pow_oi[ch_idx, ref_idx] / ref_list[ref_idx]
         
    diff = np.nanmean(diff, axis = 1)
    
    corr_diff = finnpy.cleansing.outlier_removal.run(diff, diff, broadness)
    
    corr_diff_mean = np.mean(corr_diff)

    corr_diff_var = np.sqrt(np.var(corr_diff))
     
    min_ref = corr_diff_mean - broadness * corr_diff_var
    max_ref = corr_diff_mean + broadness * corr_diff_var
    
    valid_list   = np.argwhere(np.logical_and(diff >= min_ref, diff <= max_ref)).squeeze(1)  # noqa: E221
    invalid_list = np.argwhere(np.logical_or(diff < min_ref, diff > max_ref)).squeeze(1)
    
    valid_list = np.asarray(valid_list)
    invalid_list = np.asarray(invalid_list)
    
    if (len(invalid_list) > (len(data) * 0.2)):        
        warnings.warn("Way too many noisy eeg channels")
        
    idenfity_faulty_visual_inspection_lock.acquire()
    if (visual_inspection):
        (valid_list, invalid_list) = _manual_check(len(data), ch_names, diff, min_ref, max_ref, valid_list, invalid_list)
    idenfity_faulty_visual_inspection_lock.release()
        
    return (valid_list.tolist(), invalid_list.tolist(), np.asarray(score_list))

def _manual_check(ch_cnt, ch_names, score, min_ref, max_ref, valid_list, invalid_list):
    """
    Visualize the z-score of each channel and annotates all channels.
    
    Herein, bad channel selection may be manually adjusted. Internally parallelized to speed up the drawing process.
    
    Parameters
    ----------
    ch_cnt : int
             Number of channels.
    ch_names : list
               Number of channels.
    score : list
            Z-score of each channel.
    min_ref : float
              Minimum value for a valid z-score.
    max_ref : float
              Maximum value for a valid z-score.
    valid_list : list
                 List of valid channels.
    invalid_list : list
                   List of invalid channels.
    
    Returns
    -------
    tuple of (valid_list, invalid_list)
        - valid_list : list
                       List of valid channels.
        - invalid_list : list
                         List of invalid channels, may be different from the input due to performed manual adjustments.
    """
    shared_valid_list = multiprocessing.Array('i', ch_cnt)
    
    # Need to run as a subproces to enable the starting of multiple Qapplications
    vis_sub_process = multiprocessing.Process(target = _manual_check_mp, args = [ch_cnt, ch_names, score, min_ref, max_ref, valid_list, invalid_list, shared_valid_list])
    vis_sub_process.start()
    vis_sub_process.join()
    
    invalid_list = np.argwhere(np.asarray(shared_valid_list) == 0).squeeze(1)
    valid_list = np.argwhere(np.asarray(shared_valid_list) == 1).squeeze(1)
    
    return (valid_list, invalid_list)

def _manual_check_mp(ch_cnt, ch_names, score, min_ref, max_ref, valid_list, invalid_list, shared_valid_list):
    """
    
    Parallized part of _manual_check.
    
    Parameters
    ----------
    ch_cnt : int
             Number of channels.
    ch_names : list
               Number of channels.
    score: list
           Z-score of each channel.
    min_ref : float
              Minimum value for a valid z-score.
    max_ref : float
              Maximum value for a valid z-score.
    valid_list : list
                 List of valid channels.
    invalid_list : list
                   List of invalid channels.
    shared_valid_list : list
                        Used to return information from the sub-process after its termination.
    
    Returns
    -------
    tuple of (valid_list, invalid_list).
        - valid_list : list
                       List of valid channels
        - invalid_list : list
                         List of invalid channels, may be different from the input due to performed manual adjustments.
    
    """   
    app = PyQt6.QtWidgets.QApplication([])  # pylint: disable=c-extension-no-member
    
    if (type(valid_list) is np.ndarray):
        valid_list = valid_list.tolist()
    if (type(invalid_list) is np.ndarray):
        invalid_list = invalid_list.tolist()
    
    win = _Qt_win(ch_cnt, ch_names, min_ref, max_ref, score, valid_list, invalid_list)
    
    app.exec()
    
    app.quit()
    
    invalid_list = win.invalid_list
    valid_list = win.valid_list
    
    del win
    
    valid_list = np.asarray(valid_list)
    invalid_list = np.asarray(invalid_list)
    
    for ch_idx in valid_list:
        shared_valid_list[ch_idx] = 1
    for ch_idx in invalid_list:
        shared_valid_list[ch_idx] = 0
    
    return (valid_list, invalid_list)

class _Qt_win(PyQt6.QtWidgets.QWidget):  # pylint: disable=c-extension-no-member
    """
    Window visualizing the z-score distribution of the provided channels.
     
    Parameters
    ----------
    ch_cnt : int
             Number of channels.
    ch_names : list
               Number of channels.
    min_ref : float
              Minimum value for a valid z-score.
    max_ref : float
              Maximum value for a valid z-score.
    diff : list, len(ch_cnt)
           Power level differences to average power.
    valid_list : list
                 List of valid channels.
    invalid_list : list
                  List of invalid channels.
    maxX : int
           Maximum x-value.
           
    Attributes
    ----------
    valid_list : list
                 List of valid channels.
    invalid_list : list
                   List of invalid channels.
    canvas : matplotlib.backends.backend_qtagg.FigureCanvasQTAgg
             Canvas used for scatterplot visualization.
    fig : matplotlib.figure.Figure
          Parent container of the scatterplot canvas.
    ch_cnt : int
             Number of channels.
    min_ref : float
              Minimum power threshold.
    max_ref : float
              Maximum power threshold.
    diff : list
           Differences in power level from the average.
    ch_names : list
               Channel names.
    """
    
    valid_list: list = list()
    invalid_list: list = list()
    canvas: matplotlib.backends.backend_qtagg.FigureCanvasQTAgg = None
    fig: matplotlib.figure.Figure = None
    
    ch_cnt: int = None
    min_ref: float = None
    max_ref: float = None
    diff: list = None
    ch_names: list = None
        
    def __init__(self, ch_cnt, ch_names, min_ref, max_ref, diff, valid_list, invalid_list, maxX = 8):
        super().__init__()
        
        self.ch_cnt = ch_cnt
        self.min_ref = min_ref
        self.max_ref = max_ref
        self.diff = diff
        self.ch_names = ch_names
        
        self.layout = PyQt6.QtWidgets.QGridLayout(self)  # pylint: disable=no-member, c-extension-no-member
        self.valid_list = list(); self.test = list()
        for ch_idx in valid_list:
            self.valid_list.append(ch_idx)
            self.test.append(ch_idx)
        self.invalid_list = invalid_list
        self.setWindowTitle("Manual faulty channel inspector")
        
        self.fig = matplotlib.figure.Figure()
        self.fig.subplots(1, 1)
        self.canvas = matplotlib.backends.backend_qtagg.FigureCanvasQTAgg(self.fig)
        self.init_canvas()
        self.layout.addWidget(self.canvas, 0, 0, 1, maxX + 2)
        self.canvas.setSizePolicy(PyQt6.QtWidgets.QSizePolicy(PyQt6.QtWidgets.QSizePolicy.Policy.Ignored, PyQt6.QtWidgets.QSizePolicy.Policy.Expanding))  # pylint: disable=no-member, c-extension-no-member
        self.canvas.setMinimumHeight(300)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        
        xPos = 1
        maxXPos = list()
        yPos = 1
        self.buttonList = list()
        for ch_idx in range(ch_cnt):
            if (ch_idx in invalid_list):
                button = PyQt6.QtWidgets.QPushButton(ch_names[ch_idx] + ": Invalid")  # pylint: disable=no-member, c-extension-no-member
                button.setStyleSheet("Background-color:red;")
            else:
                button = PyQt6.QtWidgets.QPushButton(ch_names[ch_idx] + ": Valid")  # pylint: disable=no-member, c-extension-no-member
                button.setStyleSheet("Background-color:green;")
            self.buttonList.append(button)
            button.setFixedWidth(120)
            button.clicked.connect(functools.partial(self.change_state, ch_idx))

            self.layout.addWidget(button, yPos, xPos, 1, 1)
            maxXPos.append(xPos)
            xPos += 1
            if (xPos == (maxX + 1)):
                xPos = 1
                yPos += 1
        maxXPos = max(maxXPos)
        
        nextWidth = button.maximumWidth()
        
        button = PyQt6.QtWidgets.QPushButton("Close")  # pylint: disable=no-member, c-extension-no-member
        button.clicked.connect(self.clicked_close)
        button.setMaximumWidth(np.max((nextWidth, 120)))

        if (maxXPos % 2 == 0):
            self.layout.addWidget(button, yPos + 1, maxXPos // 2, 1, 2)
        else:
            if (ch_cnt == 1):
                self.layout.addWidget(button, yPos + 1, (maxXPos + 1) // 2, 1, 1)
            else:
                self.layout.addWidget(button, yPos + 1, (maxXPos + 1) // 2 - 1, 1, 3)

        self.layout.setAlignment(button, PyQt6.QtCore.Qt.AlignmentFlag.AlignHCenter)  # pylint: disable=no-member, c-extension-no-member

        buffer1 = PyQt6.QtWidgets.QWidget()  # pylint: disable=no-member, c-extension-no-member
        buffer2 = PyQt6.QtWidgets.QWidget()  # pylint: disable=no-member, c-extension-no-member
        buffer1.setSizePolicy(PyQt6.QtWidgets.QSizePolicy(PyQt6.QtWidgets.QSizePolicy.Policy.Expanding, PyQt6.QtWidgets.QSizePolicy.Policy.Ignored))  # pylint: disable=no-member, c-extension-no-member
        buffer2.setSizePolicy(PyQt6.QtWidgets.QSizePolicy(PyQt6.QtWidgets.QSizePolicy.Policy.Expanding, PyQt6.QtWidgets.QSizePolicy.Policy.Ignored))  # pylint: disable=no-member, c-extension-no-member
        self.layout.addWidget(buffer1, 1, 0,           yPos + 1, 1)  # noqa: E241
        self.layout.addWidget(buffer2, 1, maxXPos + 1, yPos + 1, 1)
        
        self.show()
    
    def init_canvas(self):
        """Create the canvas for data visualization."""
        self.fig.axes[0].cla()  # pylint: disable=unsubscriptable-object
        loc_valid_list = np.asarray(self.valid_list)
        self.fig.axes[0].scatter([-1, self.ch_cnt + 1], [1, 1], color = "white")  # pylint: disable=unsubscriptable-object
        if (len(loc_valid_list) > 0):
            self.fig.axes[0].scatter(loc_valid_list, np.asarray(self.diff)[loc_valid_list], color = "green")  # pylint: disable=unsubscriptable-object
        loc_invalid_list = np.asarray(self.invalid_list)
        if (len(loc_invalid_list) > 0):
            self.fig.axes[0].scatter(loc_invalid_list, np.asarray(self.diff)[loc_invalid_list], color = "red")  # pylint: disable=unsubscriptable-object
        self.fig.axes[0].hlines([self.min_ref, self.max_ref], 0, self.ch_cnt)  # pylint: disable=unsubscriptable-object
        self.fig.axes[0].get_xaxis().set_ticks([])  # pylint: disable=unsubscriptable-object
        
        for ch_idx in range(self.ch_cnt):
            self.fig.axes[0].annotate(self.ch_names[ch_idx], [ch_idx, self.diff[ch_idx]], zorder = 1000)  # pylint: disable=unsubscriptable-object
        
        self.fig.canvas.draw()
    
    def update_canvas(self, ch_idx):
        """
        Populate the canvas with data points.
        
        Parameters
        ----------
        ch_idx: int
                Channel to be toggled.
        """
        if (ch_idx in self.valid_list):
            self.fig.axes[0].scatter(ch_idx, np.asarray(self.diff)[ch_idx], color = "green")  # pylint: disable=unsubscriptable-object
        else:
            self.fig.axes[0].scatter(ch_idx, np.asarray(self.diff)[ch_idx], color = "red")  # pylint: disable=unsubscriptable-object
        
        self.fig.canvas.draw()
    
    def change_state(self, ch_idx):
        """
        Toggles a data point from valid (green) to invalid (red) and back.
        
        Parameters
        ----------
        ch_idx: int
                Channel to be toggled.
        """
        if (ch_idx in self.invalid_list):
            self.invalid_list.remove(ch_idx)
            self.valid_list.append(ch_idx) 
            self.buttonList[ch_idx].setText(self.ch_names[ch_idx] + ": Valid")
            self.buttonList[ch_idx].setStyleSheet("Background-color:green;")
        else:
            self.valid_list.remove(ch_idx) 
            self.invalid_list.append(ch_idx)
            self.buttonList[ch_idx].setText(self.ch_names[ch_idx] + ": Invalid")   
            self.buttonList[ch_idx].setStyleSheet("Background-color:red;")
        self.update_canvas(ch_idx)
        
    def on_click(self, event):
        """Catches a mouse click event to toggle a data point from valid (green) to invalid (red) and back."""
        distPts = list()
        xVar = np.abs(self.canvas.figure.axes[0].get_xlim()[0] - self.canvas.figure.axes[0].get_xlim()[1])
        yVar = np.abs(self.canvas.figure.axes[0].get_ylim()[0] - self.canvas.figure.axes[0].get_ylim()[1])
         
        for ch_idx in range(self.ch_cnt):
            distPts.append(np.sqrt(np.power((ch_idx - event.xdata) / xVar, 2) + np.power((self.diff[ch_idx] - event.ydata) / yVar, 2)))

        closestPt = np.argmin(distPts)
        self.change_state(closestPt)
        
    def clicked_close(self):
        """Close the window when order to do so."""
        self.close()
