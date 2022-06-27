'''
Created on Jun 2, 2020

This module implements a function to identify bad channels based on increased/decreased power within a channel compared to the majority of other channels. 

:author: voodoocode
'''

import numpy as np

import PyQt5.QtWidgets
import PyQt5.Qt
import functools
import matplotlib.backends
import scipy.signal
import multiprocessing
import warnings

import finnpy.cleansing.outlier_removal


idenfity_faulty_visual_inspection_lock = multiprocessing.Lock()

def run(data, ch_names, fs, ref_areas = [[105, 120], [135, 145], [155, 195]], broadness = 3, visual_inspection = True):
    """
    Identifies which channels have substantially more or less power in the frequency ranges defined by ref_areas. Channels whose power is more different than *broadness* (default: 3) standard deviations will be primed as faulty channels. In case visual inspection is *active* (default: True), the automatic results can be further refined via manual selection.
    
    Attention: Function is parallelized. Sensitive parts are placed within a locked area to avoid unexpected behaviour. 
    
    :param data: Input data in the format channel x samples.
    :param ch_names: Names of the channels. Used for visualization purposes only. Order has to match the channel order of data.
    :param fs: List of sampling frequencies for each channel.
    :param ref_areas: Spectral reference areas used for power estimation. Only power within these ranges is defined. It is recommended to choose ranges, which are not part of any evaluation to decrease the chances of pre-processing induced biases.
    :param broadness: Number of standard deviations threshold by which channels are automatically categorized as faulty. In case visual inspection is enabled (recommended) this only results in priming the channels.
    :param visual_inspection: Toggles visual inspection on and off.
    
    :return: (valid_list, invalid_list, score). List of valid channels, invalid channels and their respective z-scores.
    """
    
    score_list = list()
    pow_oi = list()
    for ch_idx in range(0, len(data)):
        (_, power) = scipy.signal.welch(data[ch_idx], fs = fs[ch_idx], nfft = int(fs[ch_idx]), nperseg = int(fs[ch_idx]), noverlap = fs[ch_idx]//2)
        pow_oi.append(power)
    pow_oi = np.asarray(pow_oi)
    
    #Filter reference power to be within two standard deviations for each separate frequency bin and take the median value
    ref_list = np.zeros((int(np.max(fs))))
    for ref_area in ref_areas:
        for refIdx in range(ref_area[0], ref_area[1]):
            ref_list[refIdx] = np.median(finnpy.cleansing.outlier_removal.run(pow_oi[:, refIdx], pow_oi[:, refIdx]))
            
    pow_oi = np.log10(pow_oi)
    ref_list[ref_list == 0] = np.nan #Avoid divide by zero warning/error and ignore 'bad' values.
    #These may occure accidentially and are no reason for concern if they appear sparsely. 
    ref_list = np.log10(ref_list)
    
    #Compare the median power of non-outlier (harshly filtered) channels vs the power of each channel and determine distance
    diff = np.zeros((len(data), int(np.max(fs)))) * np.nan
    for ch_idx in range(len(data)):
        for ref_area in ref_areas:
            for ref_idx in range(ref_area[0], ref_area[1]):
                diff[ch_idx, ref_idx] = pow_oi[ch_idx, ref_idx]/ref_list[ref_idx]
         
    diff = np.nanmean(diff, axis = 1)
    
    corr_diff = finnpy.cleansing.outlier_removal.run(diff, diff, broadness)
    
    corr_diff_mean = np.mean(corr_diff)

    corr_diff_var = np.sqrt(np.var(corr_diff))
     
    min_ref = corr_diff_mean - broadness * corr_diff_var
    max_ref = corr_diff_mean + broadness * corr_diff_var
    
    valid_list   = np.argwhere(np.logical_and(diff >= min_ref, diff <= max_ref)).squeeze(1)
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
    Visualizes the z-score of each channel and annotates all channels. Afterwards bad channel selection may be manually adjusted. Internally parallelized to speed up the drawing process.
    
    :param ch_cnt: Number of channels.
    :param ch_names: Number of channels.
    :param score: Z-score of each channel.
    :param min_ref: Minimum value for a valid z-score.
    :param max_ref: Maximum value for a valid z-score.
    :param valid_list: List of valid channels.
    :param invalid_list: List of invalid channels.
    
    :return: (valid_list, invalid_list). List of valid and invalid channels, may be different from the input due to performed manual adjustments.
    """
    shared_valid_list = multiprocessing.Array('i', ch_cnt)
    
    #Need to run as a subproces to enable the starting of multiple Qapplications
    vis_sub_process = multiprocessing.Process(target = _manual_check_mp, args = [ch_cnt, ch_names, score, min_ref, max_ref, valid_list, invalid_list, shared_valid_list])
    vis_sub_process.start()
    vis_sub_process.join()
    
    invalid_list = np.argwhere(np.asarray(shared_valid_list) == 0).squeeze(1)
    valid_list = np.argwhere(np.asarray(shared_valid_list) == 1).squeeze(1)
    
    return(valid_list, invalid_list)

def _manual_check_mp(ch_cnt, ch_names, score, min_ref, max_ref, valid_list, invalid_list, shared_valid_list):
    """
    
    Parallized part of _manual_check.
    
    :param ch_cnt: Number of channels.
    :param ch_names: Number of channels.
    :param score: Z-score of each channel.
    :param min_ref: Minimum value for a valid z-score.
    :param max_ref: Maximum value for a valid z-score.
    :param valid_list: List of valid channels.
    :param invalid_list: List of invalid channels.
    :param shared_valid_list: Used to return information from the sub-process after its termination.
    
    :return: (valid_list, invalid_list). List of valid and invalid channels, may be different from the input due to performed manual adjustments.
    
    """   
    app = PyQt5.QtWidgets.QApplication([])
    
    if (type(valid_list) == np.ndarray):
        valid_list = valid_list.tolist()
    if (type(invalid_list) == np.ndarray):
        invalid_list = invalid_list.tolist()
    
    win = _Qt_win(app, ch_cnt, ch_names, min_ref, max_ref, score, valid_list, invalid_list)
    
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

class _Qt_win(PyQt5.QtWidgets.QWidget):
    """

    Window visualizing the z-score distribution of the provided channels.

    """
    
    button_list = list()
    valid_list = list()
    invalid_list = list()
    canvas = None
    fig = None
    
    ch_cnt = None
    min_ref = None
    max_ref = None
    diff = None
    ch_names = None
        
    def __init__(self, app, ch_cnt, ch_names, min_ref, max_ref, diff, valid_list, invalid_list, maxX = 8):
        """
        
        Constructs the window.
        
        :param app: Reference to the primary Qapplication.
        :param ch_cnt: Number of channels.
        :param ch_names: Number of channels.
        :param score: Z-score of each channel.
        :param min_ref: Minimum value for a valid z-score.
        :param max_ref: Maximum value for a valid z-score.
        :param valid_list: List of valid channels.
        :param invalid_list: List of invalid channels.
        
        """
        super().__init__()
        
        self.ch_cnt = ch_cnt
        self.min_ref = min_ref
        self.max_ref = max_ref
        self.diff = diff
        self.ch_names = ch_names
        
        self.setGeometry(PyQt5.QtWidgets.QStyle.alignedRect(PyQt5.Qt.Qt.LeftToRight, PyQt5.Qt.Qt.AlignCenter, self.size(), app.desktop().availableGeometry()))
        self.layout = PyQt5.QtWidgets.QGridLayout(self)
        self.valid_list = list(); self.test = list()
        for ch_idx in valid_list:
            self.valid_list.append(ch_idx)
            self.test.append(ch_idx)
        self.invalid_list = invalid_list
        self.setWindowTitle("Manual faulty channel inspector")
        
        self.fig = matplotlib.figure.Figure()
        self.fig.subplots(1, 1)
        self.canvas = matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg(self.fig)
        self.init_canvas()
        self.layout.addWidget(self.canvas, 0, 0, 1, maxX+2)
        self.canvas.setSizePolicy(PyQt5.Qt.QSizePolicy(PyQt5.Qt.QSizePolicy.Ignored, PyQt5.Qt.QSizePolicy.Expanding))
        self.canvas.setMinimumHeight(300)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        
        xPos = 1
        maxXPos = list()
        yPos = 1
        fm = PyQt5.QtGui.QFontMetrics(PyQt5.QtWidgets.QPushButton().font())
        self.buttonList = list()
        for ch_idx in range(ch_cnt):
            if (ch_idx in invalid_list):
                button = PyQt5.QtWidgets.QPushButton(ch_names[ch_idx] + ": Invalid")
                button.setStyleSheet("Background-color:red;")
            else:
                button = PyQt5.QtWidgets.QPushButton(ch_names[ch_idx] + ": Valid")
                button.setStyleSheet("Background-color:green;")
            self.buttonList.append(button)
            button.setFixedWidth(fm.width("CCCC: Invalid"))
            button.clicked.connect(functools.partial(self.change_state, ch_idx))

            self.layout.addWidget(button,   yPos, xPos, 1, 1)
            maxXPos.append(xPos)
            xPos += 1
            if (xPos == (maxX + 1)):
                xPos = 1
                yPos += 1
        maxXPos = max(maxXPos)
        
        nextWidth = button.maximumWidth()
        
        button = PyQt5.QtWidgets.QPushButton("Close")
        button.clicked.connect(self.clicked_close)
        button.setMaximumWidth(np.max((nextWidth, fm.width("Close"))))

        if (maxXPos % 2 == 0):
            self.layout.addWidget(button, yPos + 1, maxXPos//2, 1, 2)
        else:
            if (ch_cnt == 1):
                self.layout.addWidget(button, yPos + 1, (maxXPos + 1)//2, 1, 1)
            else:
                self.layout.addWidget(button, yPos + 1, (maxXPos + 1)//2 - 1, 1, 3)

        self.layout.setAlignment(button, PyQt5.Qt.Qt.AlignHCenter)

        buffer1 = PyQt5.QtWidgets.QWidget()
        buffer2 = PyQt5.QtWidgets.QWidget()
        buffer1.setSizePolicy(PyQt5.Qt.QSizePolicy(PyQt5.Qt.QSizePolicy.Expanding, PyQt5.Qt.QSizePolicy.Ignored))
        buffer2.setSizePolicy(PyQt5.Qt.QSizePolicy(PyQt5.Qt.QSizePolicy.Expanding, PyQt5.Qt.QSizePolicy.Ignored))
        self.layout.addWidget(buffer1, 1, 0,           yPos + 1, 1)
        self.layout.addWidget(buffer2, 1, maxXPos + 1, yPos + 1, 1)
        
        
        self.show()
    
    def init_canvas(self):
        """
        
        Creates the canvas for data visualization.
        
        """
        self.fig.axes[0].cla()
        loc_valid_list = np.asarray(self.valid_list)
        self.fig.axes[0].scatter([-1, self.ch_cnt + 1], [1, 1], color = "white")
        if (len(loc_valid_list) > 0):
            self.fig.axes[0].scatter(loc_valid_list, np.asarray(self.diff)[loc_valid_list], color = "green")
        loc_invalid_list = np.asarray(self.invalid_list)
        if (len(loc_invalid_list) > 0):
            self.fig.axes[0].scatter(loc_invalid_list, np.asarray(self.diff)[loc_invalid_list], color = "red")
        self.fig.axes[0].hlines([self.min_ref, self.max_ref], 0, self.ch_cnt)
        self.fig.axes[0].get_xaxis().set_ticks([])
        
        for ch_idx in range(self.ch_cnt):
            self.fig.axes[0].annotate(self.ch_names[ch_idx], [ch_idx, self.diff[ch_idx]], zorder = 1000)
        
        self.fig.canvas.draw()
    
    def update_canvas(self, ch_idx):
        """
        
        Populates the canvas with data points.
        
        :param ch_idx: channel to be toggled.
        
        """
        if (ch_idx in self.valid_list):
            self.fig.axes[0].scatter(ch_idx, np.asarray(self.diff)[ch_idx], color = "green")
        else:
            self.fig.axes[0].scatter(ch_idx, np.asarray(self.diff)[ch_idx], color = "red")
        
        self.fig.canvas.draw()
    
    def change_state(self, ch_idx):
        """
        
        Toggles a data point from valid (green) to invalid (red) and back.
        
        :param ch_idx: channel to be toggled.
        
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
        """
        
        Catches a mouse click event to toggle a data point from valid (green) to invalid (red) and back.
        
        """
        
        distPts = list()
        xVar = np.abs(self.canvas.figure.axes[0].get_xlim()[0] - self.canvas.figure.axes[0].get_xlim()[1])
        yVar = np.abs(self.canvas.figure.axes[0].get_ylim()[0] - self.canvas.figure.axes[0].get_ylim()[1])
         
        for ch_idx in range(self.ch_cnt):
            distPts.append(np.sqrt(np.power((ch_idx - event.xdata)/xVar, 2) + np.power((self.diff[ch_idx] - event.ydata)/yVar, 2)))

        closestPt = np.argmin(distPts)
        self.change_state(closestPt)
        
    def key_pressed(self, event):
        """
        
        Additional functionality in response to pressed keys may be added here.
        
        """
        
        pass
        #------------------------ if (event.key() == PyQt5.QtCore.Qt.Key_Space):
            #---------------------------------------------- print("Hello World")
            #---------------------------------------------- #self.clickedClose()
        
    def clicked_close(self):
        """
        
        Closes the window when order to do so.
        
        """
        
        self.close()











