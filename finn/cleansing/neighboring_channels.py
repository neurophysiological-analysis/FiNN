'''
Created on May 27, 2019

@author: voodoocode

This module contains a list of neighboring EEG channels. This file is to be moved into an external *.csv file in the future.

'''

neighbor_channels = {
    "O1" : ["PO7", "PO3", "Oz"],
    "Oz" : ["O1", "POz", "O2"],
    "O2" : ["Oz", "PO4", "PO8"],
    
    "PO9" : ["P9", "P7", "PO7"],
    "PO7" : ["PO9", "P7", "P5", "PO3", "O1"],
    "PO3" : ["PO7", "P3", "POz", "O1"],
    "POz" : ["PO3", "POz", "PO4", "Oz"],
    "PO4" : ["POz", "P4", "PO8", "O2"],
    "PO8" : ["O2", "PO4", "P6", "P8", "PO10"],
    "PO10": ["PO8", "P8", "P10"],
    
    "P9" : ["TP9", "P7", "PO9"], 
    "P7" : ["P9", "TP7", "P5", "PO7"], 
    "P5" : ["P7", "CP5", "P3", "PO7"],
    "P3" : ["P5", "CP3", "P1", "PO3"],
    "P1" : ["P3", "CP1", "Pz", "PO3"],
    "Pz" : ["P1", "CPz", "P2", "POz"],
    "P2" : ["Pz", "CP2", "P4", "PO4"],
    "P4" : ["P2", "CP4", "P6", "PO4"],
    "P6" : ["P4", "CP6", "P8", "PO8"],
    "P8" : ["P6", "TP8", "P10", "PO8"],
    "P10" : ["P8", "TP10", "PO10"],
    
    "TP9" : ["T9", "TP7", "P9"], 
    "TP7" : ["TP9", "T7", "CP5", "P7"], 
    "CP5" : ["TP7", "C5", "CP3", "P5"],
    "CP3" : ["CP5", "C3", "CP1", "P3"],
    "CP1" : ["CP3", "C1", "CPz", "P1"],
    "CPz" : ["CP1", "Cz", "CP2", "Pz"],
    "CP2" : ["CPz", "C2", "CP4", "P2"],
    "CP4" : ["CP2", "C4", "CP6", "P4"],
    "CP6" : ["CP4", "C6", "TP8", "P6"],
    "TP8" : ["CP6", "T8", "TP10", "P8"],
    "TP10" : ["TP8", "T10", "P10"],
    
    "T9" : ["FT9", "T7", "TP9"], 
    "T7" : ["T9", "FT7", "C5", "TP7"], 
    "C5" : ["T7", "FC5", "C3", "CP5"],
    "C3" : ["C5", "FC3", "C1", "CP3"],
    "C1" : ["C3", "FC1", "Cz", "CP1"],
    "Cz" : ["C1", "FCz", "C2", "CPz"],
    "C2" : ["Cz", "FC2", "C4", "CP2"],
    "C4" : ["C2", "FC4", "C6", "CP4"],
    "C6" : ["C4", "FC6", "T8", "CP6"],
    "T8" : ["C6", "FT8", "T10", "TP8"],
    "T10" : ["T8", "T10", "TP9"],
    
    "FT9" : ["F9", "FT7", "T9"], 
    "FT7" : ["FT9", "F7", "FC5", "T7"], 
    "FC5" : ["FT7", "F5", "FC3", "C5"],
    "FC3" : ["FC5", "F3", "FC1", "C3"],
    "FC1" : ["FC3", "F1", "FCz", "C1"],
    "FCz" : ["FC1", "Fz", "FC2", "Cz"],
    "FC2" : ["FCz", "F2", "FC4", "C2"],
    "FC4" : ["FC2", "F4", "FC6", "C4"],
    "FC6" : ["FC4", "F6", "FT8", "C6"],
    "FT8" : ["FC6", "F8", "FT10", "T8"],
    "FT10" : ["FT8", "F10", "T10"],
    
    "F9" : ["AF9", "F7", "FT9"], 
    "F7" : ["F9", "AF7", "F5", "FT7"], 
    "F5" : ["F7", "AF7", "F3", "FC5"],
    "F3" : ["F5", "AF3", "F1", "FC3"],
    "F1" : ["F3", "AF3", "Fz", "FC1"],
    "Fz" : ["F1", "AFz", "F2", "FCz"],
    "F2" : ["Fz", "AF4", "F4", "FC2"],
    "F4" : ["F2", "AF4", "F6", "FC4"],
    "F6" : ["F4", "AF8", "F8", "FC6"],
    "F8" : ["F6", "AF8", "F10", "FT8"],
    "F10": ["F8", "FT10", "AF10"],
    
    "AF9" : ["F9", "AF7", "F7"],
    "AF7" : ["F7", "Fp1", "AF3", "F5"],
    "AF3" : ["AF7", "Fp1", "AFz", "F3"],
    "AFz" : ["AF3", "Fpz", "AF4", "Fz"],
    "AF4" : ["AFz", "Fp2", "AF8", "F4"],
    "AF8" : ["AF4", "Fp2", "F8", "F6"],
    "AF10": ["AF8", "F10", "F8"],
    
    "Fp1" : ["AF7", "Fpz", "AF3"],
    "Fpz" : ["Fp1", "Fp2", "AFz"],
    "Fp2" : ["Fpz", "AF8", "AF4"],
    }
