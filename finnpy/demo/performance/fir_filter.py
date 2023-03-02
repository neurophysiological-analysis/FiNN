'''
Created on Aug 5, 2020

@author: voodoocode
'''

import finnpy.filters.frequency as ff
import mne.filter as mf
import numpy as np

import matplotlib.pyplot as plt

import time
import os

import finnpy.data.paths as paths

def main(mode = "fast", save_results = False, overwrite = True):
    repetitions = [1, 2, 10, 20, 120, 240, 720]
    
    if (overwrite == True or os.path.exists("filter_score.npy") == False):
        score = list()
        for repetition in repetitions:
            path = paths.per_fir_data
            data = np.load(path)[3, :]
        
            data = np.repeat(data, repetition)
            fs = 5500
            data = np.pad(data, 150000)
            
            start = time.time_ns()
            ff.fir(data, 15, 25, 0.1, fs, 10e-5, 10e-7, fs, "zero", mode)
            end = time.time_ns()
            score_ff = (end - start)/1000/1000
            print("FiNN filter: %f" % (score_ff,), end = " | ")
            start = time.time_ns()
            mf.filter_data(data, fs, 15, 25, None, 330000, 0.1, 0.1, 1, "fir", None, False, "zero", "hamming", "firwin2", pad = "constant", verbose = False)
            end = time.time_ns()
            score_mf = (end - start)/1000/1000
            print("MNE filter: %f | length of (padded) data in s: %i | sampling frequency: %i" % (score_mf, int(len(data)/fs), fs))
            
            score.append([score_ff, score_mf])
        score = np.asarray(score)
        if (save_results == True):
            np.save("filter_score.npy", score)
    else:
        score = np.load("filter_score.npy")
    
    plt.bar(np.arange(len(score[:, 0])), score[:, 1]/score[:, 0])
    plt.xticks(range(0, len(score[:, 0])), ["30s", "1m", "5m", "10m", "1h", "2h", "6h"])
    plt.show(block = True)

main()












