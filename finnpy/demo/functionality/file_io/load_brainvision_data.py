'''
Created on May 6, 2022

@author: voodoocode
'''

import matplotlib.pyplot as plt

import numpy as np

import finnpy.data.paths as paths
import finnpy.file_io.load_brainvision_data as lbvd

data = lbvd.run(paths.fct_brainvision_data).get_data()  
mrks = lbvd.read_marker(paths.fct_brainvision_data[:-5] + ".vmrk")

offset = 130000

plt.plot(data[0, offset:])

min_val = np.min(data[0, offset:])
max_val = np.max(data[0, offset:])

for mrk in mrks:
    # Markers may be identified by their type
    if (mrk["type"] == ""):
        continue
    if (mrk["type"] == "s"):
        plt.plot([mrk["idx"] - offset, mrk["idx"] - offset], [min_val, max_val], color = "green")
    if (mrk["type"] == "x"):
        plt.plot([mrk["idx"] - offset, mrk["idx"] - offset], [min_val, max_val], color = "blue")

plt.show(block = True)
