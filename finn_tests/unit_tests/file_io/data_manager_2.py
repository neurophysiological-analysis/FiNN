'''
Created on Dec 30, 2020

@author: voodoocode
'''

import finn.file_io.data_manager

data = [[[1, 2], [3, 4]], [[1, 2], [3, 4], ["a", "b", "c"], ["x", "x"]]]
#data = [[1], {'a' : [2], 2 : [3]}]
#data = 2

finn.file_io.data_manager.save(data, "/home/voodoocode/Downloads/test_file", max_depth = 2)

data2 = finn.file_io.data_manager.load("/home/voodoocode/Downloads/test_file")

print(data == data2)



