'''
Created on May 5, 2022

@author: voodoocode
'''

import unittest

import finn.data.paths as paths
import finn.file_io.load_brainvision_data as lbvd

class test_lbvd(unittest.TestCase):
    
    def test_lbvd(self):
        data = lbvd.run(paths.fct_brainvision_data, verbose = "ERROR")
        
        assert(data.get_data().shape == (68, 479200))
        
        mrk = lbvd.read_marker(paths.fct_brainvision_data[:-5] + ".vmrk")
        
        assert(mrk[1]["type"] == "s" and mrk[1]["idx"] == 144401)

if __name__ == '__main__':
    unittest.main()


