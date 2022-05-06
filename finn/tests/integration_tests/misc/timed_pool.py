'''
Created on May 5, 2022

@author: voodoocode
'''

import unittest
import finn.misc.timed_pool as tp

import multiprocessing

class test_filters(unittest.TestCase):

    lock = multiprocessing.Lock()
    cntr = multiprocessing.Value('f', lock=True)

    def foo(self):
        self.cntr.value = self.cntr.value + 1

    def test_filters(self):
        tp.run(max_child_proc_cnt = 4, func = self.foo, args = [() for _ in range(8)], delete_data = False)
        
        assert(self.cntr.value == 8)
        
if __name__ == '__main__':
    unittest.main()




