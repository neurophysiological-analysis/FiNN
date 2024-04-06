
Timed Pool
============


.. currentmodule:: misc.timed_pool
.. autofunction:: run

The following code example shows how to use the multiprocessing pool.

.. code:: python
   
   import finn.misc.timed_pool as tp
   
   NUMBER_OF_PROCESSES = 10
    
   def bar(param1, param2):
   	return foo(param1, param2)
    
   def main():
   
      output = tp.run(NUMBER_OF_PROCESSES, bar, [(arg1[idx], arg2[idx],) for idx in range(100)], max_time = 100, delete_data = True)
    
   main()


