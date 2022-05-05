'''
Created on Jan 30, 2019

Provides a memory and cpu sensitive version of the multiprocessing tool. 

:author: voodoocode
'''

import multiprocessing
import time
import numpy as np

def _manage_sub_process(func, args, child_pipe, return_data_lock):
    """
    Commands a sub-process to execute *func* with the arguments provided in *args*.
    
    :param func: Function to be executed.
    :param args: Arguments for the aformentioned function.
    :param child_pipe: Pipe used to return the result.
    :param return_data_lock: Lock to have only one sub-process return data at a time (prevents excessive memory usage in case of large returns for each sub-process.
    
    """
    result = func(*args)
    
    return_data_lock.acquire()
    child_pipe.send(result)
    return_data_lock.release()
    child_pipe.recv()
     
    child_pipe.close()
    
def _launch_child_process(sub_processes, func, args, max_time, curr_job_idx, return_data_lock):
    """
    
    Starts a child-process to execute work.
    
    :param sub_processes: List of all child-processes.
    :param func: Function to be executed.
    :param args: Arguments for the aformentioned function.
    :param max_time: Maximum time after which the sub-process is terminated. 
    :param curr_job_idx: Number of this sub-processe's job.
    :param return_data_lock: Lock to have only one sub-process return data at a time (prevents excessive memory usage in case of large returns for each sub-process.
    
    """
    
    (parent_pipe, child_pipe) = multiprocessing.Pipe(True)
    sub_process = multiprocessing.Process(target = _manage_sub_process, args = (func, args[curr_job_idx], child_pipe, return_data_lock))
    sub_processes.append((sub_process, parent_pipe, curr_job_idx, time.time()))
        
    sub_process.start()
    
def _get_child_proc_data(sub_processes, max_time, res_data, args, delete_data):
    """
    
    Gets result data from sub-processes or terminates them if they have exceeded their life-time.
    
    :param sub_processes: List of all child-processes.
    :param max_time: Maximum time after which the sub-process is terminated. 
    :param res_data: Result data to be computed from the provided function (and input data).
    :param args: Arguments for the aformentioned function.
    :param delete_data: Flag on whether input data is deleted after successful computation of it's results.
    
    :return: Computed results from the sub-processes.
    
    """
    for parent_idx in np.arange(len(sub_processes) - 1, -1, -1):
        
        elapsed_time = time.time() - sub_processes[parent_idx][3]
        if (sub_processes[parent_idx][1].poll() == True):
            
            if (delete_data == True):
                job_idx = sub_processes[parent_idx][2]
                args[job_idx] = None
 
            res_data[sub_processes[parent_idx][2]] = sub_processes[parent_idx][1].recv()
            sub_processes[parent_idx][1].send(True)
            sub_processes[parent_idx][1].close()
             
            sub_processes[parent_idx][0].join()
            sub_processes[parent_idx][0].close()
             
            sub_processes.pop(parent_idx)
            
        elif (max_time is not None and elapsed_time > max_time):
            
            sub_processes[parent_idx][1].close()
            
            sub_processes[parent_idx][0].terminate()
             
            sub_processes[parent_idx][0].join()
            sub_processes[parent_idx][0].close()
            
            sub_processes.pop(parent_idx)
            
            return parent_idx
    
    return None

def run(max_child_proc_cnt = 4, func = None, args = None, max_time = None, delete_data = True):
    """
    Creates a subprocess loop to work the issue task defined by func and it's arguments. This subprocess loop is different in two key elements from the
    default python processpool.
    
    #. Only a single subprocess can return data at a time. This drastically decreases the odds of a memory utilization spikes which otherwise would cause a crash. This is linked to how pickle handles data transfer via pipes.
    #. A maximum time can be set after which a subprocess is terminated and restarted.
    
    :param max_child_proc_cnt: Number of child processes.
    :param func: The function to be processed.
    :param args: List of arguments. Every element in the list is handled by a separate process.
    :param max_time: Maximum time to wait for a processe prior to cancellation.
    :param delete_data: Flag on whether input data is deleted after successful computation of it's results.
    
    :return The processed information from func and args as a list. The order is identical to the order in which the argument blocks were given.
    
    """
    
    if (type(args) != np.ndarray):
        args = np.asarray(args, dtype = object)
    
    job_cnt = len(args)
    curr_job_idx = 0
    
    sub_processes = list()
    res_data = [None for _ in range(job_cnt)]
    
    return_data_lock = multiprocessing.Lock()
    
    while(len(sub_processes) != 0 or curr_job_idx < job_cnt):
        if (len(sub_processes) < max_child_proc_cnt and curr_job_idx < job_cnt):
            _launch_child_process(sub_processes, func, args, max_time, curr_job_idx, return_data_lock)
            curr_job_idx += 1

        failed_idx = _get_child_proc_data(sub_processes, max_time, res_data, args, delete_data)
        if (failed_idx is not None):
            _launch_child_process(sub_processes, func, args, max_time, failed_idx, return_data_lock)
                
        time.sleep(0.01)
        
    return res_data




