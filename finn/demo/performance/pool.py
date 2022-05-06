'''
Created on Aug 5, 2020

@author: voodoocode
'''

import finn.misc.timed_pool as tp
import time
import numpy as np

import multiprocessing

import psutil

import joblib

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import finn.filters.frequency as ff

import finn.data.paths as paths

def demo_fnct2(data):
    data2 = ff.fir(data, 10, 20, 0.1, 5500, 10e-5, 10e-7, 5500, "zero")
    
    return data2

def demo_fnct3(data):
    
    data2 = ff.fir(data, 10, 20, 0.1, 5500, 10e-5, 10e-7, 5500, "zero")
    
    return data

import gc

def sub_process_work2_0(time_stamps, duration_times):
    print("framework")
    path = paths.per_pool_data
    data = np.load(path)
    data = np.repeat(data, repeat_size, axis = 1)
    data = np.expand_dims(data, axis = 1)
    args = data[:job_cnt, :, :]
    args = np.asarray(args)
    
    start = time.time_ns()
    time_stamps.append(time.time_ns())
    tp.run(parallel_processes, demo_fnct2, args)
    duration_times.append(time.time_ns() - start)
    time_stamps.append(time.time_ns())

def sub_process_work2_1(time_stamps, duration_times):
    print("loky")
    path = paths.per_pool_data
    data = np.load(path)
    data = np.repeat(data, repeat_size, axis = 1)
    
    start = time.time_ns()
    time_stamps.append(time.time_ns())
    joblib.Parallel(n_jobs = parallel_processes, backend = "loky")(joblib.delayed(demo_fnct2)(data[idx],) for idx in range(job_cnt))
    duration_times.append(time.time_ns() - start)
    time_stamps.append(time.time_ns())
    
def sub_process_work2_2(time_stamps, duration_times):
    print("multiprocessing")
    path = paths.per_pool_data
    data = np.load(path)
    data = np.repeat(data, repeat_size, axis = 1)
    
    start = time.time_ns()
    time_stamps.append(time.time_ns())
    joblib.Parallel(n_jobs = parallel_processes, backend = "multiprocessing")(joblib.delayed(demo_fnct2)(data[idx],) for idx in range(job_cnt))
    duration_times.append(time.time_ns() - start)
    time_stamps.append(time.time_ns())
    
def sub_process_work2_3(time_stamps, duration_times):
    path = paths.per_pool_data
    data = np.load(path)
    data = np.repeat(data, repeat_size, axis = 1)
    
    print("pool")
    pool = multiprocessing.Pool(parallel_processes)
    start = time.time_ns()
    time_stamps.append(time.time_ns())
    tmp = pool.starmap(demo_fnct2, [(data[idx],) for idx in range(job_cnt)])
    duration_times.append(time.time_ns() - start)
    time_stamps.append(time.time_ns())
    pool.close()
    pool.join()

    
def sub_process_work2(pipe, pause_time = 2):
    
    time_stamps = list()
    duration_times = list()
    
    sub_process_work2_0(time_stamps, duration_times)
    gc.collect();gc.collect();gc.collect()
 
    time.sleep(pause_time)

    sub_process_work2_1(time_stamps, duration_times)
    gc.collect();gc.collect();gc.collect()

    time.sleep(pause_time)

    sub_process_work2_2(time_stamps, duration_times)
    gc.collect();gc.collect();gc.collect()
 
    time.sleep(pause_time)

    sub_process_work2_3(time_stamps, duration_times)
    gc.collect();gc.collect();gc.collect()
 
    time.sleep(pause_time)
        
    pipe.send((time_stamps, duration_times))

import time

def main(save_results = False, overwrite = True, max_time_s = 30, delay_time_s = 2):
    if (overwrite):
        (parent_pipe, child_pipe) = multiprocessing.Pipe(True)
        sub_process = multiprocessing.Process(target = sub_process_work2, args = (child_pipe,))
        sub_process.start()
        
        pp = psutil.Process(sub_process.pid)
        
        mem = psutil.virtual_memory()
        
        virt_memory_usage_value = list()
        virt_memory_usage_time = list()
        start_time = time.time() # Measures a maximum of max_time_s (default: 30s) before the processes are forcefully terminated
        delay_time = None # In case the the processes return in time, delay_time_s (default: 2s) delay period is added
        while ((time.time() - start_time) < max_time_s):# (sub_process.is_alive() and wait_time < 2):
            descendants = list(pp.children(recursive = True))
            
            virt_memory = 0
            for _ in descendants:
                try:
                    mem = psutil.virtual_memory()
                    virt_memory += mem.used
                except:
                    pass
            
            virt_memory_usage_value.append(virt_memory)
            virt_memory_usage_time.append(time.time_ns())
        
            time.sleep(0.010)
            if (sub_process.is_alive() == False):
                if (delay_time is None):
                    delay_time = time.time()
                else:
                    if ((delay_time - time.time()) > delay_time_s):
                        break

                
            #print(time.time() - start_time)
        if ((time.time() - start_time) >= max_time_s):
            time_out = True
        else:
            time_out = False
        
        virt_memory_usage_value = np.asarray(virt_memory_usage_value)
        virt_memory_usage_time = np.asarray(virt_memory_usage_time)
        
        (time_stamps, duration_times) = parent_pipe.recv()
        time_stamps = np.asarray(time_stamps)
        duration_times = np.asarray(duration_times)
        
        if (save_results == True):
            np.save(config + "_time_stamps.npy", time_stamps)
            np.save(config + "_duration_times.npy", duration_times)
            
            np.save(config + "_virt_memory_usage_value.npy", virt_memory_usage_value)
            np.save(config + "_virt_memory_usage_time.npy", virt_memory_usage_time)
        print("pre-joined")
        if (time_out == False):
            sub_process.join()
        else:
            sub_process.terminate()
        print("joined")
    else:
        time_stamps = np.load(config + "_time_stamps.npy")
        duration_times = np.load(config + "_duration_times.npy")
        
        virt_memory_usage_value = np.load(config + "_virt_memory_usage_value.npy")
        virt_memory_usage_time = np.load(config + "_virt_memory_usage_time.npy")
    
    memory_peaks = list()
    for idx in range(4):
        if (idx == 0):
            pre_peak_area = np.intersect1d(np.argwhere(virt_memory_usage_time > 0).squeeze(), 
                                           np.argwhere(virt_memory_usage_time < time_stamps[int(idx * 2)]).squeeze())
        else:
            pre_peak_area = np.intersect1d(np.argwhere(virt_memory_usage_time > time_stamps[int(idx * 2 - 1)]).squeeze(), 
                                           np.argwhere(virt_memory_usage_time < time_stamps[int(idx * 2)]).squeeze())
        
        peak_area = np.intersect1d(np.argwhere(virt_memory_usage_time > time_stamps[int(idx * 2)]).squeeze(), 
                                   np.argwhere(virt_memory_usage_time < time_stamps[int(idx * 2 + 1)]).squeeze())
        
        memory_peaks.append(np.max(virt_memory_usage_value[peak_area]) - np.mean(virt_memory_usage_value[pre_peak_area]))
    
    
    plt.figure()
    plt.title("time usage")
    plt.ylim((0.8, 1.2))
    plt.xticks(range(4), ["framework", "joblib - loky", "joblib - multi", "multi - pool"], rotation = 45)
    plt.bar(range(len(duration_times)), duration_times/duration_times[0])
    plt.tight_layout()
    print("times:", duration_times)
     
    plt.figure()
    plt.title("memory usage")
    plt.ylim((0.5, 2.25))
    plt.xticks(range(5), ["framework new", "framework old", "joblib - loky", "joblib - multi", "multi - pool"], rotation = 45)
    plt.bar(range(len(memory_peaks)), memory_peaks/memory_peaks[0])
    plt.tight_layout()
    print("memory:", memory_peaks/memory_peaks[0])
    
    plt.figure()
    plt.plot(virt_memory_usage_value)
    plt.show(block = True)

config = "demo3"
parallel_processes = 12
if (config == "demo1"):
    job_cnt = 24
    repeat_size = 100
elif(config == "demo2"):
    job_cnt = 48
    repeat_size = 50
elif(config == "demo3"):
    job_cnt = 24
    repeat_size = 25

main()
print("terminated")











