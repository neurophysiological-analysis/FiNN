"""
Created on May 27, 2025.

@author: voodoocode
"""

import os
import subprocess
import shutil

def run(subject_id, cmd):
    """
    Create a custom work directory to run external function calls.
    
    Parameters
    ----------
    subject_id : string
                 Name of the subject whose data is worked on.
    cmd : string
          The freesurfer command to be executed in the custom environment.
     
    Raises
    ------
    AssertionError
        To prevent accidentially overwriting temporary data, a blocking temporary folder is created while a subject is worked on.
        In case the routine fails, this folder may have to be manually deleted.
    """
    path_to_tmp_cwd = "finnpy_" + subject_id + "_freesurfer_tmp_dir/"  # A temporary working directory is needed as freesurfer saves intermediate results in files.
        
    if (os.path.exists(path_to_tmp_cwd + ".lock")):  # Checks if the current directory is already worked in, if yes, raise error.
        raise AssertionError("Subject is already being worked on as %s already exists.\n" % (path_to_tmp_cwd + ".lock",)
                             + "This lock is in place to prevent errornous executions on the same data.\n"
                             + "If this does not apply, please remove %s prior proceeding." % (os.path.abspath(path_to_tmp_cwd),))
    
    os.makedirs(path_to_tmp_cwd, exist_ok = True)
    file = open(path_to_tmp_cwd + ".lock", "wb")
    file.close()
    
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, cwd = path_to_tmp_cwd, env = os.environ.copy())
    for c in iter(lambda: process.stderr.read(1), b""):
        try:
            print(c.decode(), end = "")
        except UnicodeDecodeError:  # Sometimes unreadable characters may get generated (i.e. bash progress animations).
            pass
    
    shutil.rmtree(path_to_tmp_cwd)  # And removed later on
