import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath('/mnt/data/Professional/projects/finnpy/latest/docs/'))

# a single build step, which keeps conf.py and versions.yaml at the main branch
# in generall we use environment variables to pass values to conf.py, see below
# and runs the build as we did locally
def build_doc(version, tag):
    os.environ["current_version"] = version
    subprocess.run("git checkout " + tag, shell=True)
    subprocess.run("git checkout develop -- conf.py", shell=True)
    subprocess.run("git checkout develop -- versions.yaml", shell=True)
    #subprocess.run("doxygen Doxyfile", shell=True)
    #subprocess.run("make html", shell=True)    

# a move dir method because we run multiple builds and bring the html folders to a 
# location which we then push to github pages
def move_dir(src, dst):
  subprocess.run(["mkdir", "-p", dst])
  subprocess.run("mv "+src+'* ' + dst, shell=True)

# to separate a single local build from all builds we have a flag, see conf.py
os.environ["build_all_docs"] = str(True)
os.environ["pages_root"] = "https://neurophysiological-analysis.github.io/FiNN/build/develop/index.html" 

# manually the main branch
build_doc("latest", "develop")
#move_dir("./_build/html/", "../build/")

import pathlib
import yaml
# reading the yaml file
with open(str(pathlib.Path().resolve()) + "/../versions.yaml", "r") as yaml_file:
  docs = yaml.safe_load(yaml_file)

# and looping over all values to call our build with version, and its tag
for version, details in docs.items():
  tag = details.get('tag', '')
  print(version, details, tag)
  #build_doc(version, version)
  #move_dir("./_build/html/", "../build/"+version+'/')
