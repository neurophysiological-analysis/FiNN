#!/bin/bash -p

source $FREESURFER_HOME/SetUpFreeSurfer.sh

$FASTSURFER_HOME/run_fastsurfer.sh --sid $1 --t1 $2 --py $FASTSURFER_PYTHON_PATH --sd $SUBJECTS_DIR --fs_license $FREESURFER_LICENSE_PATH --threads 1 --device cpu

