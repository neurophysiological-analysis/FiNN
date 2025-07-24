#!/bin/bash -p

source $FREESURFER_HOME/SetUpFreeSurfer.sh

$FREESURFER_HOME/bin/recon-all -s $1 -i $2 -all -qcache

