#!/bin/bash -p

source $FREESURFER_HOME/SetUpFreeSurfer.sh

$FREESURFER_HOME/bin/mkheadsurf -s $1 -srcvol T1.mgz

