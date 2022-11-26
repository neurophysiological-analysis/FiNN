#!/bin/bash -p


#export FREESURFER_HOME=/usr/local/freesurfer/7.2.0
#export SUBJECTS_DIR=/mnt/data/Professional/UHN/projects/data/MEG_TRD/fs2
#export SUBJECT=al0010a
#export FNAME=/mnt/data/Professional/UHN/projects/data/MEG_TRD/TRD/al0010a/dcm/_AX_3D_SPGR_IrPREP_20071005085905_2.nii

source $FREESURFER_HOME/SetUpFreeSurfer.sh

$FREESURFER_HOME/bin/mkheadsurf -s $1 -srcvol T1.mgz

