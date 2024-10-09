
.. _src_rec_install_label:

FreeSurfer installation guide
=============================

This guide explains how to install FreeSurfer. In general, it is recommended to save FreeSurfer in a folder (or subfolder) indicating the version of FreeSurfer and, if possible, to stay with the same version throughout a project. Although randomized steps are only pseud-randomized via seeding random numbers with *1234*, alterations to the feature extraction algorithms may produce varying results. As such, preproducibility is limited if different FreeSurfer versions are mixed in the context of a single project.

Linux
-----

FreeSurfer may be directly downloaded from the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall).
In addition, the following libraries are required: bc, libjpeg6-turbo, libxp, tcsh, libpng12 (version 7.4.1 - 2024/01/26).
It is highly recommended to move the downloaded FreeSurfer directory to a permanent location, such as /usr/local/FreeSurfer.

Windows (by Prerana Keerthi)
----------------------------

As FreeSurfer does not support Windows directly, the application may be used through the WIndow Subsystem for Linux as follows:

1. Windows Subsystem for Linux (WSL) Installation: https://learn.microsoft.com/en-us/windows/wsl/install
2. Freesurfer on WSL Installation: https://surfer.nmr.mgh.harvard.edu/fswiki/FS7_wsl_ubuntu
3. Freesurfer on Xming Installation: https://surfer.nmr.mgh.harvard.edu/fswiki/FSL_wsl_xming

Windows Subsystem for Linux (WSL) Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run freesurfer on Windows you will need Windows Subsystem for Linux (WSL). If WSL is not already installed on your PC, in Windows Powershell or Windows command prompt enter the following command:

.. code-block::

	wsl –install

If WSL is already installed, please ensure an up to date version is used. The following command indicates which distros are installed and the asterisk in the output represents the active distribution. 

.. code-block::

	wsl --list –verbose

The following lines of code demonstrate how to install Ubuntu-22.04 as a distribution and set it as the default:

.. code-block::

	wsl --install -d Ubuntu-22.04
	wsl --setdefault Ubuntu-22.04

Freesurfer Installation
^^^^^^^^^^^^^^^^^^^^^^^

This installation guide is inspired by the original from FreeSurfer, available at https://surfer.nmr.mgh.harvard.edu/fswiki/FS7_wsl_ubuntu.

Via the following commands, FreeSurfer may be installed into the WSL.

.. code-block::

	cd
	wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer_ubuntu22-7.4.1_amd64.deb

	ls freesurfer_ubuntu22-7.4.1_amd64.deb
	sudo apt-get update -y
	sudo apt-get -y install ./freesurfer_ubuntu22-7.4.1_amd64.deb
	export FREESURFER_HOME=/usr/local/freesurfer/7.4.1 
	echo "export FREESURFER_HOME=/usr/local/freesurfer/7.4.1" >> $HOME/.bashrc
	
Executing the ls command will display all items within the FreeSurfer directory, and as such should list files such as sources.sh, SetUpFreeSurfer.sh, and the bin folder.
	
.. code-block::

	ls $FREESURFER_HOME

Afterwards, a free Freesurfer license needs to be optained via the following link, https://surfer.nmr.mgh.harvard.edu/registration.html. The received license.txt needs to be moved into the FreeSurfer directory (within the WSL) and shortcuts created. 

.. code-block::

	mv /mnt/c/Users/username/Downloads/licence.txt $HOME
	echo "export FS_LICENSE=$HOME/license.txt" >> $HOME/.bashrc
	echo "export XDG_RUNTIME_DIR=$HOME/.xdg" >> $HOME/.bashrc
	echo "export DISPLAY=:0" >> $HOME/.bashrc

As such, when executing the following code

.. code-block::

	export FREESURFER_HOME=/usr/local/freesurfer/7.4.1 
	source $FREESURFER_HOME/SetUpFreeSurfer.sh

the corresponding output should be.

.. code-block::

	-------- freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460 --------
	Setting up environment for FreeSurfer/FS-FAST (andFSL)
	FREESURFER_HOME		/usr/local/freesurfer/7.4.1
	FSFAST_HOME		/usr/local/freesurfer/7.4.1/fsfast
	FSF_OUTPUT_FORMAT	nii.gz
	SUBJECTS_DIR		YOUR_SUBJECTS_DIRECTORY
	MNI_DIR			/usr/local/freesurfer/7.4.1/mni

Results may slightly vary for different linux distributions (first line) or different FreeSurfer versions (7.4.1 used in this example).

X-server Installation
^^^^^^^^^^^^^^^^^^^^^

To run image viewing software such as freesurfer on WSL (for reference, see https://surfer.nmr.mgh.harvard.edu/fswiki/FSL_wsl_xming), an X-server needs to be set up. The previous link describes how to set up the Xming X-server. Once Xming is installed, correct installation may be verified via the following WSL terminal command.

.. code-block:: python

	freeview

If the command fails, potential troubleshooting is described below.

If the following error occurs

.. code-block::

	QXcbConnection: Could not connect to display :0 
	Could not connect to any X display.

Resolution details are provided in https://surfer.nmr.mgh.harvard.edu/fswiki/FS7_wsl_ubuntu.

The error

.. code-block::

	qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
	This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

may be resolved by installing VcxSrv instead of Xmin, see https://sourceforge.net/projects/vcxsrv.
Xmin should be configured as follows:

- Display settings: Multiple Windows
- How to start clients: Start no client
- Extra settings, Clipboard: Checked
- Extra settings, Primary Selection: Checked
- Extra settings, Native opengl: Checked

