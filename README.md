# PlagDetect

Repository for Deep Learning based petrography of igneous Plagioclase crystals based on circular polarised light images of thin sections. We make extensive use of the MMDetection library, with the work based on DetectoRS models.

## Install

In order to install this private package you must be able to access it (which you can if you're reading this) and run have/create a python 3.7 environment for relevant package requirements (PyTorch can be a pain like that). 

#### Step 1
Activate environment: 
	
	eg conda activate PlagDetectEnv

#### Step 2
Install required libraries:

	pip install torch==1.7.0+cu110 torchvision==0.8.0 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
	
	pip install openmim

	pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

	mim install "mmdet<3.0.0"

#### Install PlagDetect
Install using the following command: 

	pip install git+ssh://git@github.com/norberttoth398/PlagDetect#egg=PlagDetect


#### Cite
Publication for the above repository is formally in the preparation stage. This work is presented at the EGU 2023 conference however, until formal publication of a manuscript please cite:

	N Toth, J Maclennan (2023) "A Deep Learning Enabled Approach for Igneous Textural Timescales", EGU 2023, DOI: 10.5194/egusphere-egu23-9289 
