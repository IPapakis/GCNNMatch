# GCNNMatch

This repository is the official code implementation of the paper GCNNMatch: [Graph Convolutional Neural Networks for Multi-Object Tracking via Sinkhorn Normalization](https://arxiv.org/abs/2010.00067). 

The method has been tested on MOT16 & 17 Datasets performing at [57% MOTA](https://motchallenge.net/method/MOT=3392&chl=10).

## Installing & Preparation:

* Install singularity following instructions from its [website](https://sylabs.io/guides/3.0/user-guide/quick_start.html#quick-installation-steps).

* Git clone this repo folder and cd to it.

* "sudo singularity build geometric.sif singularity". Follow instructions from [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric/tree/master/docker) to change settings if needed for your system.

* Download MOT17 Dataset from [MOT website](https://motchallenge.net/data/MOT17/) and place it in a folder /MOT_dataset. 

* "mkdir overlay". It will allow you to install additional packages if needed in the future.

* "sudo singularity run --nv -B /MOT_dataset/:/data --overlay overlay/geometric.sif"

* "./create_folders.sh"

## Training:

* Command: ./train.sh

* Result: Training will start and save the trained models in /models. Settings can be changed in tracking.py

## Testing:

* Specify which trained model to use in tracking.py. A trained model can be found [here](https://drive.google.com/drive/folders/1b0ZF7WAQFIXv6xydyU3OGGBW-7EhegSv?usp=sharing).

* Command: ./test.sh

* Result: Testing will start and produce txt files and videos saved in /output. Settings can be changed in tracking.py

For Benchmark evaluation the pre-processed with Tracktor detection files from [this repo](https://github.com/dvl-tum/mot_neural_solver) were used.