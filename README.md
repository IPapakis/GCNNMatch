# GCNNMatch- Under construction repo

This repository is the official code implementation of the paper GCNNMatch: [Graph Convolutional Neural Networks for Multi-Object Tracking via Sinkhorn Normalization](https://arxiv.org/abs/2010.00067). 

The method has been tested on MOT16 & 17 Datasets performing at [57% MOTA](https://motchallenge.net/method/MOT=3392&chl=10).

## Installing & Preparation:

* Install docker from its website.

* Git clone this repo folder and cd to it.

* docker build -t gcnnmatch .

* docker run -it --rm --gpus=all -v /local_repo_folder:/workspace gcnnmatch bash

Place MOT17 Dataset after download from [MOT website](https://motchallenge.net/data/MOT17/) and specify its location in tracking.py->get_data(). 

The following need to be run after "docker run", once you are inside the docker container.

## Training:

* Command: ./train.sh

* Result: Training will start and save the trained models in /models. Settings can be changed in tracking.py

## Testing

* Specify which trained model to use in tracking.py. A trained model can be found [here](https://drive.google.com/drive/folders/1b0ZF7WAQFIXv6xydyU3OGGBW-7EhegSv?usp=sharing).

* Command: ./test.sh

* Result: Testing will start and produce txt files and videos saved in /output. Settings can be changed in tracking.py

For Benchmark evaluation the pree-processed with Tracktor detection files from [this repo](https://github.com/dvl-tum/mot_neural_solver) were used.
