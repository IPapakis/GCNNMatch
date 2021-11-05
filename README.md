# GCNNMatch: Graph Convolutional Neural Networks for Multi-Object Tracking via Sinkhorn Normalization

This repository is the official code implementation of the GCNNMatch: Graph Convolutional Neural Networks for Multi-Object Tracking via Sinkhorn Normalization on [IEEE](https://ieeexplore.ieee.org/document/9564655) and on [Arxiv](https://arxiv.org/abs/2010.00067). Link to access a new traffic vehicle monitoring dataset named "VA Beach Traffic Dataset" will be provided here.

## Citing:
If you find this paper or code useful, please cite using the following:

```
@article{papakis2020gcnnmatch,
  title={GCNNMatch: Graph Convolutional Neural Networks for Multi-Object Tracking via Sinkhorn Normalization},
  author={Papakis, Ioannis and Sarkar, Abhijit and Karpatne, Anuj},
  journal={arXiv preprint arXiv:2010.00067},
  year={2020}
}
```

## Installing & Preparation:

* Install singularity following instructions from its [website](https://sylabs.io/guides/3.0/user-guide/quick_start.html#quick-installation-steps).

* Git clone this repo folder and cd to it.

* "sudo singularity build geometric.sif singularity". Follow instructions from [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric/tree/master/docker) to change settings if needed for your system.

* Download MOT17 Dataset from [MOT website](https://motchallenge.net/data/MOT17/) and place it in a folder /MOT_dataset. 

* "mkdir overlay". It will allow you to install additional packages if needed in the future.

* "sudo singularity run --nv -B /MOT_dataset/:/data --overlay overlay/ geometric.sif"

* "./create_folders.sh"

## Training:

* Command: ./train.sh

* Result: Training will start and save the trained models in /models. Settings can be changed in tracking.py.

## Testing:

* Specify which trained model to use in tracking.py. A trained model can be found [here](https://drive.google.com/drive/folders/1b0ZF7WAQFIXv6xydyU3OGGBW-7EhegSv?usp=sharing).

* Command: ./test.sh

* Result: Testing will start and produce txt files and videos saved in /output. Settings can be changed in tracking.py

For Benchmark evaluation the pre-processed with Tracktor detection files from [this repo](https://github.com/dvl-tum/mot_neural_solver) were used.

