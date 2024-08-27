# Human3.6-Occluded: Dataset Augmentation for Occluded Human Pose Estimation

This repository provides code for generating the occluded version of the Human3.6M dataset used in our paper <b>"Multi-view Pose Fusion for Occlusion-Aware 3D
Human Pose Estimation" (presented at ACVR @ ECCV 2024)</b>.
We take object and animals images from the Pascal VOC 2012 dataset as occluders. Considering the four available camera views for each scene in the dataset, we partially cover the subject's body on three out of four views.
An occluded view includes two random objects pasted over the human bounding box. Object size and location inside the box are chosen at random. 
This code has been produced following the work of I. Sárándi et al. [1] .

## Disclaimer
> We have NO permission to redistribute the Human3.6M dataset or parts of it. Please do NOT ask us for a copy of Human3.6M dataset.

## References
[1] I. Sárándi; T. Linder; K. O. Arras; B. Leibe: "[How Robust is 3D Human Pose Estimation to Occlusion?](https://arxiv.org/abs/1808.09316)" in IEEE/RSJ Int. Conference on Intelligent Robots and Systems (IROS'18) Workshops (2018) arXiv:1808.09316
