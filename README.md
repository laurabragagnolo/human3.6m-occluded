# Human3.6-Occluded: Dataset Augmentation for Occluded Human Pose Estimation

This repository provides code for generating the occluded version of the Human3.6M dataset used in our paper "Multi-view Pose Fusion for Occlusion-Aware 3D
Human Pose Estimation" (presented at ACVR @ ECCV 2024).
We take object and animals images from the Pascal VOC 2012 dataset as occluders. Considering the four available camera views for each scene in the dataset, we partially cover the subject's body on three out of four views.
An occluded view includes two random objects pasted over the human bounding box. Object size and location inside the box are chosen at random. 

Disclaimer:
> We have NO permission to redistribute the Human3.6M data. Please do NOT ask us for a copy of Human3.6M dataset.
