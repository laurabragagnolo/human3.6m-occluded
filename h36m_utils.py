import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools
import spacepy.pycdf


def get_h36m_activity_names(h36m_path, i_subject):
    
    subject_images_root = f'{h36m_path}/S{i_subject}/'
    subdirs = [elem for elem in os.listdir(subject_images_root)
               if os.path.isdir(f'{subject_images_root}/{elem}')]
    activity_names = set(elem.split('.')[0] for elem in subdirs if '_' not in elem)
    return sorted(activity_names)


def get_n_frames(h36m_path, i_subject, activity_name, frame_step=64):
    
    pose_folder = f'{h36m_path}/S{i_subject}/MyPoseFeatures'
    coord_path = f'{pose_folder}/D3_Positions/{activity_name}.cdf'

    with spacepy.pycdf.CDF(coord_path) as cdf_file:
        coords_raw_all = np.array(cdf_file['Pose'], np.float32)[0]
    
    coords_raw = coords_raw_all[::frame_step]
    n_frames = coords_raw.shape[0]
    return n_frames
