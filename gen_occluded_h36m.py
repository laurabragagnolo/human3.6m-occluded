import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
import argparse

from aug_utils import *
from h36m_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Generate occluded Human3.6M dataset')
    parser.add_argument('--pascalvoc_path', type=str, default='VOCdevkit/VOC2012', help='Path to Pascal VOC dataset')
    parser.add_argument('--h36m_path', type=str, default='/research/eccvw-24/h36m/processed', help='Path to Human3.6M dataset')
    parser.add_argument('--h36m_occ_path', type=str, default='/research/eccvw-24/h36m-fetch/processed/occluded/', help='Path to save occluded Human3.6M dataset')
    parser.add_argument('--num_occluded_images', type=int, default=3, help='Number of images to occlude in each sequence')
    parser.add_argument('--frame_step', type=int, default=64, help='Frame step for processing activities')
    return parser.parse_args()

def main():
    
    args = parse_args()

    print('Loading occluders from Pascal VOC dataset...')
    occluders = load_occluders(pascal_voc_root_path=args.pascalvoc_path)

    print('Loading images from Human3.6M dataset...')
    camera_names = ['54138969', '55011271', '58860488', '60457274']

    # add occlusions to h36m test set only
    subjects = [9, 11]

    total_activities = sum(len(get_h36m_activity_names(subj)) for subj in subjects)
    
    random.seed(42)
    with tqdm.tqdm(total=total_activities, desc="Processing Activities") as pbar:
        for i_subj in subjects:
            for activity in get_h36m_activity_names(args.h36m_path, i_subj):
                n_frames_total = get_n_frames(args.h36m_path, i_subj, activity)
                
                for i_frame in range(0, n_frames_total, args.frame_step):
                    bboxes = []
                    image_relpaths = []
                    for camera_name in camera_names:
                        image_relfolder = f'{args.h36m_path}/S{i_subj}/Images/{activity}.{camera_name}'
                        image_relpaths += [
                            f'{image_relfolder}/frame_{i_frame:06d}.jpg']
                        bbox_path = f'{args.h36m_path}/S{i_subj}/BBoxes/{activity}.{camera_name}.npy' 
                        bboxes.append(np.load(bbox_path)[i_frame])
                    
                    images = []
                    for image_path in image_relpaths:
                        image = cv2.imread(image_path)
                        images.append(image)

                    images_w_occlusion = dict()
                    occluded_image_indices = random.sample(range(0, 4), args.num_occluded_images)
                    
                    for i in occluded_image_indices:
                        occluded_im = occlude_with_objects(images[i], occluders, bboxes[i])
                        images_w_occlusion[i] = occluded_im
                            
                    for i in range(len(camera_names)):
                        output_dir = f'{args.h36m_occ_path}/S{i_subj}/Images-Occ-{args.num_occluded_images}/{activity}.{camera_names[i]}'
                        os.makedirs(output_dir, exist_ok=True)  # create the directory if it does not exist
                        if i in (occluded_image_indices):
                            cv2.imwrite(f'{output_dir}/frame_{i_frame:06d}.jpg', images_w_occlusion[i])
                        else:
                            cv2.imwrite(f'{output_dir}/frame_{i_frame:06d}.jpg', images[i])

                pbar.update(1) # update progress bar


if __name__=='__main__':
    main()