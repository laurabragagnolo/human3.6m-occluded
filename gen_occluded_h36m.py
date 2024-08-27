#!/usr/bin/env python

import functools
import os
import random
import sys
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2
import PIL.Image
import itertools
import spacepy.pycdf
import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Generate occluded Human3.6M dataset')
    parser.add_argument('--pascalvoc_path', type=str, default='VOCdevkit/VOC2012', help='Path to Pascal VOC dataset')
    parser.add_argument('--h36m_path', type=str, default='/research/eccvw-24/h36m/processed', help='Path to Human3.6M dataset')
    parser.add_argument('--h36m_occ_path', type=str, default='/research/eccvw-24/h36m-fetch/processed/occluded/', help='Path to save occluded Human3.6M dataset')
    parser.add_argument('--num_occluded_images', type=int, default=3, help='Number of images to occlude')
    parser.add_argument('--frame_step', type=int, default=1, help='Frame step for processing activities')
    return parser.parse_args()
pascalvoc_path = 'VOCdevkit/VOC2012'
h36m_path = '/home/laura/research/eccvw-24/h36m-fetch/processed'
h36m_new_path = '/home/laura/research/eccvw-24/h36m-fetch/processed/occluded/'

def main():
    
    num_occluded_images = 3 # number of images to occlude

    print('Loading occluders from Pascal VOC dataset...')
    occluders = load_occluders(pascal_voc_root_path=pascalvoc_path)
    print('Found {} suitable objects'.format(len(occluders)))  # excluding 'person' objects

    print('Loading images from Human3.6M dataset...')
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    frame_step = 1

    # subject_list = [1, 5, 6, 7, 8, 9, 11]
    # action_list = [x for x in range(2, 17)]
    # action_name_list = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting', 
    # 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']
    # subaction_list = [x for x in range(1, 3)]
    # camera_list = [x for x in range(1, 5)]
    # train_list = [1, 5, 6, 7, 8]
    # test_list = [9, 11]

    subjects = [9, 11]

    total_activities = sum(len(get_activity_names(subj)) for subj in subjects)
    # total_activities = len(test_list) * len(action_list) * len(subaction_list) * len(camera_list)
    
    random.seed(42)
    with tqdm.tqdm(total=total_activities, desc="Processing Activities") as pbar:
        for i_subj in subjects:  # 9, 11
            for activity in get_activity_names(i_subj):
                n_frames_total = get_n_frames(i_subj, activity)
                print("n_frames_total: ", n_frames_total)
                
                for i_frame in range(0, n_frames_total, frame_step):
                    bboxes = []
                    image_relpaths = []
                    for camera_name in camera_names:
                        image_relfolder = f'{h36m_path}/S{i_subj}/Images/{activity}.{camera_name}'
                        image_relpaths += [
                            f'{image_relfolder}/frame_{i_frame:06d}.jpg']
                        bbox_path = f'{h36m_path}/S{i_subj}/BBoxes/{activity}.{camera_name}.npy' 
                        # print(bbox_path)
                        bboxes.append(np.load(bbox_path)[i_frame])
                    # print(image_relpaths)
                            
                    
                    images = []
                    for image_path in image_relpaths:
                        image = cv2.imread(image_path)
                        images.append(image)

                    images_w_occlusion = dict()
                    # apply occlusion to 1/2/3 images out of 4
                    occluded_image_indices = random.sample(range(0, 4), num_occluded_images)
                    # print(occluded_image_indices)

                    
                    for i in occluded_image_indices:
                        occluded_im = occlude_with_objects(images[i], occluders, bboxes[i])
                        # bbox = bboxes[i]
                        # x, y, w, h = bbox
                        # cv2.rectangle(occluded_im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                        # cv2.imshow('image', occluded_im)
                        # cv2.waitKey(0)
                        images_w_occlusion[i] = occluded_im
                            

                    # save images
                    for i in range(len(camera_names)):
                        output_dir = f'{h36m_new_path}/S{i_subj}/Images-Occ-2/{activity}.{camera_names[i]}'
                        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it does not exist
                        if i in (occluded_image_indices):
                            cv2.imwrite(f'{output_dir}/frame_{i_frame:06d}.jpg', images_w_occlusion[i])
                        else:
                            cv2.imwrite(f'{output_dir}/frame_{i_frame:06d}.jpg', images[i])

                pbar.update(1)  # Update progress after each activity is processed



def get_activity_names(i_subject):
    h36m_root = h36m_path
    subject_images_root = f'{h36m_root}/S{i_subject}/'
    subdirs = [elem for elem in os.listdir(subject_images_root)
               if os.path.isdir(f'{subject_images_root}/{elem}')]
    activity_names = set(elem.split('.')[0] for elem in subdirs if '_' not in elem)
    return sorted(activity_names)


def get_n_frames(i_subject, activity_name, frame_step=64):
    
    pose_folder = f'{h36m_path}/S{i_subject}/MyPoseFeatures'
    coord_path = f'{pose_folder}/D3_Positions/{activity_name}.cdf'

    # print("coord_path: ", coord_path)

    with spacepy.pycdf.CDF(coord_path) as cdf_file:
        coords_raw_all = np.array(cdf_file['Pose'], np.float32)[0]
    
    coords_raw = coords_raw_all[::frame_step]
    n_frames = coords_raw_all.shape[0]
    return n_frames


def get_n_frames(i_subject, activity_name):
    h36m_root = h36m_path
    images_root = f'{h36m_root}/S{i_subject}/{activity_name}/'
    n_frames = len(os.listdir(images_root + 'imageSequence' + '/' + '54138969'))
    
    return n_frames



def list_h36m_paths(i_subject, frame_step=64):

    camera_names = ['54138969', '55011271', '58860488', '60457274']
    all_image_relpaths = []
    for i_subj in i_subject:
        for activity, cam_id in itertools.product(get_activity_names(i_subj), range(4)):

            n_frames_total = get_n_frames(
                i_subject=i_subj, activity_name=activity, frame_step=frame_step)
            
            camera_name = camera_names[cam_id]
            image_relfolder = f'{h36m_path}/S{i_subj}/Images/{activity}.{camera_name}'
            all_image_relpaths += [
                f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                for i_frame in range(0, n_frames_total, frame_step)]
            
    return all_image_relpaths


def load_h36m_images(i_subject=(9,)):
    
    images = []
    all_images_paths = list_h36m_paths(i_subject=i_subject)
    for image_path in all_images_paths:
        image = cv2.imread(image_path)
        images.append(image)
    return images


def load_occluders(pascal_voc_root_path):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    
    annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_person = (obj.find('name').text == 'person')
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')

        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path,'SegmentationObject', seg_filename)

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)*255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                # Ignore small objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
            
            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            object = cv2.cvtColor(object_with_mask[:, :, :3], cv2.COLOR_BGR2RGB)
            object_with_mask = np.concatenate([object, object_with_mask[:, :, 3].reshape(object_with_mask.shape[0], object_with_mask.shape[1], 1)], axis=-1)
            occluders.append(object_with_mask)

    return occluders


def generate_bounding_box_with_min_area(original_bbox, min_area_percentage=0.2):
    """
    Generate a bounding box with an area of at least min_area_percentage of the given bounding box.

    Parameters:
    original_bbox (tuple): (x, y, width, height) of the original bounding box
    min_area_percentage (float): Minimum area percentage for the new bounding box

    Returns:
    tuple: (x, y, width, height) of the new bounding box
    """
    orig_x, orig_y, orig_width, orig_height = original_bbox
    original_area = orig_width * orig_height
    
    min_area = min_area_percentage * original_area
    
    while True:
        new_width = random.uniform(0.1 * orig_width, orig_width)
        new_height = random.uniform(0.1 * orig_height, orig_height)
        new_area = new_width * new_height
        
        if new_area >= min_area:
            new_x = random.uniform(orig_x, orig_x + orig_width - new_width)
            new_y = random.uniform(orig_y, orig_y + orig_height - new_height)
            
            return (new_x, new_y, new_width, new_height)


def generate_random_rect(bbox, occluder_shape):
    """
    Generate a random rectangle inside a given bounding box.

    Parameters:
    bbox (tuple): A tuple containing the bounding box in the format (xmin, ymin, width, height).

    Returns:
    tuple: A tuple containing the coordinates of the random rectangle in the format (rx1, ry1, rx2, ry2).
    """
    xmin, ymin, width, height = bbox

    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box dimensions. Ensure that width > 0 and height > 0.")

    # Calculate the maximum x and y values for the bounding box
    xmax = xmin + width
    ymax = ymin + height

    # # Generate random coordinates for the top-left corner of the rectangle
    # rx1 = random.randint(xmin, xmax - 1)
    # ry1 = random.randint(ymin, ymax - 1)

    # # Generate random coordinates for the bottom-right corner of the rectangle
    # rx2 = random.randint(rx1 + 1, xmax)
    # ry2 = random.randint(ry1 + 1, ymax)

    # Generate random coordinates for the bottom-right corner of the rectangle
    rx2 = random.randint(xmin, xmax - 1)
    ry2 = random.randint(ymin, ymax - 1)

    # return (rx1, ry1, rx2-rx1, ry2-ry1)

    # return (rx1, ry1, occluder_shape[1], occluder_shape[0])

    return (max(0, rx2 - occluder_shape[1]), max(0, ry2 - occluder_shape[0]), occluder_shape[1], occluder_shape[0])


def occlude_with_objects(im, occluders, bbox):
    """
    Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset.

    The occluders are randomly resized, rotated, and pasted onto `im`.

    Args:
        im (numpy.ndarray): The input image to be augmented.
        occluders (list): A list of occluder images from the Pascal VOC dataset.

    Returns:
        numpy.ndarray: The augmented image with occluders.

    """

    result = im.copy()
    # width_height = np.asarray([im.shape[1], im.shape[0]])
    # print(width_height)
    # # retrieve human bounding box
    # im_scale_factor = min(width_height) / 500
    width_height_person = np.asarray([bbox[2], bbox[3]])
    # print(im_scale_factor)
    count = 2
    # print("count", count)
    # print('Adding {} objects'.format(count))

    for _ in range(count):
        occluder = random.choice(occluders)
        # print(occluder.shape)
        random_scale_factor = random.uniform(0.5, 1.0)      # occ-hard is 2 obj
        # print(random_scale_factor)
        width_height_occluder = np.asarray([occluder.shape[1], occluder.shape[0]])
        occl_scale_factor = min(width_height_person) / min(width_height_occluder)
        scale_factor = random_scale_factor * occl_scale_factor
        occluder = resize_by_factor(occluder, scale_factor)
        # occluder_bbox = generate_random_rect(bbox, occluder.shape)
        x, y, width, height = bbox
        center = (random.uniform(x, x + width), random.uniform(y, y + height))
        # print(center)
        result = paste_over(im_src=occluder, im_dst=result, center=center)

    return result


def paste_over(im_src, im_dst, bbox=(0, 0, 512, 512), center=(256, 256)):
    """
    Pasting `im_src` onto `im_dst` at a specified bounding box `bbox`.
    `bbox` is a tuple of four integers with the format (xmin, ymin, width, height).
    """
    """
    im_dst = cv2.resize(im_dst, (1000, 1000))
    
    xmin, ymin, width, height = bbox
    xmin = int(xmin)
    ymin = int(ymin)
    width = int(width)
    height = int(height)

    xmax = xmin + width
    ymax = ymin + height

    # Resize `im_src` to fit into the bounding box
    # change this linee!!!!!!
    im_src = cv2.resize(im_src, (min(width, 1000), min(height, 1000)))

    # print("has alpa channel: ", im_src.shape[2] == 4)

    print(im_dst.shape)
    print(im_src.shape)

    # If `im_src` has an alpha channel, blend it with `im_dst` based on the alpha channel
    if im_src.shape[2] == 4:
        alpha = im_src[:, :, 3:4] / 255.0  # Normalize the alpha channel to [0, 1]
        im_dst[ymin:ymax, xmin:xmax, ...] = alpha * im_src[:, :, :3] + (1 - alpha) * im_dst[ymin:ymax, xmin:xmax, ...]
    else:
        # If `im_src` does not have an alpha channel, paste, it onto `im_dst` as before
        im_dst[ymin:ymax, xmin:xmax, ...] = im_src[:, :, :3]

    return im_dst

    """

    # original code from isarandi
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)
    
    return im_dst


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


# def scale_image_with_min_length(image, reference_image, min_length_ratio=0.15):
#     # Get the dimensions of the reference image (larger image)
#     ref_height, ref_width = reference_image.shape[:2]
    
#     # Calculate the minimum allowed dimensions based on the reference image
#     min_length = min(ref_height, ref_width) * min_length_ratio
    
#     # Get the dimensions of the smaller image
#     height, width = image.shape[:2]
    
#     # Calculate the minimum scale factor to ensure the smaller image meets the minimum length requirement
#     min_scale_factor = max(min_length / height, min_length / width)
    
#     # Generate a random scale factor between the minimum scale factor and 1
#     scale_factor = np.random.uniform(min_scale_factor, 1)
    
#     # Calculate new dimensions using the random scale factor
#     new_width = int(width * scale_factor)
#     new_height = int(height * scale_factor)
    
#     # Scale the image
#     scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
#     return scaled_image


if __name__=='__main__':
    main()

    # load segmentation masks
    # find location inside segmentation mask

