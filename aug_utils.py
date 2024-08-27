import os
import random
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image


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
    
    xmin, ymin, width, height = bbox

    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box dimensions. Ensure that width > 0 and height > 0.")

    # Calculate the maximum x and y values for the bounding box
    xmax = xmin + width
    ymax = ymin + height

    # Generate random coordinates for the bottom-right corner of the rectangle
    rx2 = random.randint(xmin, xmax - 1)
    ry2 = random.randint(ymin, ymax - 1)

    return (max(0, rx2 - occluder_shape[1]), max(0, ry2 - occluder_shape[0]), occluder_shape[1], occluder_shape[0])


def occlude_with_objects(im, occluders, bbox):

    result = im.copy()
    width_height_person = np.asarray([bbox[2], bbox[3]])
    count = 2

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

    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))
