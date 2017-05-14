'''
Hard-negative mining
----------------------
For each image and each possible scale of each image,
sample its false positives and re-train the detector.
'''
import os
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2gray

import config
from detector import detect_faster
from extract_features import compute_feature


def sample_hard_negatives(img, feature_type, annotations, downscale=1.5,
                          overlap_threshold=0.5):
    """sample hard negatives from detection data
    and save their features into csv"""
    # img: is the grayscale image
    # annotations: name, h, w, x1, y1, x2, y2, shoe_type, orientation
    detections = detect_faster(img, feature_type, downscale, apply_nms=False)
    batch = []
    for det, _ in detections:
        x_min = max(0, det[0] - 1)
        y_min = max(0, det[1] - 1)
        im_window = img[y_min:det[3] - 1, x_min:det[2] - 1]
        if im_window.shape != (config.img_height, config.img_width):
            continue
        area = (det[2] - det[0] + 1) * (det[3] - det[1] + 1)
        xx1 = max(det[0], annotations[3])
        yy1 = max(det[1], annotations[4])
        xx2 = min(det[2], annotations[5])
        yy2 = min(det[3], annotations[6])

        w = max(0, xx2 - xx1 + 1)
        h = max(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area + 0.0)
        if overlap < overlap_threshold:
            # this is hard negative
            feature = compute_feature(im_window, feature_type)
            batch.append(feature)

    # write into neg_features_csv
    print 'Wrote hard negatives of {}'.format(annotations[0])
    feature_mat = np.array(batch)
    data_frame = pd.DataFrame(feature_mat)
    data_frame.to_csv(config.neg_features_csv, mode='a', header=False)


def hard_negatives_mining(feature_type, downscale=1.5, overlap_threshold=0.5):
    """do hard negatives mining over detection images"""
    # read annotations
    data = pd.read_table(config.det_annotation_path, sep=' ').values
    for annotation in data:
        img_file = os.path.join(config.det_images_path, annotation[0])
        img = cv2.imread(img_file)
        img = rgb2gray(img)
        print 'Begin to process {}'.format(annotation[0])
        sample_hard_negatives(img, feature_type, annotation, downscale,
                              overlap_threshold)


if __name__ == '__main__':
    hard_negatives_mining('HOG')
