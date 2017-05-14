'''
Evaluate detector
----------------------
Evaluate detector and compare the bounding box with the annotations
'''
import os
import cv2
import pandas as pd
from skimage.color import rgb2gray

import config
from detector import detect_faster
from helper import get_overlap


def eval_detection(feature_type, downscale=1.5, overlap_threshold=.5):
    data = pd.read_table(config.eval_annotation_path, sep=' ').values
    corrects = 0.0
    if not os.path.exists(config.eval_res_path):
        os.makedirs(config.eval_res_path)

    for annotation in data:
        print 'Evaluate {}'.format(annotation[0])
        img_file = os.path.join(config.eval_images_path, annotation[0])
        img = cv2.imread(img_file)
        im_gray = rgb2gray(img)
        detection = detect_faster(im_gray, feature_type, downscale)

        if len(detection) > 0:
            det = detection[0]
            if get_overlap(annotation[3:7], det) > overlap_threshold:
                corrects += 1
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # draw bounding box
            cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]),
                          color=color, thickness=2)

        cv2.rectangle(img, (annotation[3], annotation[4]),
                      (annotation[5], annotation[6]),
                      color=(255, 0, 0), thickness=2)

        # save res image
        res_file = os.path.join(config.eval_res_path, annotation[0])
        cv2.imwrite(res_file, img)
        print 'Wrote detection result {}'.format(annotation[0])

    acc = corrects / len(data)
    print 'Detection accuracy (threshold={}): {}'.format(overlap_threshold,
                                                         acc)


if __name__ == '__main__':
    eval_detection('HOG')
