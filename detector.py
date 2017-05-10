'''
Detector
-----------------
Detector workflow:
sliding window, extract features, classify it, generate candidates.
'''
import cv2
import numpy as np
from skimage.color import rgb2gray
from sklearn.externals import joblib

import config
import helper
from extract_features import compute_feature


def detect(img, feature_type, downscale=1.5, visualize=False):
    """use sliding window and pyramid to detect object"""
    detections = []  # detected candidates
    min_window_size = (config.img_width, config.img_height)
    classifier = joblib.load(config.model_path)
    scale_level = 0

    for scaled_img in helper.pyramid(img, downscale, min_window_size):
        # detections at current scale used for visualization
        curr_scale_dets = []

        for (x, y, im_window) in helper.sliding_window(
                scaled_img, min_window_size, config.step_size):
            # filter out not standard spliting window
            if im_window.shape[0] != min_window_size[1] or \
               im_window.shape[1] != min_window_size[0]:
                continue

            # compute feature
            feature = compute_feature(im_window, feature_type)

            # prediction
            pred = classifier.predict(feature)
            confidence = classifier.decision_function(feature)
            if pred == 1:
                print 'Detection at location ({}, {})'.format(x, y)
                print 'scale level: {}, confidence: {}'.format(scale_level,
                                                               confidence)

                # TODO: potential bug for coordinate restore
                # attr_vec: [x, y, width, height]
                attr_vec = np.array([x, y, min_window_size[0],
                                     min_window_size[1]])
                curr_scale_dets.append((attr_vec, confidence))

                expand_rate = downscale ** scale_level
                attr_vec = np.around(attr_vec * expand_rate).astype(int)
                detections.append((attr_vec, confidence))

            # visualize: draw current sliding withdow
            # and detections at this scale
            if visualize:
                im_copy = scaled_img.copy()
                for attrs, _ in curr_scale_dets:
                    cv2.rectangle(im_copy, (attrs[0], attrs[1]),
                                  (attrs[0] + attrs[2], attrs[1] + attrs[3]),
                                  color=(0, 0, 0), thickness=2)

                cv2.rectangle(im_copy, (x, y),
                              (x + im_window.shape[1], y + im_window.shape[0]),
                              color=(255, 255, 255), thickness=2)

                cv2.imshow('sliding window', im_copy)
                cv2.waitKey(20)

        scale_level += 1

    return detections


if __name__ == '__main__':
    test_img = '/home/billxue/Datasets/shoes7k/detection_data/r3/P70502-162009.jpg'
    img = cv2.imread(test_img)
    img = rgb2gray(img)
    detect(img, 'HOG', visualize=True)
