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
from nms import non_max_suppression
from extract_features import compute_feature


def detect(img, feature_type, downscale=1.5, visualize=False, apply_nms=True):
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
            pred = classifier.predict([feature])[0]
            confidence = classifier.decision_function([feature])[0]
            if pred == 1:
                print 'Detection at location ({}, {})'.format(x, y)
                print 'scale level: {}, confidence: {}'.format(scale_level,
                                                               confidence)

                # TODO: potential bug for coordinate restore
                # attr_vec: [x1, y1, x2, y2]
                attr_vec = np.array([x, y, min_window_size[0] + x,
                                     min_window_size[1] + y])
                curr_scale_dets.append((attr_vec, confidence))

                expand_rate = downscale ** scale_level
                attr_vec = np.around(attr_vec * expand_rate).astype('int')
                # detection: ([x1, y1, x2, y2], confidence)
                detections.append((attr_vec, confidence))

            # visualize: draw current sliding withdow
            # and detections at this scale
            # TODO: show confidence on the bounding box
            if visualize:
                im_copy = scaled_img.copy()
                for det, _ in curr_scale_dets:
                    cv2.rectangle(im_copy, (det[0], det[1]), (det[2], det[3]),
                                  color=(0, 0, 0), thickness=2)

                cv2.rectangle(im_copy, (x, y),
                              (x + im_window.shape[1], y + im_window.shape[0]),
                              color=(255, 255, 255), thickness=2)

                cv2.imshow('sliding window', im_copy)
                cv2.waitKey(20)

        scale_level += 1

    if not apply_nms:
        # withour non-maximum suppression, return with confidence
        # can be used for hard-negative mining and graphcut segmentation
        return detections

    # apply non-maximum suppression
    # dets = np.array(detections)[:, :-1]
    dets = np.array([i[0] for i in detections])
    detections = non_max_suppression(dets)
    if visualize:
        im_copy = img.copy()
        helper.draw_detections(im_copy, detections)

    return detections


if __name__ == '__main__':
    # test_img = '/home/billxue/Datasets/shoes7k/detection_data/r3/P70502-162009.jpg'
    # test_img = '/home/billxue/Datasets/shoes7k/detection_data/r3/P70502-162039.jpg'
    test_img = '/home/billxue/Datasets/shoes7k/detection_data/r3/P70502-162105.jpg'
    img = cv2.imread(test_img)
    img = rgb2gray(img)
    detect(img, 'HOG', visualize=True)
