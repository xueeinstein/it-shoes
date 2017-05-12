'''
Segmentation
------------------
Apply graphcut segmentation algorithm to
image with bounding box detected shoes
'''
import cv2
from skimage.color import rgb2gray
import numpy as np


def segment(img, bounding_box, mask=None, iter_count=5):
    """segment shoes over the image"""
    # img: the rgb image, i.e. origin image
    # mask: a mask image which specify which areas are background, foreground
    if not mask:
        mask = np.zeros(img.shape[:2], np.uint8)
    bounding_box = tuple(bounding_box)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    mask, _, __ = cv2.grabCut(img, mask, bounding_box,
                              background_model, foreground_model,
                              iter_count, cv2.GC_INIT_WITH_RECT)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    seg_img = img * mask[:, :, np.newaxis]
    return seg_img


if __name__ == '__main__':
    test_img = '/home/billxue/Datasets/shoes7k/detection_data/r3/P70502-163002.jpg'
    img = cv2.imread(test_img)

    from matplotlib import pyplot as plt
    from detector import detect
    from helper import find_biggest_window
    detections = detect(rgb2gray(img), 'HOG', visualize=True)
    det = find_biggest_window(detections)
    print det
    seg_img = segment(img, det)
    plt.imsave('seg.jpg', seg_img)
    plt.imshow(seg_img)
    plt.show()
