'''
Helper functions
'''
import cv2


def sliding_window(img, window_size, step_size):
    """slide a window across the image and yield windows data iteratively"""
    for y in xrange(0, img.shape[0], step_size[1]):
        for x in xrange(0, img.shape[1], step_size[0]):
            # yield current window
            # NOTE: this window may be smaller than expected window_size
            yield (x, y, img[y:y + window_size[1], x:x + window_size[0]])


def pyramid(img, downscale=1.5, min_size=(30, 30)):
    """compute image pyramid through down sampling"""
    # min_size: (w, h)
    yield img

    while True:
        h = int(img.shape[0] / downscale)
        w = int(img.shape[1] / downscale)
        img = cv2.resize(img, (w, h))

        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break

        yield img


def draw_detections(img, detections):
    for det in detections:
        cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]),
                      color=(0, 0, 0), thickness=2)

    cv2.imshow('detection', img)
    cv2.waitKey(0)
