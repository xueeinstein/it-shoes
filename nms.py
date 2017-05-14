'''
Non-maximum suppression
--------------------------
To remove redundant and overlapping bounding box

ref: https://github.com/quantombone/exemplarsvm/blob/master/internal/esvm_nms.m
rewrite to python
'''
import numpy as np


def non_max_suppression(boxes, overlap_threshold=.5, only_one=False):
    """apply non-maximum suppression algorithm to remove some bounding boxes"""
    # boxes is matrix of box: [x1, y1, x2, y2]
    if len(boxes) == 0:
        return []

    # for doing divisions
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the overlap ratio
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that is larger than threshold
        to_delete = np.where(overlap > overlap_threshold)[0]
        idxs = np.delete(idxs, np.concatenate(([last], to_delete)))

    res = boxes[pick].astype('int')
    if only_one:
        res = merge_to_one(res)

    return res


def merge_to_one(boxes):
    """merge multiple boxes into single box"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    xx1 = np.min(x1)
    yy1 = np.min(y1)
    xx2 = np.max(x2)
    yy2 = np.max(y2)

    return np.array([[xx1, yy1, xx2, yy2]])
