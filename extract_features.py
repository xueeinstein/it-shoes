'''
Extract features
-------------------
Extract features from LMDB and save into csv files
'''

import lmdb
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from caffe.io import datum_to_array
from caffe.proto import caffe_pb2

import config

POS, NEG = (1, 0)


def compute_feature(img, feature_type):
    """compute feature from gray image"""
    if feature_type == 'HOG':
        feature = hog(img)
    elif feature_type == 'LBP':
        feature = local_binary_pattern(img)

    return feature


def extract_features(img_label, feature_type):
    """extract features from training or test image set and save into csv"""
    if img_label == POS:
        lmdb_file = config.pos_images_lmdb
        feature_file = config.pos_features_csv
    elif img_label == NEG:
        lmdb_file = config.neg_images_lmdb
        feature_file = config.neg_features_csv

    feature_lst = []
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    is_first = True
    batch_size = 256
    item_id = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)

        img = datum_to_array(datum)
        img = rgb2gray(img)
        feature = compute_feature(img, feature_type)
        feature_lst.append(feature)

        # save batch to csv
        if (item_id + 1) % batch_size == 0:
            feature_mat = np.array(feature_lst)
            data_frame = pd.DataFrame(feature_mat)
            feature_lst = []
            if is_first:
                data_frame.to_csv(feature_file)
                is_first = False
            else:
                data_frame.to_csv(feature_file, mode='a', header=False)
            print 'saved', item_id + 1, 'image feature'

        item_id += 1

    # save extra feature to csv
    if (item_id + 1) % batch_size != 0:
        feature_mat = np.array(feature_lst)
        data_frame = pd.DataFrame(feature_mat)
        if is_first:
            data_frame.to_csv(feature_file)
        else:
            data_frame.to_csv(feature_file, mode='a', header=False)

        print 'saved', item_id + 1, 'image feature'
        print 'save', feature_file


if __name__ == '__main__':
    # extract_features(POS, 'HOG')
    extract_features(NEG, 'HOG')
