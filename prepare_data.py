'''
Prepare dataset
-------------------

Resize training and test images and store into LMDB
'''
import os
import cv2
import lmdb
import numpy as np
from caffe.io import array_to_datum
from caffe.proto import caffe_pb2

import config

POS, NEG = (1, 0)
SIMPLE_NORM, SQUARE_NORM = (0, 1)


def process_single_img(lmdb_txn, item_id, im, ptype):
    """convert single image according to different normalization methods"""
    datum = caffe_pb2.Datum()
    if ptype == SIMPLE_NORM:
        im = cv2.resize(im, (config.img_height, config.img_width))
        datum = array_to_datum(im)
        keystr = '{:0>8d}'.format(item_id)
        lmdb_txn.put(keystr, datum.SerializeToString())
    elif ptype == SQUARE_NORM:
        rim = np.transpose(im, axes=(1, 0, 2))
        im = cv2.resize(im, (config.img_height, config.img_width))
        rim = cv2.resize(rim, (config.img_height, config.img_width))

        # save origin image
        datum = array_to_datum(im)
        keystr = '{:0>8d}'.format(item_id)
        lmdb_txn.put(keystr, datum.SerializeToString())

        # save rotated image
        datum = array_to_datum(rim)
        keystr = '1{:0>7d}'.format(item_id)
        lmdb_txn.put(keystr, datum.SerializeToString())


def generate_lmdb(img_label, ptype=SIMPLE_NORM):
    """generate images lmdb"""
    if img_label == POS:
        lmdb_file = config.pos_images_lmdb
        images_path = config.pos_images_path
    elif img_label == NEG:
        lmdb_file = config.neg_images_lmdb
        images_path = config.neg_images_path

    if not os.path.exists(lmdb_file):
        batch_size = 256
        lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True)
        item_id = 0
        images = os.listdir(images_path)
        print('Start to generate lmdb')

        for img in images:
            if img[-4:] in ['.jpg', '.png']:
                im = cv2.imread(os.path.join(images_path, img))
                process_single_img(lmdb_txn, item_id, im, ptype)

                # write batch
                if (item_id + 1) % batch_size == 0:
                    lmdb_txn.commit()
                    lmdb_txn = lmdb_env.begin(write=True)
                    print('wrote {} images'.format(item_id + 1))

                item_id += 1

        # write last batch
        if (item_id + 1) % batch_size != 0:
            lmdb_txn.commit()
            print('wrote {} images'.format(item_id + 1))
            print('Generated {}'.format(lmdb_file))
    else:
        print('{} already exists'.format(lmdb_file))


if __name__ == '__main__':
    generate_lmdb(POS, SQUARE_NORM)
    generate_lmdb(NEG, SQUARE_NORM)
