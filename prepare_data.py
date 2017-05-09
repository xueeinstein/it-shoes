'''
Prepare dataset
-------------------

Resize training and test images and store into LMDB
'''
import os
import cv2
import lmdb
from caffe.io import array_to_datum
from caffe.proto import caffe_pb2

import config

POS, NEG = (1, 0)


def generate_lmdb(img_label):
    """generate images lmdb"""
    if img_label == POS:
        lmdb_file = config.pos_images_lmdb
        images_path = config.pos_images_path
    elif img_label == NEG:
        lmdb_file = config.neg_images_lmdb
        images_path = config.neg_images_path

    if not os.path.exists(lmdb_file):
        batch_size = 256
        datum = caffe_pb2.Datum()
        lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True)
        item_id = 0
        images = os.listdir(images_path)
        print 'Start to generate lmdb'

        for img in images:
            if img[-4:] in ['.jpg', '.png']:
                # prepare data
                # NOTE: maybe rotate image and re-train classifier
                im = cv2.imread(os.path.join(images_path, img))
                im = cv2.resize(im, (config.img_height, config.img_width))
                datum = array_to_datum(im)
                keystr = '{:0>8d}'.format(item_id)
                lmdb_txn.put(keystr, datum.SerializeToString())

                # write batch
                if (item_id + 1) % batch_size == 0:
                    lmdb_txn.commit()
                    lmdb_txn = lmdb_env.begin(write=True)
                    print 'wrote', item_id + 1, 'images'

                item_id += 1

        # write last batch
        if (item_id + 1) % batch_size != 0:
            lmdb_txn.commit()
            print 'wrote', item_id + 1, 'images'
            print 'Generated', lmdb_file
    else:
        print lmdb_file, 'already exists'


if __name__ == '__main__':
    generate_lmdb(POS)
    generate_lmdb(NEG)
