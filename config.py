'''
config
'''
import ConfigParser as cp


config = cp.RawConfigParser()
config.read('./config.cfg')

pos_images_path = config.get('paths', 'pos_images_path')
neg_images_path = config.get('paths', 'neg_images_path')
pos_images_lmdb = config.get('paths', 'pos_images_lmdb')
neg_images_lmdb = config.get('paths', 'neg_images_lmdb')

# size of train and test images
img_height = config.get('image', 'height')
img_width = config.get('image', 'width')
