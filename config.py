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
pos_features_csv = config.get('paths', 'pos_features_path')
neg_features_csv = config.get('paths', 'neg_features_path')
model_path = config.get('paths', 'model_path')

# size of train and test images
img_height = config.getint('image', 'height')
img_width = config.getint('image', 'width')
