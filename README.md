# it-shoes
It, shoes? It's a demo project to use both traditional computer vision methods and deep learning to detect and recognize shoes based on shoes7k dataset

## Configuration

Create a configuration file `config.cfg` under project folder `it-shoes` in format like:

```
[paths]
pos_images_path: /path/to/datasets/shoes7k/classification
neg_images_path: /path/to/datasets/shoes7k/classificationNeg
pos_images_lmdb: /path/to/datasets/shoes7k/pos_images_lmdb
neg_images_lmdb: /path/to/datasets/shoes7k/neg_images_lmdb
pos_features_path: /path/to/datasets/shoes7k/pos_features.csv
neg_features_path: /path/to/datasets/shoes7k/neg_features.csv
det_images_path: /path/to/datasets/shoes7k/detection_data/r3
det_annotation_path: /path/to/datasets/shoes7k/detection_data/r3.txt
model_path: /path/to/datasets/shoes7k/svm.model

[image]
height: 228
width: 228

[detector]
step_size: [10, 10]
jobs: 2

[evaluation]
eval_images_path: /path/to/datasets/shoes7k/detection_data/r2p2
eval_annotation_path: /path/to/datasets/shoes7k/detection_data/r2p2.txt
eval_res_path: /path/to/datasets/shoes7k/eval_res
```
