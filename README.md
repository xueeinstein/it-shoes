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

[image]
height: 228
width: 228
```
