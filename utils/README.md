# utils: A set of dataset preprocess or postprocess scripts.
## 100doh_preprocess
The representation of keys in 100DOH dataset can be found in [readme](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/README_100K.md).
It contains bounding box information of objects, so we need to use SAM to make segmentation conditioned on the boxes.

Arguments:
+ dataset_dir: directory storing original 100DOH dataset with VOC format.
+ save_dir: directory to save processed dataset.
+ sam_checkpoint: path of SAM checkpoint.