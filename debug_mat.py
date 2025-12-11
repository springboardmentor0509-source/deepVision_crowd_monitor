#!/usr/bin/env python
from scipy.io import loadmat
import numpy as np

mat_path = 'Dataset/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_1.mat'
mat = loadmat(mat_path)
img_info = mat['image_info']

print("image_info shape:", img_info.shape)
obj = img_info[0, 0]
print("Object dtype names:", obj.dtype.names)

for attr in obj.dtype.names:
    val = obj[attr]
    if hasattr(val, 'shape'):
        print(f"{attr}: shape={val.shape}, dtype={val.dtype}")
        if val.size > 0:
            if len(val.shape) > 1 and val.shape[0] > 0 and val.shape[1] > 0:
                print(f"  First element: {val[0, 0]}")
            elif len(val.shape) == 1 and val.shape[0] > 0:
                print(f"  First 5 elements: {val[:min(5, val.shape[0])]}")
