#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:28:16 2020
Creates all the dataset for O3 Cl2.
The video size is shape to 75, 100, 100, 3
75 is 75 seconds.
Picked grid 7, 
@author: Shengli Jiang (sjiang87@wisc.edu)
"""

import numpy as np
import h5py
import glob
import cv2
import skimage.color


def produce_data(ind, data_dirs):
    data_dir = data_dirs[ind]
    y = data_dir.split('_')
    y1 = int(float(y[0].split('\\')[-1]))
    y2 = float(y[-2])
    y3 = float(y[-1].split('.h5')[0])
    y = np.array([y1, y2, y3])

    h5f = h5py.File(data_dir, 'r')
    x = h5f.get('dataset_1')
    x = np.array(x) / 255.0
    h5f.close()

    for i in [6, 7, 11, 12, 13, 14, 17, 18, 19, 20, 24, 25]:
        image_tensor = np.empty(shape=(240, 48, 48, 3))
        size_t = int(len(x))
        x_temp = x[:, i, :, :]
        if size_t < 240:
            for j in range(size_t):
                img_temp = x_temp[j, :, :, :]
                img_temp = cv2.resize(img_temp, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
                image_tensor[j, :, :, :] = img_temp
            for j in range(size_t, 240):
                image_tensor[j, :, :, :] = img_temp
        else:
            for j in range(240):
                img_temp = x_temp[j, :, :, :]
                img_temp = cv2.resize(img_temp, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
                image_tensor[j, :, :, :] = img_temp

        for k in range(4):
            image_tensor = np.rot90(image_tensor, k=1, axes=(1, 2))
            lab_tensor = skimage.color.rgb2lab(image_tensor)
            h5f = h5py.File(r'F:\\O3Cl2_RGB\\O3Cl2_{}_{}_{}_{}_{}.h5'.format(y1, y2, y3, i, k), 'w')
            h5f.create_dataset('x', data=image_tensor, compression="gzip")
            h5f.create_dataset('y', data=y, compression="gzip")
            h5f.close()
            h5f = h5py.File(r'F:\\O3Cl2_LAB\\O3Cl2_{}_{}_{}_{}_{}.h5'.format(y1, y2, y3, i, k), 'w')
            h5f.create_dataset('x', data=lab_tensor, compression="gzip")
            h5f.create_dataset('y', data=y, compression="gzip")
            h5f.close()
            print(r'O3Cl2_{}_{}_{}_{}_{} finished...'.format(y1, y2, y3, i, k))


def main():
    data_dirs = glob.glob(r'E:\\O3Cl2\\*.h5')
    for ind in range(len(data_dirs)):
        produce_data(ind, data_dirs)


if __name__ == "__main__":
    main()
