# -*- coding: utf-8 -*-
import sys
sys.path.append('.')

import argparse
import numpy as np
import cv2
from torch.utils.data.dataloader import DataLoader

from h5dataloader.pytorch import HDF5Dataset
from pointsmap import depth2colormap

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, metavar='PATH', nargs='*', required=True, help='HDF5 Files')
parser.add_argument('-c', '--config', type=str, metavar='PATH', required=True, help='Config File')
args = parser.parse_args()

h5dataset = HDF5Dataset(h5_paths=args.dataset, config=args.config, quiet=True, visibility_filter_radius=3, block_size=16, use_mods=(0, 14))
print(h5dataset.minibatch)
h5_dataloader = DataLoader(h5dataset, batch_size=1)

for batch_itr, batch in enumerate(h5_dataloader):
    print("\r", batch_itr, end=' ')
    map_depth = np.squeeze(np.transpose(batch['map'].numpy()[0], [1, 2, 0]))
    rgb_img = np.transpose(batch['rgb'].numpy()[0], [1, 2, 0])
    print(rgb_img.shape)
    depth = np.squeeze(np.transpose(batch['depth'].numpy()[0], [1, 2, 0]))
    segmentation = batch['label'].numpy()[0]

    segmentation_color = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for color_config in h5dataset.label_color_configs['5class']:
        segmentation_color[np.where(segmentation == int(color_config['label']))] = color_config['color']

    cv2.imshow('RGB', (rgb_img * 255.).astype(np.uint8))
    # cv2.imshow('RGB', rgb_img.astype(np.uint8))
    cv2.imshow('Map', depth2colormap(map_depth, 0.0, 1.0, invert=True))
    cv2.imshow('Depth', depth2colormap(depth, 0.0, 1.0, invert=True))
    cv2.imshow('Segmentation', segmentation_color)
    # cv2.imwrite('test/rgb_%06d.png'%(batch_itr), (rgb_img[0] * 255.).astype(np.uint8))
    # cv2.imwrite('test/map_%06d.png'%(batch_itr), depth2colormap(map_depth[0], 0.0, 1.0, invert=True))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
