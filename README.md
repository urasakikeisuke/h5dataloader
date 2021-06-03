# H5-DataLoader

A data loader for using H5Dataset with PyTorch

## Dependency

- Library
  - libpcl-dev
  - libopencv-dev
  - python3-numpy
- Python3
  - setuptools
  - wheel
  - scikit-build
  - cmake
  - ninja
  - numpy
  - h5py
  - opencv-python
  - pointsmap

## Install

```bash
pip3 install git+https://github.com/shikishima-TasakiLab/h5dataloader
```

## How to Use

### Config

See [H5-DataLoader-Config](https://github.com/shikishima-TasakiLab/h5dataloader-config).

### PyTorch

```python
import torch
from torch.utils.data.dataloader import DataLoader
from h5dataloader.pytorch import HDF5Dataset

device = torch.device('cuda:0' if args.cuda else 'cpu')

train_dataset = HDF5Dataset(h5paths=['train1.hdf5', 'train2.hdf5', ...], config='config.json')
train_dataloader = DataLoader(h5dataset, batch_size=2, shuffle=True)

for train_itr, train_batch in enumerate(train_dataloader):
    img_rgb = train_batch['rgb'].to(device)
    img_gt = train_batch['gt'].to(device)
    ...
```