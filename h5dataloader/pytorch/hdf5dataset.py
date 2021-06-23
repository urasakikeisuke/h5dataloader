# -*- coding: utf-8 -*-

from typing import Dict, List
import torch
from torch.utils.data import Dataset

from ..common.structure import *
from h5dataloader.pytorch.structure import CONVERT_NUMPY, CONVERT_TORCH, DTYPE_TORCH
from ..common import HDF5DatasetNumpy

class HDF5Dataset(Dataset, HDF5DatasetNumpy):
    """HDF5Dataset

    PyTorch用HDF5Dataset

    Args:
        h5_paths (List[str]): HDF5データセットのパスのリスト
        config (str): 設定ファイルのパス
        quiet (bool, optional): Trueの場合, 標準出力を行わない
        block_size (int, optional): データをブロックごとに区切って使用する際のサイズ. 0 の場合, 使用しない. Defaults to 0.
        use_mods (Tuple[int, int], optional): ブロック中で使用する範囲 (start, end). Defaults to None.
        visibility_filter_radius (int, optional): Defaults to 0.
        visibility_filter_threshold (float, optional): Defaults to 3.0.
    """
    def __init__(self, h5_paths:List[str]=[], config:str=None, quiet:bool=False, block_size:int=0, use_mods:Tuple[int, int]=None, visibility_filter_radius:int=0, visibility_filter_threshold:float=3.0) -> None:
        Dataset.__init__(self)
        HDF5DatasetNumpy.__init__(self, h5_paths, config, quiet, block_size, use_mods, visibility_filter_radius, visibility_filter_threshold)

    def __getitem__(self, index:int) -> dict:
        """__getitem__

        データセットからmini-batchを取得する

        Args:
            index (int): データの番号

        Returns:
            dict: mini-batch
        """
        link_idx = self.get_link_idx(index=index)
        hdf5_key = self.get_key(index=index)

        items:Dict[str, torch.Tensor] = {}
        for key, minibatch_config in self.minibatch.items():
            dataType = minibatch_config[CONFIG_TAG_TYPE]
            tensor_np:np.ndarray = DTYPE_TORCH[dataType](minibatch_config[CONFIG_TAG_CREATEFUNC](hdf5_key, link_idx, minibatch_config))
            items[key] = torch.from_numpy(CONVERT_TORCH[dataType](tensor_np))

        return items

    def to_numpy(self, tensor: torch.Tensor, tag: str) -> np.ndarray:
        """to_numpy

        Tensorをndarrayに変換する

        Args:
            tensor (torch.Tensor): (C, H, W)のテンソル
            tag (str): minibatchのタグ

        Returns:
            np.ndarray: (H, W[,C])の行列
        """
        if tag not in self.minibatch.keys():
            raise KeyError(f'"{tag}" is not in "minibatch".')
        dataType:str = self.minibatch[tag][CONFIG_TAG_TYPE]
        return CONVERT_NUMPY[dataType](tensor.cpu().detach().numpy())

    def __len__(self) -> int:
        """__len__

        データセットの長さを返す

        Returns:
            int: データセットの長さ
        """
        return super().__len__()
