# -*- coding: utf-8 -*-

from types import MethodType
import numpy as np
from typing import Any, List, Dict
import h5py
import json
import os
import uuid

from .create_funcs import CreateFuncs
from .structure import *

LINK_H5_DIR = '/tmp'

def parse_src_dst(label_dict:Dict[str, Dict[str, int]], quiet:bool=False) -> List[Dict[str, int]]:
    """parse_src_dst

    ラベル変換用の辞書を生成

    Args:
        label_dict (dict): ラベルの変換情報が記述された辞書
        quiet (bool, optional): Trueの場合, 標準出力を行わない

    Returns:
        list: 相互変換用の辞書のリスト
    """
    dst_list:List[Dict[str, int]] = []
    for key, item in label_dict[CONFIG_TAG_CONVERT].items():
        src_dst:Dict[str, int] = {}
        src_dst[CONFIG_TAG_SRC] = int(key)
        src_dst[CONFIG_TAG_DST] = item
        if quiet is False:
            print('{0:>3d} > {1:>3d}'.format(int(key), item))
        dst_list.append(src_dst)
    return dst_list

def parse_colors(label_dict:Dict[str, Dict[str, Dict[str, List[int]]]]) -> List[Dict[str, Union[int, np.ndarray]]]:
    """parse_colors

    ラベルをカラーへ変換する際の辞書を作成

    Args:
        label_dict (dict): ラベルの変換情報が記述された辞書

    Returns:
        List[Dict[str, Union[int, np.ndarray]]]: カラー変換用の辞書のリスト
    """
    dst_list:List[Dict[str, Union[int, np.ndarray]]] = []
    for key, item in label_dict[CONFIG_TAG_DST].items():
        dst_color:Dict[str, Union[int, np.ndarray]] = {}
        dst_color[CONFIG_TAG_LABEL] = int(key)
        dst_color[CONFIG_TAG_COLOR] = np.array(item[CONFIG_TAG_COLOR], dtype=np.uint8)
        dst_list.append(dst_color)
    return dst_list

def find_tf_tree_key(key:str, tf_tree:dict) -> Union[List[str], None]:
    """find_tf_tree_key

    TF-Treeから特定のキー(child_frame_id)を検索し, rootからのキーのリストを生成する

    Args:
        key (str): キー(child_frame_id)
        tf_tree (dict): 検索対象のTF-Tree

    Returns:
        Union[List[str], None]: rootからのキーのリスト. 見つからない場合はNone.
    """
    for tree_key, tree_item in tf_tree.items():
        if key == tree_key:
            return [tree_key]
        key_list = find_tf_tree_key(key, tree_item)
        if isinstance(key_list, list):
            key_list.insert(0, tree_key)
            return key_list

class HDF5DatasetNumpy(CreateFuncs):
    """HDF5DatasetNumpy

    HDF5DatasetのBaseとなるクラス

    Args:
        h5_paths (List[str]): HDF5データセットのパスのリスト
        config (str): 設定ファイルのパス
        quiet (bool, optional): Trueの場合, 標準出力を行わない
    """

    def __init__(self, h5_paths:List[str]=[], config:str=None, quiet:bool=False,
        block_size:int=0, use_mods:Tuple[int, int]=None, visibility_filter_radius:int=0, visibility_filter_threshold:float=3.0) -> None:
        """__init__

        コンストラクタ

        Args:
            h5_paths (List[str]): HDF5データセットのパスのリスト
            config (str): 設定ファイルのパス
            quiet (bool, optional): Trueの場合, 標準出力を行わない
            block_size (int, optional): データをブロックごとに区切って使用する際のサイズ. 0 の場合, 使用しない. Defaults to 0.
            use_mods (Tuple[int, int], optional): ブロック中で使用する範囲 (start, end). Defaults to None.
            visibility_filter_radius (int, optional): Defaults to 0.
            visibility_filter_threshold (float, optional): Defaults to 3.0.
        """
        super(HDF5DatasetNumpy, self).__init__(quiet=quiet, visibility_filter_radius=visibility_filter_radius, visibility_filter_threshold=visibility_filter_threshold)

        # configファイルが見つからない場合に終了
        if os.path.isfile(config) is False:
            print('File not found : "%s"'%(config))
            exit(1)

        self.block_size = block_size
        if self.block_size > 0:
            if isinstance(use_mods, tuple) is False:
                print('"use_mods" must be tuple (start, end).')
                exit(1)
            if isinstance(use_mods[0], int) is False:
                print('"use_mods[0]" must be int.')
                exit(1)
            if isinstance(use_mods[1], int) is False:
                print('"use_mods[1]" must be int.')
                exit(1)
            self.use_mods_start = use_mods[0]
            self.use_mods_end = use_mods[1]
            if self.block_size < self.use_mods_end:
                self.use_mods_end = self.block_size
            if self.use_mods_end - self.use_mods_start <= 0:
                print('use_mod[0] < use_mods[1]')
                exit(1)
            self.block_use_len = self.use_mods_end - self.use_mods_start

        # configファイルをロード -> 辞書型へ
        config_dict:Dict[str, Dict[str, Dict[str, Any]]] = {}
        with open(config, mode='r') as configfile:
            config_dict = json.load(configfile)

        # ラベルの設定の辞書を生成
        if CONFIG_TAG_LABEL in config_dict:
            for key_label, item_label in config_dict[CONFIG_TAG_LABEL][CONFIG_TAG_CONFIG].items():
                print('Label "%s"'%(key_label))
                self.label_convert_configs[key_label] = parse_src_dst(item_label, quiet=quiet)
                self.label_color_configs[key_label] = parse_colors(item_label)

        # HDF5ファイルのパスのリスト
        self.h5_paths:List[str] = h5_paths
        self.h5_paths.sort()

        # 複数のHDF5を扱うための一時ファイルを生成
        self.length:int = 0
        start_idxs:List[int] = []
        self.h5link_path = os.path.join(LINK_H5_DIR, 'h5dataloader-' + str(uuid.uuid4()) + '-link.hdf5')
        with h5py.File(self.h5link_path, mode='w') as h5link:
            h5_len_tmp:int = 0
            for link_cnt, h5_path in enumerate(self.h5_paths):
                if os.path.isfile(h5_path) is False:
                    print('Not Found : "%s"', h5_path)
                    exit(1)
                h5link[str(link_cnt)] = h5py.ExternalLink(h5_path, '/') # 番号からlinkを作成
                start_idxs.append(h5_len_tmp)
                with h5py.File(h5_path, mode='r') as h5file:
                    h5_len_tmp += h5file['header/length'][()]
                link_cnt += 1
            if block_size > 0:
                self.length = h5_len_tmp // self.block_size * (self.use_mods_end - self.use_mods_start)
            else:
                self.length = h5_len_tmp
        self.start_idxs = np.array(start_idxs)

        # 一時ファイルを開く
        self.h5links = h5py.File(self.h5link_path, mode='r')

        # tfの設定
        self.tf = config_dict[CONFIG_TAG_TF]

        # データセットの設定
        self.minibatch:Dict[str, Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]] = {}     # mini-batchの設定

        for key, item in config_dict[CONFIG_TAG_MINIBATCH].items():
            data_dict:Dict[str, Dict[str, Any]] = {}
            data_dict[CONFIG_TAG_FROM] = item.get(CONFIG_TAG_FROM)
            data_dict[CONFIG_TAG_TYPE] = item.get(CONFIG_TAG_TYPE)

            if data_dict[CONFIG_TAG_FROM] is None or data_dict[CONFIG_TAG_TYPE] is None:
                print('keys "from" and "type" must not be null')
                exit(1)

            if isinstance(item.get(CONFIG_TAG_SHAPE), list) is True:
                data_dict[CONFIG_TAG_SHAPE] = tuple(item[CONFIG_TAG_SHAPE])
            else:
                data_dict[CONFIG_TAG_SHAPE] = None

            if isinstance(item.get(CONFIG_TAG_RANGE), list) is True:
                depth_range = []
                depth_range.append(0.0 if item[CONFIG_TAG_RANGE][0] < 0.0 else item[CONFIG_TAG_RANGE][0])
                depth_range.append(np.inf if item[CONFIG_TAG_RANGE][1] is None else item[CONFIG_TAG_RANGE][1])
                data_dict[CONFIG_TAG_RANGE] = tuple(depth_range)
            else:
                data_dict[CONFIG_TAG_RANGE] = DEFAULT_RANGE[data_dict[CONFIG_TAG_TYPE]]

            data_dict[CONFIG_TAG_NORMALIZE] = item.get(CONFIG_TAG_NORMALIZE)
            data_dict[CONFIG_TAG_FRAMEID] = item.get(CONFIG_TAG_FRAMEID)
            data_dict[CONFIG_TAG_LABELTAG] = item.get(CONFIG_TAG_LABELTAG)

            tf_from:str = data_dict[CONFIG_TAG_FROM].get(TYPE_POSE)
            tf_to:str = data_dict[CONFIG_TAG_FRAMEID]

            tf_calc:List[Tuple[str, bool]] = []

            if tf_from is not None and tf_to is not None:
                tf_from_list:List[str] = find_tf_tree_key(tf_from, self.tf[CONFIG_TAG_TREE])
                tf_to_list:List[str] = find_tf_tree_key(tf_to, self.tf[CONFIG_TAG_TREE])

                while True:
                    if len(tf_from_list) < 1 or len(tf_to_list) < 1: break
                    if tf_from_list[0] != tf_to_list[0]: break
                    tf_from_list.pop(0)
                    tf_to_list.pop(0)
                tf_from_list.reverse()

                for tf_from_str in tf_from_list:
                    tf_calc.append((tf_from_str, True))
                for tf_to_str in tf_to_list:
                    tf_calc.append((tf_to_str, False))

            data_dict[CONFIG_TAG_TF] = tf_calc

            data_dict[CONFIG_TAG_CREATEFUNC] = self.bind_createFunc(data_dict)

            self.minibatch[key] = data_dict

    def get_link_idx(self, index:int) -> int:
        """get_link_idx

        与えられたindexから各HDF5ファイルへのリンクのインデックスを生成する

        Args:
            index (int): PyTorchのDataLoaderで生成されるindex

        Returns:
            int: 各HDF5ファイルへのリンクのインデックス
        """
        idx:int = index
        if self.block_size > 0:
            idx = index // self.block_use_len * self.block_size + self.use_mods_start + index % self.block_use_len
        return np.where(self.start_idxs <= idx)[0][-1]

    def get_key(self, index:int) -> str:
        """get_key

        与えられたindexからHDF5データセットへアクセスするためのキーを生成する

        Args:
            index (int): PyTorchのDataLoaderで生成されるindex

        Returns:
            str: HDF5データセットへアクセスするためのキー ( [LINK]/data/[INDEX] )
        """
        idx:int = index
        link_idx:int = self.get_link_idx(idx)
        if self.block_size > 0:
            idx = index // self.block_use_len * self.block_size + self.use_mods_start + index % self.block_use_len
        return str(link_idx) + '/data/' + str(idx - self.start_idxs[link_idx])

    def keys(self):
        return self.minibatch.keys()

    def items(self):
        return self.minibatch.items()

    def __getitem__(self, index: int) -> dict:
        """__getitem__

        データセットからmini-batchを取得する

        Args:
            index (int): データの番号
        Returns:
            dict: mini-batch
        """
        link_idx = self.get_link_idx(index=index)
        hdf5_key = self.get_key(index=index)

        items:Dict[str, np.ndarray] = {}
        for key, minibatch_config in self.minibatch.items():
            items[key] = minibatch_config[CONFIG_TAG_CREATEFUNC](hdf5_key, link_idx, minibatch_config)

        return items


    def __len__(self) -> int:
        """__len__

        データセットの長さを返す

        Returns:
            int: データセットの長さ
        """
        return self.length

def hwc2chw(hwc: np.ndarray) -> np.ndarray:
    return hwc.transpose([2, 0, 1])

def hw2chw(hw: np.ndarray) -> np.ndarray:
    return hw[np.newaxis, :, :]

def chw2hw(chw: np.ndarray) -> np.ndarray:
    return chw.squeeze(axis=0)

def chw2hwc(chw: np.ndarray) -> np.ndarray:
    return chw.transpose([1, 2, 0])

def nchw2nhwc(nchw: np.ndarray) -> np.ndarray:
    return nchw.transpose([0, 2, 3, 1])

def nchw2nhw(nchw: np.ndarray) -> np.ndarray:
    return nchw.squeeze(axis=1)

def nochange(array: Any) -> Any:
    return array
