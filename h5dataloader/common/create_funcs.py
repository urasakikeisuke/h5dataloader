# -*- coding: utf-8 -*-

import os
from types import MethodType
from typing import Any, Dict, List, Union
import numpy as np
import cv2
import h5py

from pointsmap import Points, VoxelGridMap, invertTransform, combineTransforms, Depth
from .structure import *

NORMALIZE_INF = 2.0

class CreateFuncs():
    def __init__(self, quiet:bool=True, visibility_filter_radius:int=0, visibility_filter_threshold:float=3.0) -> None:
        """__init__

        コンストラクタ

        Args:
            visibility_filter_radius (int, optional): Visibility Filterにおけるカーネル半径. Defaults to 0.
            visibility_filter_threshold (float, optional): Visibility Filterにおける閾値. Defaults to 3.0.
            quiet (bool, optional): Trueの場合, ERROR, WARNING以外の標準出力を行わない
        """
        # データセットの設定
        self.h5links:h5py.File = None
        self.label_convert_configs:Dict[str, List[Dict[str, int]]] = {}                     # ラベルの変換の設定を格納した辞書
        self.label_color_configs:Dict[str, List[Dict[str, Union[int, np.ndarray]]]] = {}    # ラベルの色の設定を格納した辞書
        self.link_maps:Dict[str, str] = {}                                                  # linkと地図を紐づけする辞書
        self.maps:Dict[str, VoxelGridMap] = {}                                              # 地図とVoxelGridMapを紐づけする辞書
        self.tf:Dict[str, dict] = {}                                                        # TFの設定を格納した辞書
        self.visibility_filter_radius = visibility_filter_radius                            # Visibility Filterにおけるカーネル半径
        self.visibility_filter_threshold = visibility_filter_threshold                      # Visibility Filterにおける閾値
        self.quiet:bool = quiet                                                             # ERROR, WARNING以外の標準出力をしない

    def bind_createFunc(self, config:Dict[str, Union[bool, str, Dict[str, str], List[Union[int, float]], None]]) -> MethodType:
        """bind_createFunc

        Args:
            config (Dict[str, Union[bool, str, Dict[str, str], List[Union[int, float]], None]]): mini-batchの設定

        Returns:
            MethodType: create funcition
        """
        if config.get(CONFIG_TAG_TYPE) is None:
            print('keys "type" must not be null')
            exit(1)
        if config.get(CONFIG_TAG_FROM) is None:
            print('keys "from" must not be null')
            exit(1)

        dst_type:str = config.get(CONFIG_TAG_TYPE)
        froms:Dict[str, str] = config.get(CONFIG_TAG_FROM)

        froms_keys:set = set(froms.keys())
        if len(froms_keys - {TYPE_POSE}) > 0:
            froms_keys -= {TYPE_POSE}

        if dst_type in [TYPE_UINT8, TYPE_INT8, TYPE_INT16, TYPE_INT32, TYPE_INT64, TYPE_FLOAT16, TYPE_FLOAT32, TYPE_FLOAT64]:
            if froms_keys <= {TYPE_UINT8, TYPE_INT8, TYPE_INT16, TYPE_INT32, TYPE_INT64, TYPE_FLOAT16, TYPE_FLOAT32, TYPE_FLOAT64}: return self.create_number
        elif dst_type == TYPE_MONO8:
            if froms_keys <= {TYPE_MONO8}: return self.create_mono8_from_mono8
            elif froms_keys <= {TYPE_MONO16}: return self.create_mono8_from_mono16
            elif froms_keys <= {TYPE_BGR8}: return self.create_mono8_from_bgr8
            elif froms_keys <= {TYPE_RGB8}: return self.create_mono8_from_rgb8
            elif froms_keys <= {TYPE_BGRA8}: return self.create_mono8_from_bgra8
            elif froms_keys <= {TYPE_RGBA8}: return self.create_mono8_from_rgba8
        elif dst_type == TYPE_MONO16:
            if froms_keys <= {TYPE_MONO8}: return self.create_mono16_from_mono8
            elif froms_keys <= {TYPE_MONO16}: return self.create_mono16_from_mono16
            elif froms_keys <= {TYPE_BGR8}: return self.create_mono16_from_bgr8
            elif froms_keys <= {TYPE_RGB8}: return self.create_mono16_from_rgb8
            elif froms_keys <= {TYPE_BGRA8}: return self.create_mono16_from_bgra8
            elif froms_keys <= {TYPE_RGBA8}: return self.create_mono16_from_rgba8
        elif dst_type == TYPE_BGR8:
            if froms_keys <= {TYPE_MONO8}: return self.create_bgr8_from_mono8
            elif froms_keys <= {TYPE_MONO16}: return self.create_bgr8_from_mono16
            elif froms_keys <= {TYPE_BGR8}: return self.create_bgr8_from_bgr8
            elif froms_keys <= {TYPE_RGB8}: return self.create_bgr8_from_rgb8
            elif froms_keys <= {TYPE_BGRA8}: return self.create_bgr8_from_bgra8
            elif froms_keys <= {TYPE_RGBA8}: return self.create_bgr8_from_rgba8
            elif froms_keys <= {TYPE_SEMANTIC2D}: return self.create_bgr8_from_semantic2d
            elif froms_keys <= {TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_bgr8_from_semantic3d
            elif froms_keys <= {TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_bgr8_from_voxelsemantic3d
        elif dst_type == TYPE_RGB8:
            if froms_keys <= {TYPE_MONO8}: return self.create_rgb8_from_mono8
            elif froms_keys <= {TYPE_MONO16}: return self.create_rgb8_from_mono16
            elif froms_keys <= {TYPE_BGR8}: return self.create_rgb8_from_bgr8
            elif froms_keys <= {TYPE_RGB8}: return self.create_rgb8_from_rgb8
            elif froms_keys <= {TYPE_BGRA8}: return self.create_rgb8_from_bgra8
            elif froms_keys <= {TYPE_RGBA8}: return self.create_rgb8_from_rgba8
            elif froms_keys <= {TYPE_SEMANTIC2D}: return self.create_rgb8_from_semantic2d
            elif froms_keys <= {TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_rgb8_from_semantic3d
            elif froms_keys <= {TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_rgb8_from_voxelsemantic3d
        elif dst_type == TYPE_BGRA8:
            if froms_keys <= {TYPE_MONO8}: return self.create_bgra8_from_mono8
            elif froms_keys <= {TYPE_MONO16}: return self.create_bgra8_from_mono16
            elif froms_keys <= {TYPE_BGR8}: return self.create_bgra8_from_bgr8
            elif froms_keys <= {TYPE_RGB8}: return self.create_bgra8_from_rgb8
            elif froms_keys <= {TYPE_BGRA8}: return self.create_bgra8_from_bgra8
            elif froms_keys <= {TYPE_RGBA8}: return self.create_bgra8_from_rgba8
            elif froms_keys <= {TYPE_SEMANTIC2D}: return self.create_bgra8_from_semantic2d
            elif froms_keys <= {TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_bgra8_from_semantic3d
            elif froms_keys <= {TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_bgra8_from_voxelsemantic3d
        elif dst_type == TYPE_RGBA8:
            if froms_keys <= {TYPE_MONO8}: return self.create_rgba8_from_mono8
            elif froms_keys <= {TYPE_MONO16}: return self.create_rgba8_from_mono16
            elif froms_keys <= {TYPE_BGR8}: return self.create_rgba8_from_bgr8
            elif froms_keys <= {TYPE_RGB8}: return self.create_rgba8_from_rgb8
            elif froms_keys <= {TYPE_BGRA8}: return self.create_rgba8_from_bgra8
            elif froms_keys <= {TYPE_RGBA8}: return self.create_rgba8_from_rgba8
            elif froms_keys <= {TYPE_SEMANTIC2D}: return self.create_rgba8_from_semantic2d
            elif froms_keys <= {TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_rgba8_from_semantic3d
            elif froms_keys <= {TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_rgba8_from_voxelsemantic3d
        elif dst_type == TYPE_DEPTH:
            if froms_keys <= {TYPE_DEPTH}: return self.create_depth_from_depth
            elif froms_keys <= {TYPE_DISPARITY, TYPE_INTRINSIC}: return self.create_depth_from_disparity
            elif froms_keys <= {TYPE_POINTS, TYPE_INTRINSIC, TYPE_POSE}:
                tmp_link:h5py.Group = self.h5links['0']
                tmp_points:h5py.Dataset = tmp_link.get(froms[TYPE_POINTS])
                if tmp_points is None:
                    tmp_map_id = None
                else:
                    tmp_map_id:Union[str, None] = self.convert_str(tmp_points.attrs.get(H5_ATTR_MAPID))
                if tmp_map_id is not None:
                    for key_link, item_link in self.h5links.items():
                        map_points:h5py.Dataset = item_link.get(froms[TYPE_POINTS])
                        if map_points is None:
                            print('key "%s" is not exist'%(froms[TYPE_POINTS]))
                            exit(1)
                        elif isinstance(map_points, h5py.Dataset) is False:
                            print('key "%s" is not "h5py.Dataset"'%(froms[TYPE_POINTS]))
                            exit(1)
                        map_id = self.convert_str(map_points.attrs.get(H5_ATTR_MAPID))
                        if map_id is None:
                            print('attribute "map_id" is not exist')
                            exit(1)
                        self.link_maps[key_link] = map_id
                        if map_id in self.maps.keys(): continue

                        vgm = VoxelGridMap(quiet=self.quiet)
                        intrinsic:np.ndarray = self.create_intrinsic_array(key_link, int(key_link), config)
                        vgm.set_intrinsic(intrinsic)
                        vgm.set_shape(config[CONFIG_TAG_SHAPE])
                        vgm.set_depth_range(config[CONFIG_TAG_RANGE])
                        vgm.set_pointsmap(map_points[()])
                        self.maps[map_id] = vgm
                    return self.create_depth_from_pointsmap
                else:
                    return self.create_depth_from_points
            elif froms_keys <= {TYPE_SEMANTIC3D, TYPE_INTRINSIC, TYPE_POSE}:
                tmp_link:h5py.Group = self.h5links['0']
                tmp_semantic3d:h5py.Group = tmp_link.get(froms[TYPE_SEMANTIC3D])
                if tmp_semantic3d is None:
                    tmp_map_id = None
                else:
                    tmp_map_id:Union[str, None] = self.convert_str(tmp_semantic3d.attrs.get(H5_ATTR_MAPID))
                if tmp_map_id is not None:
                    for key_link, item_link in self.h5links.items():
                        map_points:h5py.Group = item_link.get(froms[TYPE_SEMANTIC3D])
                        if map_points is None:
                            print('key "%s" is not exist'%(froms[TYPE_SEMANTIC3D]))
                            exit(1)
                        elif isinstance(map_points, h5py.Group) is False:
                            print('key "%s" is not "h5py.Group"'%(froms[TYPE_SEMANTIC3D]))
                            exit(1)
                        map_id = self.convert_str(map_points.attrs.get(H5_ATTR_MAPID))
                        if map_id is None:
                            print('attribute "map_id" is not exist')
                            exit(1)
                        self.link_maps[key_link] = map_id
                        if map_id in self.maps.keys(): continue

                        vgm = VoxelGridMap(quiet=self.quiet)
                        intrinsic:np.ndarray = self.create_intrinsic_array(key_link, int(key_link), config)
                        vgm.set_intrinsic(intrinsic)
                        vgm.set_shape(config[CONFIG_TAG_SHAPE])
                        vgm.set_depth_range(config[CONFIG_TAG_RANGE])
                        vgm.set_semanticmap(map_points[TYPE_POINTS][()], map_points[TYPE_SEMANTIC1D][()])
                        self.maps[map_id] = vgm
                    return self.create_depth_from_pointsmap
                else:
                    return self.create_depth_from_semantic3d
            elif froms_keys <= {TYPE_VOXEL_POINTS, TYPE_POSE, TYPE_INTRINSIC}: return self.create_depth_from_voxelpoints
            elif froms_keys <= {TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_depth_from_voxelsemantic3d
        elif dst_type == TYPE_POINTS:
            if froms_keys <= {TYPE_POINTS, TYPE_POSE}:
                raise NotImplementedError()
            elif froms_keys <= {TYPE_DEPTH, TYPE_INTRINSIC, TYPE_POSE}:
                raise NotImplementedError()
            elif froms_keys <= {TYPE_SEMANTIC3D}:
                raise NotImplementedError()
        elif dst_type == TYPE_SEMANTIC1D:
            if froms_keys <= {TYPE_SEMANTIC1D}:
                raise NotImplementedError()
            elif froms_keys <= {TYPE_SEMANTIC3D}:
                raise NotImplementedError()
        elif dst_type == TYPE_SEMANTIC2D:
            if froms_keys <= {TYPE_SEMANTIC2D}: return self.create_semantic2d_from_sematic2d
            elif froms_keys <= {TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_semantic2d_from_semantic3d
            elif froms_keys <= {TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC}: return self.create_semantic2d_from_voxelsemantic3d
        elif dst_type == TYPE_SEMANTIC3D:
            if froms_keys <= {TYPE_SEMANTIC3D, TYPE_POSE}:
                raise NotImplementedError()
            elif froms_keys <= {TYPE_SEMANTIC1D, TYPE_POINTS}:
                raise NotImplementedError()
            elif froms_keys <= {TYPE_SEMANTIC2D, TYPE_POINTS, TYPE_INTRINSIC, TYPE_POSE}:
                raise NotImplementedError()
            elif froms_keys <= {TYPE_SEMANTIC2D, TYPE_DEPTH, TYPE_INTRINSIC, TYPE_POSE}:
                raise NotImplementedError()
            elif froms_keys <= {TYPE_VOXEL_SEMANTIC3D, TYPE_POSE}:
                raise NotImplementedError()
        elif dst_type == TYPE_POSE:
            if froms_keys <= {TYPE_POSE}: return self.create_pose_from_pose
            elif froms_keys <= {TYPE_TRANSLATION, TYPE_QUATERNION}: return self.create_pose_from_translation_quaternion
        elif dst_type == TYPE_TRANSLATION:
            if froms_keys <= {TYPE_TRANSLATION}: return self.create_translation_from_translation
            elif froms_keys <= {TYPE_POSE}: return self.create_translation_from_pose
        elif dst_type == TYPE_QUATERNION:
            if froms_keys <= {TYPE_QUATERNION}: return self.create_quaternion_from_quaternion
            elif froms_keys <= {TYPE_POSE}: return self.create_quaternion_from_pose
        elif dst_type == TYPE_INTRINSIC:
            if froms_keys <= {TYPE_INTRINSIC}: return self.create_intrinsic_array
        elif dst_type == TYPE_COLOR:
            if froms_keys <= {TYPE_COLOR}: return self.create_color

        raise TypeError('%s cannot convert to "%s"'%(str(list(froms_keys)), dst_type))

    def create_h5_key(self, data_type:str, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> str:
        """create_h5_key

        データを読み出す際のキーを生成する

        Args:
            data_type (str): データの型 (TYPE_...)
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            str: データを読み出す際のキー
        """
        h5_key:str = minibatch_config[CONFIG_TAG_FROM][data_type]
        if h5_key[0] == '/':
            return str(link_idx) + h5_key
        else:
            return os.path.join(key, h5_key)

    def image_common(self, src:np.ndarray, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """image_common

        画像生成時の共通処理

        Args:
            src (np.ndarray): 画像
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: srcと同じ型の画像. 正規化した場合は"np.float32"の画像
        """
        dst:np.ndarray = src

        shape:tuple = minibatch_config[CONFIG_TAG_SHAPE]
        if shape[:2] != src.shape[:2]:
            dst = cv2.resize(src, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

        dst_range = minibatch_config.get(CONFIG_TAG_RANGE)
        if dst_range is None:
            print('"range" is not defined.')
            exit(1)

        dst_type:str = minibatch_config[CONFIG_TAG_TYPE]

        if minibatch_config[CONFIG_TAG_NORMALIZE] is True:
            dst = (dst - dst_range[0]) / (dst_range[1] - dst_range[0])
            dst = dst.astype(np.float32)
        elif dst_range != DEFAULT_RANGE[dst_type]:
            len_shape = len(dst.shape)
            if len_shape == 3:
                target = np.zeros_like(dst[:,:,0], dtype=np.bool)
                for i in range(dst.shape[2]):
                    target |= dst[:,:,i] < dst_range[0]
                    target |= dst_range[1] < dst[:,:,i]
                dst[np.where(target)] = ZERO_VALUE[dst_type]
            elif len_shape == 2:
                dst = np.where((dst < dst_range[0]) | (dst_range[1] < dst), ZERO_VALUE[dst_type], dst)
            else:
                print('length of "shape" must be 2 or 3.')
                exit(1)

        return dst

    def create_number(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.dtype:
        """create_number

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.dtype: 数値
        """
        from_type = set(minibatch_config[CONFIG_TAG_FROM].keys()).pop()
        h5_key:str = self.create_h5_key(from_type, key, link_idx, minibatch_config)
        dst = self.h5links[h5_key][()]

        dst_range = minibatch_config.get(CONFIG_TAG_RANGE)
        if dst_range is None:
            print('"range" is not defined.')
            exit(1)

        if minibatch_config[CONFIG_TAG_NORMALIZE] is True:
            dst = np.float32((dst - dst_range[1]) / (dst_range[1] - dst_range[0]))
        else:
            dst = DTYPE_NUMPY[from_type](dst_range[0]) if dst < dst_range[0] else DTYPE_NUMPY[from_type](dst)
            dst = DTYPE_NUMPY[from_type](dst_range[1]) if dst_range[1] < dst else DTYPE_NUMPY[from_type](dst)
        return dst

    def create_mono8_from_mono8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono8_from_mono8

        'mono8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono8'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO8, key, link_idx, minibatch_config)
        dst = self.h5links[h5_key][()]
        return self.image_common(dst, minibatch_config)

    def create_mono8_from_mono16(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono8_from_mono16

        'mono16'の画像から'mono8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono8'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO16, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = src / 257.
        return self.image_common(dst.astype(np.uint8), minibatch_config)

    def create_mono8_from_bgr8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono8_from_bgr8

        'bgr8'の画像から'mono8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono8'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return self.image_common(dst, minibatch_config)

    def create_mono8_from_rgb8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono8_from_rgb8

        'rgb8'の画像から'mono8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono8'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGB8, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        return self.image_common(dst, minibatch_config)

    def create_mono8_from_bgra8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono8_from_bgra8

        'bgra8'の画像から'mono8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono8'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGRA8, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        return self.image_common(dst, minibatch_config)

    def create_mono8_from_rgba8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono8_from_rgba8

        'rgba8'の画像から'mono8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono8'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGBA8, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)
        return self.image_common(dst, minibatch_config)

    def create_mono16_from_mono8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono16_from_mono8

        'mono8'の画像から'mono16'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono16'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = src * 257
        return self.image_common(dst, minibatch_config)

    def create_mono16_from_mono16(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono16_from_mono16

        'mono16'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono16'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO16, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        return self.image_common(src, minibatch_config)

    def create_mono16_from_bgr8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono16_from_bgr8

        'bgr8'の画像から'mono16'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono16'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return self.image_common(dst * 257, minibatch_config)

    def create_mono16_from_rgb8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono16_from_rgb8

        'rgb8'の画像から'mono16'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono16'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGB8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        return self.image_common(dst * 257, minibatch_config)

    def create_mono16_from_bgra8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono16_from_bgra8

        'bgra8'の画像から'mono16'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono16'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGRA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        return self.image_common(dst * 257, minibatch_config)

    def create_mono16_from_rgba8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_mono16_from_rgba8

        'rgba8'の画像から'mono16'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'mono16'の画像, 正規化した場合は"np.float32"の画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGBA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)
        return self.image_common(dst * 257, minibatch_config)

    def create_bgr8_from_mono8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_mono8

        'mono8'の画像から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        return self.image_common(dst, minibatch_config)

    def create_bgr8_from_mono16(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_mono16

        'mono16'の画像から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO16, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()] / 257.
        dst:np.ndarray = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return self.image_common(dst, minibatch_config)

    def create_bgr8_from_bgr8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_bgr8

        'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        return self.image_common(src, minibatch_config)

    def create_bgr8_from_rgb8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_rgb8

        'rgb8'の画像から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGB8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        return self.image_common(dst, minibatch_config)

    def create_bgr8_from_bgra8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_bgra8

        'bgra8'の画像から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGRA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
        return self.image_common(dst, minibatch_config)

    def create_bgr8_from_rgba8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_rgba8

        'rgba8'の画像から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGBA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGBA2BGR)
        return self.image_common(dst, minibatch_config)

    def create_bgr8_from_semantic2d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_semantic2d

        'semantic2d'の画像から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.convert_label(src, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_bgr8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_SEMANTIC2D, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.convert_label(src, minibatch_config[CONFIG_TAG_LABELTAG])
        dst:np.ndarray = self.convert_semantic2d_to_bgr8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_bgr8_from_semantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_semantic3d

        'semantic3d'の点群から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_bgr8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_bgr8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_bgr8_from_voxelsemantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgr8_from_voxelsemantic3d

        'voxel-semantic3d'の点群から'bgr8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgr8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_bgr8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_bgr8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_rgb8_from_mono8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_mono8

        'mono8'の画像から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_bgr8_from_mono8(key, link_idx, minibatch_config)
        """
        return self.create_bgr8_from_mono8(key, link_idx, minibatch_config)

    def create_rgb8_from_mono16(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_mono16

        'mono16'の画像から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_bgr8_from_mono16(key, link_idx, minibatch_config)
        """
        return self.create_bgr8_from_mono16(key, link_idx, minibatch_config)

    def create_rgb8_from_bgr8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_bgr8

        'bgr8'の画像から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        return self.image_common(dst, minibatch_config)

    def create_rgb8_from_rgb8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_rgb8

        'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGB8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        return self.image_common(src, minibatch_config)

    def create_rgb8_from_bgra8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_bgra8

        'bgra8'の画像から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGRA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGRA2RGB)
        return self.image_common(dst, minibatch_config)

    def create_rgb8_from_rgba8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_rgba8

        'rgba8'の画像から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGBA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGBA2RGB)
        return self.image_common(dst, minibatch_config)

    def create_rgb8_from_semantic2d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_semantic2d

        'semantic2d'の画像から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.convert_label(src, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_rgb8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_SEMANTIC2D, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.convert_label(src, minibatch_config[CONFIG_TAG_LABELTAG])
        dst:np.ndarray = self.convert_semantic2d_to_rgb8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_rgb8_from_semantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_semantic3d

        'semantic3d'の点群から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_rgb8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_rgb8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_rgb8_from_voxelsemantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgb8_from_voxelsemantic3d

        'voxel-semantic3d'の点群から'rgb8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgb8'の画像, 正規化した場合は"np.float32"の3ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_rgb8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_rgb8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_mono8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_mono8

        'mono8'の画像から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_GRAY2BGRA)
        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_mono16(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_mono16

        'mono16'の画像から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO16, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()] / 257.
        dst:np.ndarray = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_GRAY2BGRA)
        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_bgr8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_bgr8

        'bgr8'の画像から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_rgb8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_rgb8

        'rgb8'の画像から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGB8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGB2BGRA)
        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_bgra8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_bgra8

        'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGRA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        return self.image_common(src, minibatch_config)

    def create_bgra8_from_rgba8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_rgb8

        'rgba8'の画像から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGBA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGBA2BGRA)
        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_semantic2d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_semantic2d

        'semantic2d'の画像から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.convert_label(src, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_bgra8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_SEMANTIC2D, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.convert_label(src, minibatch_config[CONFIG_TAG_LABELTAG])
        dst:np.ndarray = self.convert_semantic2d_to_bgra8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_semantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_semantic3d

        'semantic3d'の点群から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_bgra8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_bgra8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_bgra8_from_voxelsemantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_bgra8_from_voxelsemantic3d

        'voxel-semantic3d'の点群から'bgra8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'bgra8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_bgra8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_bgra8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_mono8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_mono8

        'mono8'の画像から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_GRAY2RGBA)
        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_mono16(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_mono16

        'mono16'の画像から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_MONO16, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()] / 257.
        dst:np.ndarray = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_GRAY2RGBA)
        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_bgr8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_bgr8

        'bgr8'の画像から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGR2RGBA)
        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_rgb8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_rgb8

        'rgb8'の画像から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGB8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_RGB2RGBA)
        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_bgra8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_bgra8

        'bgra8'の画像から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_BGRA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        dst:np.ndarray = cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA)
        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_rgba8(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_rgb8

        'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_RGBA8, key, link_idx, minibatch_config)

        src:np.ndarray = self.h5links[h5_key][()]
        return self.image_common(src, minibatch_config)

    def create_rgba8_from_semantic2d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_semantic2d

        'semantic2d'の画像から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.convert_label(src, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_rgba8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_SEMANTIC2D, key, link_idx, minibatch_config)
        src:np.ndarray = self.h5links[h5_key][()]
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.convert_label(src, label_tag)
        dst:np.ndarray = self.convert_semantic2d_to_rgba8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_semantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_semantic3d

        'semantic3d'の点群から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_rgba8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_rgba8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def create_rgba8_from_voxelsemantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_rgba8_from_voxelsemantic3d

        'voxel-semantic3d'の点群から'rgba8'の画像を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'rgba8'の画像, 正規化した場合は"np.float32"の4ch画像

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.convert_semantic2d_to_rgba8(src, label_tag)
            HDF5Dataset.image_common(src, minibatch_config)
        """
        label_tag:str = minibatch_config[CONFIG_TAG_LABELTAG]

        if label_tag not in self.label_convert_configs.keys():
            print(f'"{label_tag}" is not in "/label"')
            exit(1)

        tmp:np.ndarray = self.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
        dst:np.ndarray = self.convert_semantic2d_to_rgba8(tmp, label_tag)

        return self.image_common(dst, minibatch_config)

    def semantic2d_common(self, src:np.ndarray, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """semantic2d_common

        'semantic2d'画像生成時の共通処理

        Args:
            src (np.ndarray): 画像
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'semantic2d'画像
        """
        dst:np.ndarray = src
        shape = minibatch_config[CONFIG_TAG_SHAPE]
        if shape != src.shape[:2]:
            dst = cv2.resize(src, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return dst

    def create_semantic2d_from_sematic2d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_semantic2d_from_sematic2d

        Semanticラベルを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: Semanticラベル

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.convert_label(src, minibatch_config)
            HDF5Dataset.semantic2d_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_SEMANTIC2D, key, link_idx, minibatch_config)
        dst:np.ndarray = self.h5links[h5_key][()]

        if minibatch_config[CONFIG_TAG_LABELTAG] in self.label_convert_configs.keys():
            dst = self.convert_label(dst, minibatch_config[CONFIG_TAG_LABELTAG])

        return self.semantic2d_common(dst, minibatch_config)

    def common_semantic2d_from_semantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """common_semantic2d_from_semantic3d

        Semantic3dからSemantic2dを生成する時の共通処理

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: Semanticラベル

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.create_intrinsic_array(key, link_idx, minibatch_config)
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
            HDF5Dataset.convert_label(src, minibatch_config)
        """
        semantic3d_key:str = self.create_h5_key(TYPE_SEMANTIC3D, key, link_idx, minibatch_config)

        pts = Points(quiet=self.quiet)
        pts.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        pts.set_shape(minibatch_config[CONFIG_TAG_SHAPE])
        pts.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        pts.set_semanticpoints(self.h5links[semantic3d_key][SUBTYPE_POINTS][()], self.h5links[semantic3d_key][SUBTYPE_SEMANTIC1D][()])

        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)

        dst:np.ndarray = pts.create_semantic2d(translation=pose[:3], quaternion=pose[-4:], filter_radius=self.visibility_filter_radius, filter_threshold=self.visibility_filter_threshold)

        if minibatch_config[CONFIG_TAG_LABELTAG] in self.label_convert_configs.keys():
            dst = self.convert_label(dst, minibatch_config[CONFIG_TAG_LABELTAG])

        return dst

    def create_semantic2d_from_semantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_semantic2d_from_semantic3d

        Semantic3DからSemantic2Dラベルを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: Semantic2Dラベル

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.semantic2d_common(src, minibatch_config)
        """
        dst:np.ndarray = self.common_semantic2d_from_semantic3d(key, link_idx, minibatch_config)

        return self.semantic2d_common(dst, minibatch_config)

    def common_semantic2d_from_voxelsemantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """common_semantic2d_from_voxelsemantic3d

        Voxel-Semantic3dからSemantic2dを生成する時の共通処理

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: Semanticラベル

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.create_intrinsic_array(key, link_idx, minibatch_config)
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
            HDF5Dataset.convert_label(src, minibatch_config)
        """
        voxelsemantic3d_key:str = self.create_h5_key(TYPE_VOXEL_SEMANTIC3D, key, link_idx, minibatch_config)

        vgm = VoxelGridMap(quiet=self.quiet)
        vgm.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        vgm.set_shape(minibatch_config[CONFIG_TAG_SHAPE])
        vgm.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        vgm.set_empty_voxelgridmap(
            self.h5links[voxelsemantic3d_key].shape, self.h5links[voxelsemantic3d_key].attrs[H5_ATTR_VOXELSIZE],
            tuple(self.h5links[voxelsemantic3d_key].attrs[H5_ATTR_VOXELMIN]), tuple(self.h5links[voxelsemantic3d_key].attrs[H5_ATTR_VOXELMAX]),
            tuple(self.h5links[voxelsemantic3d_key].attrs[H5_ATTR_VOXELCENTER]), tuple(self.h5links[voxelsemantic3d_key].attrs[H5_ATTR_VOXELORIGIN].tolist())
        )

        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)

        indexs = vgm.get_voxels_include_frustum(translation=pose[:3], quaternion=pose[-4:])
        vgm.set_voxels(indexs, self.h5links[voxelsemantic3d_key][()][indexs])
        dst:np.ndarray = vgm.create_semantic2d(translation=pose[:3], quaternion=pose[-4:], filter_radius=self.visibility_filter_radius, filter_threshold=self.visibility_filter_threshold)

        if minibatch_config[CONFIG_TAG_LABELTAG] in self.label_convert_configs.keys():
            dst = self.convert_label(dst, minibatch_config[CONFIG_TAG_LABELTAG])

        return dst

    def create_semantic2d_from_voxelsemantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_semantic2d_from_voxelsemantic3d

        Voxel-Semantic3DからSemantic2Dラベルを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: Semantic2Dラベル

        Dependent Functions:
            HDF5Dataset.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
            HDF5Dataset.semantic2d_common(src, minibatch_config)
        """
        dst:np.ndarray = self.common_semantic2d_from_voxelsemantic3d(key, link_idx, minibatch_config)
        return self.semantic2d_common(dst, minibatch_config)

    def depth_common(self, src:np.ndarray, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """depth_common

        'depth'画像生成時の共通処理

        Args:
            src (np.ndarray): 画像
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'depth'画像
        """
        dst = src
        if minibatch_config[CONFIG_TAG_NORMALIZE] is True:
            range_min = minibatch_config[CONFIG_TAG_RANGE][0]
            range_max = minibatch_config[CONFIG_TAG_RANGE][1]
            dst = np.where(dst > range_max, NORMALIZE_INF, (dst - range_min) / (range_max - range_min))

        shape = minibatch_config[CONFIG_TAG_SHAPE]
        if shape != dst.shape[:2]:
            dst = cv2.resize(dst, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

        return dst

    def create_depth_from_depth(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_depth

        深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 深度マップ

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.depth_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_DEPTH, key, link_idx, minibatch_config)
        dst:np.ndarray = self.h5links[h5_key][()]

        return self.depth_common(dst, minibatch_config)

    def create_depth_from_disparity(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_depth

        Disparityから深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 深度マップ

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.create_intrinsic_array(key, link_idx, minibatch_config)
            HDF5Dataset.depth_common(src, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_DISPARITY, key, link_idx, minibatch_config)

        depth = Depth()
        depth.set_base_line(self.h5links[h5_key].attrs[H5_ATTR_BASELINE])
        tmp_config = minibatch_config.copy()
        tmp_config[CONFIG_TAG_SHAPE] = list(self.h5links[h5_key].shape)
        depth.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        depth.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        depth.set_shape(self.h5links[h5_key].shape)
        depth.set_disparity(self.h5links[h5_key][()])

        return self.depth_common(depth.get_depthmap(), minibatch_config)

    def create_depth_from_points(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_points

        Pose (Translation & Quaternion) と三次元点群から深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 三次元地図を投影した深度マップ

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.create_intrinsic_array(key, link_idx, minibatch_config)
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
            HDF5Dataset.depth_common(src, minibatch_config)
        """
        points_key:str = self.create_h5_key(TYPE_POINTS, key, link_idx, minibatch_config)

        pts = Points(quiet=self.quiet)
        pts.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        pts.set_shape(minibatch_config[CONFIG_TAG_SHAPE])
        pts.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        pts.set_points(self.h5links[points_key][()])

        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)

        dst:np.ndarray = pts.create_depthmap(translation=pose[:3], quaternion=pose[-4:], filter_radius=self.visibility_filter_radius, filter_threshold=self.visibility_filter_threshold)

        return self.depth_common(dst, minibatch_config)

    def create_depth_from_semantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_semantic3d

        Pose (Translation & Quaternion) とSemantic3Dから深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 三次元点群を投影した深度マップ

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.create_intrinsic_array(key, link_idx, minibatch_config)
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
            HDF5Dataset.depth_common(src, minibatch_config)
        """
        semantic3d_key:str = self.create_h5_key(TYPE_SEMANTIC3D, key, link_idx, minibatch_config)

        pts = Points(quiet=True)
        pts.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        pts.set_shape(minibatch_config[CONFIG_TAG_SHAPE])
        pts.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        pts.set_semanticpoints(self.h5links[semantic3d_key][SUBTYPE_POINTS][()], self.h5links[semantic3d_key][SUBTYPE_SEMANTIC1D][()])

        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)

        dst:np.ndarray = pts.create_depthmap(translation=pose[:3], quaternion=pose[-4:], filter_radius=self.visibility_filter_radius, filter_threshold=self.visibility_filter_threshold)

        return self.depth_common(dst, minibatch_config)

    def create_depth_from_pointsmap(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_pointsmap

        Pose (Translation & Quaternion) と三次元地図から深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 三次元地図を投影した深度マップ

        Dependent Functions:
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
            HDF5Dataset.depth_common(src, minibatch_config)
        """
        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)
        dst = self.maps[self.link_maps[str(link_idx)]].create_depthmap(translation=pose[:3], quaternion=pose[-4:], filter_radius=self.visibility_filter_radius, filter_threshold=self.visibility_filter_threshold)

        return self.depth_common(dst, minibatch_config)

    def common_depth_from_voxel(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]], voxel_key:str) -> np.ndarray:
        """common_depth_from_voxel

        VoxelGridMapから深度マップを生成する際の共通処理

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定
            voxel_key (str): VoxelGridMapのキー

        Returns:
            np.ndarray: VoxelGridMapを投影した深度マップ

        Dependent Functions:
            HDF5Dataset.create_intrinsic_array(key, link_idx, minibatch_config)
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
            HDF5Dataset.depth_common(src, minibatch_config)
        """
        vgm = VoxelGridMap(quiet=self.quiet)
        vgm.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        vgm.set_shape(minibatch_config[CONFIG_TAG_SHAPE])
        vgm.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        vgm.set_empty_voxelgridmap(
            self.h5links[voxel_key].shape, self.h5links[voxel_key].attrs[H5_ATTR_VOXELSIZE],
            tuple(self.h5links[voxel_key].attrs[H5_ATTR_VOXELMIN]), tuple(self.h5links[voxel_key].attrs[H5_ATTR_VOXELMAX]),
            tuple(self.h5links[voxel_key].attrs[H5_ATTR_VOXELCENTER]), tuple(self.h5links[voxel_key].attrs[H5_ATTR_VOXELORIGIN].tolist())
        )

        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)

        indexs = vgm.get_voxels_include_frustum(translation=pose[:3], quaternion=pose[-4:])
        vgm.set_voxels(indexs, self.h5links[voxel_key][()][indexs])
        dst:np.ndarray = vgm.create_depthmap(translation=pose[:3], quaternion=pose[-4:], filter_radius=self.visibility_filter_radius, filter_threshold=self.visibility_filter_threshold)

        return self.depth_common(dst, minibatch_config)

    def create_depth_from_voxelpoints(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_voxelpoints

        'voxel-points'から深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'voxel-points'を投影した深度マップ

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.common_depth_from_voxel(key, link_idx, minibatch_config, voxel_key)
        """
        voxel_key:str = self.create_h5_key(TYPE_VOXEL_POINTS, key, link_idx, minibatch_config)
        return self.common_depth_from_voxel(key, link_idx, minibatch_config, voxel_key)

    def create_depth_from_voxelsemantic3d(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_voxelsemantic3d

        'voxel-semantic3d'から深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 'voxel-semantic3d'を投影した深度マップ

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
            HDF5Dataset.common_depth_from_voxel(key, link_idx, minibatch_config, voxel_key)
        """
        voxel_key:str = self.create_h5_key(TYPE_VOXEL_SEMANTIC3D, key, link_idx, minibatch_config)
        return self.common_depth_from_voxel(key, link_idx, minibatch_config, voxel_key)

    def create_pose_from_pose(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_pose_from_pose

        "pose"を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: [tx, ty, tz, qx, qy, qz, qw]
        """
        translations = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        quaternions = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)]

        for child_frame_id, invert in minibatch_config[CONFIG_TAG_TF]:
            h5_key = self.tf[CONFIG_TAG_DATA][child_frame_id][CONFIG_TAG_KEY]
            if h5_key[0] == '/':
                h5_key = str(link_idx) + h5_key
            else:
                h5_key = os.path.join(key, h5_key)
            trns:np.ndarray = self.h5links[h5_key][SUBTYPE_TRANSLATION][()]
            qtrn:np.ndarray = self.h5links[h5_key][SUBTYPE_ROTATION][()]
            if invert is True:
                trns, qtrn = invertTransform(translation=trns, quaternion=qtrn)
            translations.append(trns)
            quaternions.append(qtrn)

        translation, quaternion = combineTransforms(translations=translations, quaternions=quaternions)

        return np.concatenate([translation, quaternion])

    def create_pose_from_translation_quaternion(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_pose_from_translation_quaternion

        "translation"と"quaternion"から"pose"を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: [tx, ty, tz, qx, qy, qz, qw]

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
        """
        translation_key:str = self.create_h5_key(TYPE_TRANSLATION, key, link_idx, minibatch_config)
        translation:np.ndarray = self.h5links[translation_key][()]

        quaternion_key:str = self.create_h5_key(TYPE_QUATERNION, key, link_idx, minibatch_config)
        quaternion:np.ndarray = self.h5links[quaternion_key][()]

        return np.concatenate([translation, quaternion])

    def create_translation_from_translation(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_translation_from_translation

        "translation"を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: [tx, ty, tz]

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_TRANSLATION, key, link_idx, minibatch_config)

        return self.h5links[h5_key][()]

    def create_translation_from_pose(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_translation_from_pose

        "pose"から"translation"を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: [tx, ty, tz]

        Dependent Functions:
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
        """
        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)

        return pose[:3]

    def create_quaternion_from_quaternion(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_quaternion_from_quaternion

        "quaternion"を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: [qx, qy, qz, qw]

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_QUATERNION, key, link_idx, minibatch_config)

        return self.h5links[h5_key][()]

    def create_quaternion_from_pose(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_quaternion_from_pose

        "pose"から"quaternion"を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: [qx, qy, qz, qw]

        Dependent Functions:
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
        """
        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)

        return pose[-4:]

    def create_intrinsic_array(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_intrinsic_array

        カメラ内部パラメータの3x3行列を生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: カメラ行列

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_INTRINSIC, key, link_idx, minibatch_config)
        shape:tuple = minibatch_config[CONFIG_TAG_SHAPE]

        coef_x = float(shape[1]) / float(self.h5links[h5_key + '/width'][()])
        coef_y = float(shape[0]) / float(self.h5links[h5_key + '/height'][()])

        fx = self.h5links[h5_key + '/Fx'][()] * coef_x
        fy = self.h5links[h5_key + '/Fy'][()] * coef_y
        cx = self.h5links[h5_key + '/Cx'][()] * coef_x
        cy = self.h5links[h5_key + '/Cy'][()] * coef_y

        return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    def create_color(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_color

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: [b, g, r]

        Dependent Functions:
            HDF5Dataset.create_h5_key(data_type, key, link_idx, minibatch_config)
        """
        h5_key:str = self.create_h5_key(TYPE_COLOR, key, link_idx, minibatch_config)
        return self.h5links[h5_key][()]

    def convert_str(self, src:Any) -> str:
        """convert_str

        bytes型にも対応したstr()

        Args:
            src (Any): strに変換する値

        Returns:
            str: 文字列 or None
        """
        if src is None:
            return None
        elif isinstance(src, str):
            return src
        elif isinstance(src, bytes):
            return src.decode()
        else:
            return str(src)

    def convert_label(self, src:np.ndarray, label_tag:str) -> np.ndarray:
        """convert_label

        ラベルの変換を行う

        Args:
            src (np.ndarray): 変換するNumpy行列
            label_tag (str): ラベルの設定のタグ

        Returns:
            np.ndarray: 変換後のNumpy行列
        """
        tmp:np.ndarray = src.copy()
        for label_convert_config in self.label_convert_configs[label_tag]:
            tmp = np.where(src == label_convert_config[CONFIG_TAG_SRC], label_convert_config[CONFIG_TAG_DST], tmp)
        return tmp

    def convert_semantic2d_to_bgr8(self, src:np.ndarray, label_tag:str) -> np.ndarray:
        """convert_semantic2d_to_bgr8

        'semantic2d'のラベルから'bgr8'のカラー画像へ変換する

        Args:
            src (np.ndarray): 変換する'semantic2d'
            label_tag (str): ラベルの設定のタグ

        Returns:
            np.ndarray: 変換後の'bgr8'画像
        """
        dst:np.ndarray = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
        for color_config in self.label_color_configs[label_tag]:
            dst[np.where(src == int(color_config[CONFIG_TAG_LABEL]))] = color_config[CONFIG_TAG_COLOR]
        return dst

    def convert_semantic2d_to_rgb8(self, src:np.ndarray, label_tag:str) -> np.ndarray:
        """convert_semantic2d_to_rgb8

        'semantic2d'のラベルから'rgb8'のカラー画像へ変換する

        Args:
            src (np.ndarray): 変換する'semantic2d'
            label_tag (str): ラベルの設定のタグ

        Returns:
            np.ndarray: 変換後の'rgb8'画像
        """
        dst:np.ndarray = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
        for color_config in self.label_color_configs[label_tag]:
            dst[np.where(src == int(color_config[CONFIG_TAG_LABEL]))] = color_config[CONFIG_TAG_COLOR][::-1]
        return dst

    def convert_semantic2d_to_bgra8(self, src:np.ndarray, label_tag:str) -> np.ndarray:
        """convert_semantic2d_to_bgra8

        'semantic2d'のラベルから'bgra8'のカラー画像へ変換する

        Args:
            src (np.ndarray): 変換する'semantic2d'
            label_tag (str): ラベルの設定のタグ

        Returns:
            np.ndarray: 変換後の'bgra8'画像
        """
        dst:np.ndarray = np.zeros((src.shape[0], src.shape[1], 4), np.uint8)
        for color_config in self.label_color_configs[label_tag]:
            dst[np.where(src == int(color_config[CONFIG_TAG_LABEL]))] = np.append(color_config[CONFIG_TAG_COLOR], 255)
        return dst

    def convert_semantic2d_to_rgba8(self, src:np.ndarray, label_tag:str) -> np.ndarray:
        """convert_semantic2d_to_rgba8

        'semantic2d'のラベルから'rgba8'のカラー画像へ変換する

        Args:
            src (np.ndarray): 変換する'semantic2d'
            label_tag (str): ラベルの設定のタグ

        Returns:
            np.ndarray: 変換後の'rgba8'画像
        """
        dst:np.ndarray = np.zeros((src.shape[0], src.shape[1], 4), np.uint8)
        for color_config in self.label_color_configs[label_tag]:
            dst[np.where(src == int(color_config[CONFIG_TAG_LABEL]))] = np.append(color_config[CONFIG_TAG_COLOR][::-1], 255)
        return dst
