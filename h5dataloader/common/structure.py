# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union
import numpy as np
import cv2

TYPE_FLOAT16:str = 'float16'
TYPE_FLOAT32:str = 'float32'
TYPE_FLOAT64:str = 'float64'
TYPE_UINT8:str = 'uint8'
TYPE_INT8:str = 'int8'
TYPE_INT16:str = 'int16'
TYPE_INT32:str = 'int32'
TYPE_INT64:str = 'int64'
TYPE_MONO8:str = 'mono8'
TYPE_MONO16:str = 'mono16'
TYPE_BGR8:str = 'bgr8'
TYPE_RGB8:str = 'rgb8'
TYPE_BGRA8:str = 'bgra8'
TYPE_RGBA8:str = 'rgba8'
TYPE_DEPTH:str = 'depth'
TYPE_POINTS:str = 'points'
TYPE_VOXEL_POINTS:str = 'voxel-points'
TYPE_SEMANTIC1D:str = 'semantic1d'
TYPE_SEMANTIC2D:str = 'semantic2d'
TYPE_SEMANTIC3D:str = 'semantic3d'
TYPE_VOXEL_SEMANTIC3D:str = 'voxel-semantic3d'
TYPE_POSE:str = 'pose'
TYPE_TRANSLATION:str = 'translation'
TYPE_QUATERNION:str = 'quaternion'
TYPE_INTRINSIC:str = 'intrinsic'
TYPE_COLOR:str = 'color'

SUBTYPE_TRANSLATION:str = 'translation'
SUBTYPE_ROTATION:str = 'rotation'
SUBTYPE_POINTS:str = 'points'
SUBTYPE_SEMANTIC1D:str = 'semantic1d'
SUBTYPE_FX:str = 'Fx'
SUBTYPE_FY:str = 'Fy'
SUBTYPE_CX:str = 'Cx'
SUBTYPE_CY:str = 'Cy'
SUBTYPE_HEIGHT:str = 'height'
SUBTYPE_WIDTH:str = 'width'
SUBTYPE_NAME:str = 'name'
SUBTYPE_VOXEL_POINTS:str = 'points-voxel'
SUBTYPE_VOXEL_SEMANTIC3D:str = 'semantic3d-voxel'

CONFIG_TAG_MINIBATCH:str = 'mini-batch'
CONFIG_TAG_TYPE:str = 'type'
CONFIG_TAG_FROM:str = 'from'
CONFIG_TAG_SHAPE:str = 'shape'
CONFIG_TAG_NORMALIZE:str = 'normalize'
CONFIG_TAG_RANGE:str = 'range'
CONFIG_TAG_KEY:str = 'key'
CONFIG_TAG_TF:str = 'tf'
CONFIG_TAG_TREE:str = 'tree'
CONFIG_TAG_LIST:str = 'list'
CONFIG_TAG_DATA:str = 'data'
CONFIG_TAG_FRAMEID:str = 'frame-id'
CONFIG_TAG_CHILDFRAMEID:str = 'child-frame-id'
CONFIG_TAG_CREATEFUNC:str = 'create-func'
CONFIG_TAG_SRCDATA:str = 'src-data'
CONFIG_TAG_TAG:str = 'tag'
CONFIG_TAG_LABEL:str = 'label'
CONFIG_TAG_CLASS:str = 'class'
CONFIG_TAG_SRC:str = 'src'
CONFIG_TAG_DST:str = 'dst'
CONFIG_TAG_CONFIG:str = 'config'
CONFIG_TAG_CONVERT:str = 'convert'
CONFIG_TAG_COLOR:str = 'color'
CONFIG_TAG_LABELTAG:str = 'label-tag'

H5_KEY_HEADER:str = 'header'
H5_KEY_LENGTH:str = 'length'
H5_KEY_LABEL:str = 'label'
H5_KEY_DATA:str = 'data'
H5_KEY_NAME:str = 'name'
H5_ATTR_TYPE:str = 'type'
H5_ATTR_STAMPSEC:str = 'stamp.sec'
H5_ATTR_STAMPNSEC:str = 'stamp.nsec'
H5_ATTR_FRAMEID:str = 'frame_id'
H5_ATTR_CHILDFRAMEID:str = 'child_frame_id'
H5_ATTR_ARRAY:str = 'array'
H5_ATTR_FILEPATH:str = 'file_path'
H5_ATTR_MAPID:str = 'map_id'
H5_ATTR_LABELTAG:str = 'label_tag'
H5_ATTR_VOXELSIZE:str = 'voxel_size'
H5_ATTR_VOXELMIN:str = 'voxel_min'
H5_ATTR_VOXELMAX:str = 'voxel_max'
H5_ATTR_VOXELCENTER:str = 'voxel_center'
H5_ATTR_VOXELORIGIN:str = 'voxel_origin'

DTYPE_NUMPY:Dict[str, np.dtype] = {
    TYPE_FLOAT16: np.float16,
    TYPE_FLOAT32: np.float32,
    TYPE_FLOAT64: np.float64,
    TYPE_UINT8: np.uint8,
    TYPE_INT8: np.int8,
    TYPE_INT16: np.int16,
    TYPE_INT32: np.int32,
    TYPE_INT64: np.int64,
    TYPE_MONO8: np.uint8,
    TYPE_MONO16: np.uint16,
    TYPE_BGR8: np.uint8,
    TYPE_RGB8: np.uint8,
    TYPE_BGRA8: np.uint8,
    TYPE_RGBA8: np.uint8,
    TYPE_DEPTH: np.float32,
    TYPE_POINTS: np.float32,
    TYPE_VOXEL_POINTS: np.object,
    SUBTYPE_VOXEL_POINTS: np.dtype([('x', np.float32), ('y', np.float32), ('z',np.float32)]),
    TYPE_SEMANTIC1D: np.uint8,
    TYPE_SEMANTIC2D: np.uint8,
    TYPE_SEMANTIC3D: None,
    TYPE_VOXEL_SEMANTIC3D: np.object,
    SUBTYPE_VOXEL_SEMANTIC3D: np.dtype([('x', np.float32), ('y', np.float32), ('z',np.float32), ('label',np.uint8)]),
    TYPE_POSE: None,
    TYPE_TRANSLATION: np.float32,
    TYPE_QUATERNION: np.float32,
    TYPE_INTRINSIC: np.float32,
    TYPE_COLOR: np.uint8,
}

INTERPOLATION_FLAG:Dict[str, Union[int, None]] = {
    TYPE_FLOAT16: None,
    TYPE_FLOAT32: None,
    TYPE_FLOAT64: None,
    TYPE_UINT8: None,
    TYPE_INT8: None,
    TYPE_INT16: None,
    TYPE_INT32: None,
    TYPE_INT64: None,
    TYPE_MONO8: cv2.INTER_LINEAR,
    TYPE_MONO16: cv2.INTER_LINEAR,
    TYPE_BGR8: cv2.INTER_LINEAR,
    TYPE_RGB8: cv2.INTER_LINEAR,
    TYPE_BGRA8: cv2.INTER_LINEAR,
    TYPE_RGBA8: cv2.INTER_LINEAR,
    TYPE_DEPTH: cv2.INTER_NEAREST,
    TYPE_POINTS: None,
    TYPE_VOXEL_POINTS: None,
    TYPE_SEMANTIC1D: None,
    TYPE_SEMANTIC2D: cv2.INTER_NEAREST,
    TYPE_SEMANTIC3D: None,
    TYPE_VOXEL_SEMANTIC3D: None,
    TYPE_POSE: None,
    TYPE_TRANSLATION: None,
    TYPE_QUATERNION: None,
    TYPE_INTRINSIC: None,
    TYPE_COLOR: None,
}

DEFAULT_RANGE:Dict[str, Tuple[Union[int, float], Union[int, float]]] = {
    TYPE_FLOAT16: (-np.inf, np.inf),
    TYPE_FLOAT32: (-np.inf, np.inf),
    TYPE_FLOAT64: (-np.inf, np.inf),
    TYPE_UINT8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
    TYPE_INT8: (np.iinfo(np.int8).min, np.iinfo(np.int8).max),
    TYPE_INT16: (np.iinfo(np.int16).min, np.iinfo(np.int16).max),
    TYPE_INT32: (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
    TYPE_INT64: (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
    TYPE_MONO8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
    TYPE_MONO16: (np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
    TYPE_BGR8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
    TYPE_RGB8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
    TYPE_BGRA8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
    TYPE_RGBA8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
    TYPE_DEPTH: (0., np.inf),
    TYPE_POINTS: (-np.inf, np.inf),
    TYPE_VOXEL_POINTS: (-np.inf, np.inf),
    TYPE_SEMANTIC1D: None,
    TYPE_SEMANTIC2D: None,
    TYPE_SEMANTIC3D: (-np.inf, np.inf),
    TYPE_VOXEL_SEMANTIC3D: (-np.inf, np.inf),
    TYPE_POSE: None,
    TYPE_TRANSLATION: None,
    TYPE_QUATERNION: None,
    TYPE_INTRINSIC: None,
    TYPE_COLOR: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
}

ZERO_VALUE:Dict[str, Union[int, np.ndarray]] = {
    TYPE_FLOAT16: 0.0,
    TYPE_FLOAT32: 0.0,
    TYPE_FLOAT64: 0.0,
    TYPE_UINT8: 0,
    TYPE_INT8: 0,
    TYPE_INT16: 0,
    TYPE_INT32: 0,
    TYPE_INT64: 0,
    TYPE_MONO8: 0,
    TYPE_MONO16: 0,
    TYPE_BGR8: np.array([0, 0, 0], dtype=np.uint8),
    TYPE_RGB8: np.array([0, 0, 0], dtype=np.uint8),
    TYPE_BGRA8: np.array([0, 0, 0, 0], dtype=np.uint8),
    TYPE_RGBA8: np.array([0, 0, 0, 0], dtype=np.uint8),
    TYPE_DEPTH: 0.0,
    TYPE_POINTS: None,
    TYPE_VOXEL_POINTS: None,
    TYPE_SEMANTIC1D: None,
    TYPE_SEMANTIC2D: None,
    TYPE_SEMANTIC3D: None,
    TYPE_VOXEL_SEMANTIC3D: None,
    TYPE_POSE: None,
    TYPE_TRANSLATION: None,
    TYPE_QUATERNION: None,
    TYPE_INTRINSIC: None,
    TYPE_COLOR: np.array([0, 0, 0], dtype=np.uint8),
}

FROM_TYPES:Dict[str, List[List[str]]] = {
    TYPE_FLOAT16: [
        [TYPE_FLOAT16],
        [TYPE_FLOAT32],
        [TYPE_FLOAT64],
        [TYPE_UINT8],
        [TYPE_INT8],
        [TYPE_INT16],
        [TYPE_INT32],
        [TYPE_INT64],
    ],
    TYPE_FLOAT32: [
        [TYPE_FLOAT32],
        [TYPE_FLOAT16],
        [TYPE_FLOAT64],
        [TYPE_UINT8],
        [TYPE_INT8],
        [TYPE_INT16],
        [TYPE_INT32],
        [TYPE_INT64],
    ],
    TYPE_FLOAT64: [
        [TYPE_FLOAT64],
        [TYPE_FLOAT16],
        [TYPE_FLOAT32],
        [TYPE_UINT8],
        [TYPE_INT8],
        [TYPE_INT16],
        [TYPE_INT32],
        [TYPE_INT64],
    ],
    TYPE_UINT8: [
        [TYPE_UINT8],
        [TYPE_INT8],
        [TYPE_INT16],
        [TYPE_INT32],
        [TYPE_INT64],
        [TYPE_FLOAT16],
        [TYPE_FLOAT32],
        [TYPE_FLOAT64],
    ],
    TYPE_INT8: [
        [TYPE_INT8],
        [TYPE_UINT8],
        [TYPE_INT16],
        [TYPE_INT32],
        [TYPE_INT64],
        [TYPE_FLOAT16],
        [TYPE_FLOAT32],
        [TYPE_FLOAT64],
    ],
    TYPE_INT16: [
        [TYPE_INT16],
        [TYPE_UINT8],
        [TYPE_INT8],
        [TYPE_INT32],
        [TYPE_INT64],
        [TYPE_FLOAT16],
        [TYPE_FLOAT32],
        [TYPE_FLOAT64],
    ],
    TYPE_INT32: [
        [TYPE_INT32],
        [TYPE_UINT8],
        [TYPE_INT8],
        [TYPE_INT16],
        [TYPE_INT64],
        [TYPE_FLOAT16],
        [TYPE_FLOAT32],
        [TYPE_FLOAT64],
    ],
    TYPE_INT64: [
        [TYPE_INT64],
        [TYPE_UINT8],
        [TYPE_INT8],
        [TYPE_INT16],
        [TYPE_INT32],
        [TYPE_FLOAT16],
        [TYPE_FLOAT32],
        [TYPE_FLOAT64],
    ],
    TYPE_MONO8: [
        [TYPE_MONO8],
        [TYPE_MONO16],
        [TYPE_BGR8],
        [TYPE_RGB8],
        [TYPE_BGRA8],
        [TYPE_RGBA8],
    ],
    TYPE_MONO16: [
        [TYPE_MONO16],
        [TYPE_MONO8],
        [TYPE_BGR8],
        [TYPE_RGB8],
        [TYPE_BGRA8],
        [TYPE_RGBA8],
    ],
    TYPE_BGR8: [
        [TYPE_BGR8],
        [TYPE_MONO8],
        [TYPE_MONO16],
        [TYPE_RGB8],
        [TYPE_BGRA8],
        [TYPE_RGBA8],
        [TYPE_SEMANTIC2D],
        [TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
    ],
    TYPE_RGB8: [
        [TYPE_RGB8],
        [TYPE_MONO8],
        [TYPE_MONO16],
        [TYPE_BGR8],
        [TYPE_BGRA8],
        [TYPE_RGBA8],
        [TYPE_SEMANTIC2D],
        [TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
    ],
    TYPE_BGRA8: [
        [TYPE_BGRA8],
        [TYPE_MONO8],
        [TYPE_MONO16],
        [TYPE_BGR8],
        [TYPE_RGB8],
        [TYPE_RGBA8],
        [TYPE_SEMANTIC2D],
        [TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
    ],
    TYPE_RGBA8: [
        [TYPE_RGBA8],
        [TYPE_MONO8],
        [TYPE_MONO16],
        [TYPE_BGR8],
        [TYPE_RGB8],
        [TYPE_BGRA8],
        [TYPE_SEMANTIC2D],
        [TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
    ],
    TYPE_DEPTH: [
        [TYPE_DEPTH],
        [TYPE_POINTS, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_POINTS, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
    ],
    TYPE_POINTS: [
        [TYPE_POINTS, TYPE_POSE],
        [TYPE_DEPTH, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_POINTS, TYPE_POSE],
        [TYPE_SEMANTIC3D, TYPE_POSE],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_POSE],
    ],
    TYPE_SEMANTIC1D: [
        [TYPE_SEMANTIC1D],
        [TYPE_SEMANTIC3D]
    ],
    TYPE_SEMANTIC2D: [
        [TYPE_SEMANTIC2D],
        [TYPE_POINTS, TYPE_SEMANTIC1D, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_SEMANTIC3D, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_INTRINSIC, TYPE_POSE],
    ],
    TYPE_SEMANTIC3D: [
        [TYPE_SEMANTIC3D, TYPE_POSE],
        [TYPE_SEMANTIC1D, TYPE_POINTS, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_SEMANTIC2D, TYPE_DEPTH, TYPE_POSE, TYPE_INTRINSIC],
        # [TYPE_SEMANTIC2D, TYPE_POINTS, TYPE_POSE, TYPE_INTRINSIC],
        [TYPE_VOXEL_SEMANTIC3D, TYPE_POSE],
    ],
    TYPE_POSE: [
        [TYPE_POSE],
        [TYPE_TRANSLATION, TYPE_QUATERNION],
    ],
    TYPE_TRANSLATION: [
        [TYPE_TRANSLATION],
        [TYPE_POSE],
    ],
    TYPE_QUATERNION: [
        [TYPE_QUATERNION],
        [TYPE_POSE],
    ],
    TYPE_INTRINSIC: [
        [TYPE_INTRINSIC],
    ],
    TYPE_COLOR: [
        [TYPE_COLOR],
    ],
}

ENABLE_NORMALIZE:Dict[str, bool] = {
    TYPE_FLOAT16: True,
    TYPE_FLOAT32: True,
    TYPE_FLOAT64: True,
    TYPE_UINT8: True,
    TYPE_INT8: True,
    TYPE_INT16: True,
    TYPE_INT32: True,
    TYPE_INT64: True,
    TYPE_MONO8: True,
    TYPE_MONO16: True,
    TYPE_BGR8: True,
    TYPE_RGB8: True,
    TYPE_BGRA8: True,
    TYPE_RGBA8: True,
    TYPE_DEPTH: True,
    TYPE_POINTS: True,
    TYPE_VOXEL_POINTS: False,
    TYPE_SEMANTIC1D: False,
    TYPE_SEMANTIC2D: False,
    TYPE_SEMANTIC3D: True,
    TYPE_VOXEL_SEMANTIC3D: False,
    TYPE_POSE: False,
    TYPE_TRANSLATION: False,
    TYPE_QUATERNION: False,
    TYPE_INTRINSIC: False,
    TYPE_COLOR: True,
}

USE_LABEL:Dict[str, bool] = {
    TYPE_FLOAT16: False,
    TYPE_FLOAT32: False,
    TYPE_FLOAT64: False,
    TYPE_UINT8: False,
    TYPE_INT8: False,
    TYPE_INT16: False,
    TYPE_INT32: False,
    TYPE_INT64: False,
    TYPE_MONO8: False,
    TYPE_MONO16: False,
    TYPE_BGR8: False,
    TYPE_RGB8: False,
    TYPE_BGRA8: False,
    TYPE_RGBA8: False,
    TYPE_DEPTH: False,
    TYPE_POINTS: False,
    TYPE_VOXEL_POINTS: False,
    TYPE_SEMANTIC1D: True,
    TYPE_SEMANTIC2D: True,
    TYPE_SEMANTIC3D: True,
    TYPE_VOXEL_SEMANTIC3D: True,
    TYPE_POSE: False,
    TYPE_TRANSLATION: False,
    TYPE_QUATERNION: False,
    TYPE_INTRINSIC: False,
    TYPE_COLOR: False,
}
