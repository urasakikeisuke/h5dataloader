# -*- coding: utf-8 -*-

from typing import Dict
import numpy as np

from h5dataloader.common.structure import *

DTYPE_TORCH:Dict[str, np.dtype] = {
    TYPE_FLOAT16: np.float32,
    TYPE_FLOAT32: np.float32,
    TYPE_FLOAT64: np.float32,
    TYPE_UINT8: np.int64,
    TYPE_INT8: np.int64,
    TYPE_INT16: np.int64,
    TYPE_INT32: np.int64,
    TYPE_INT64: np.int64,
    TYPE_MONO8: np.float32,
    TYPE_MONO16: np.float32,
    TYPE_BGR8: np.float32,
    TYPE_RGB8: np.float32,
    TYPE_BGRA8: np.float32,
    TYPE_RGBA8: np.float32,
    TYPE_DEPTH: np.float32,
    TYPE_POINTS: np.float32,
    TYPE_SEMANTIC1D: np.int64,
    TYPE_SEMANTIC2D: np.int64,
    TYPE_SEMANTIC3D: np.float32,
    TYPE_POSE: np.float32,
    TYPE_TRANSLATION: np.float32,
    TYPE_QUATERNION: np.float32,
    TYPE_INTRINSIC: np.float32,
    TYPE_COLOR: np.float32,
}

def hwc2chw(hwc:np.ndarray) -> np.ndarray:
    return np.transpose(hwc, [2, 0, 1])

def hw2chw(hw:np.ndarray) -> np.ndarray:
    return np.transpose(hw[:,:,np.newaxis], [2, 0, 1])

def nochange(array:np.ndarray) -> np.ndarray:
    return array

CONVERT_TORCH:Dict[str, type] = {
    TYPE_FLOAT16: nochange,
    TYPE_FLOAT32: nochange,
    TYPE_FLOAT64: nochange,
    TYPE_UINT8: nochange,
    TYPE_INT8: nochange,
    TYPE_INT16: nochange,
    TYPE_INT32: nochange,
    TYPE_INT64: nochange,
    TYPE_MONO8: hw2chw,
    TYPE_MONO16: hw2chw,
    TYPE_BGR8: hwc2chw,
    TYPE_RGB8: hwc2chw,
    TYPE_BGRA8: hwc2chw,
    TYPE_RGBA8: hwc2chw,
    TYPE_DEPTH: hw2chw,
    TYPE_POINTS: nochange,
    TYPE_SEMANTIC1D: nochange,
    TYPE_SEMANTIC2D: nochange,
    TYPE_SEMANTIC3D: nochange,
    TYPE_POSE: nochange,
    TYPE_TRANSLATION: nochange,
    TYPE_QUATERNION: nochange,
    TYPE_INTRINSIC: nochange,
    TYPE_COLOR: nochange,
}
