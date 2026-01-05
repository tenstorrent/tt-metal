import ttnn
from yoloworld_pcc_drop.small_640_variant_neck import utils
import pytest
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Any, Optional, Tuple
from models.common.utility_functions import comp_pcc
from loguru import logger


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_0 = ttnn.Tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_Tensor_1 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
        ],
        [40],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_floor_0 = ttnn.floor(
        ttnn_Tensor_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_Tensor_1, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_floor_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_floor_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_typecast_0,
        [40, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_reshape_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_typecast_1,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_permute_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_Tensor_0,
        [1, 20],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_Tensor_0, False)
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_typecast_3,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_permute_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_eq_0 = ttnn.eq(
        ttnn_typecast_2,
        ttnn_typecast_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn.deallocate(ttnn_typecast_2, False)
    util_create_list_0 = [ttnn_eq_0]
    return util_create_list_0


def main_const_eval_1(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    util_create_list_1 = [ttnn_reshape_2]
    return util_create_list_1


def main_const_eval_2():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 8, 20, 20]),
        fill_value=5.65625,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_2 = [ttnn_full_0]
    return util_create_list_2


def main_const_eval_3(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_1 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_1,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    util_create_list_3 = [ttnn_reshape_3]
    return util_create_list_3


def main_const_eval_4(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_2 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_repeat_0 = ttnn.repeat(ttnn_reshape_4, ttnn.Shape([1, 3, 1]))
    ttnn.deallocate(ttnn_reshape_4, False)
    util_create_list_4 = [ttnn_repeat_0]
    return util_create_list_4


def main_const_eval_5(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_3 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_5,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_to_layout_3,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    util_create_list_5 = [ttnn_reshape_5]
    return util_create_list_5


def main_const_eval_6(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_4 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_4,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_4, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_to_layout_4,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    util_create_list_6 = [ttnn_reshape_6]
    return util_create_list_6


def main_const_eval_7(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_5 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_to_device_5,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_5, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_to_layout_5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    util_create_list_7 = [ttnn_reshape_7]
    return util_create_list_7


def main_const_eval_8(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_8 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_6 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_8,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_to_device_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_6, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_to_layout_6,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    util_create_list_8 = [ttnn_reshape_8]
    return util_create_list_8


def main_const_eval_9(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_9 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_7 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_9,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_to_device_7,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_7, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_to_layout_7,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_7, False)
    util_create_list_9 = [ttnn_reshape_9]
    return util_create_list_9


def main_const_eval_10(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_10 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_8 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_10,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_to_device_8,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_8, False)
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_to_layout_8,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_8, False)
    util_create_list_10 = [ttnn_reshape_10]
    return util_create_list_10


def main_const_eval_11(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_11 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_9 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_11,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_to_device_9,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_9, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_to_layout_9,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_9, False)
    util_create_list_11 = [ttnn_reshape_11]
    return util_create_list_11


def main_const_eval_12(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_12 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_10 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_12,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_to_device_10,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_10, False)
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_to_layout_10,
        [1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_10, False)
    ttnn_repeat_1 = ttnn.repeat(ttnn_reshape_12, ttnn.Shape([1, 3, 1]))
    ttnn.deallocate(ttnn_reshape_12, False)
    util_create_list_12 = [ttnn_repeat_1]
    return util_create_list_12


def main_const_eval_13(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_13 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_11 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_13,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_to_device_11,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_11, False)
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_to_layout_11,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_11, False)
    util_create_list_13 = [ttnn_reshape_13]
    return util_create_list_13


def main_const_eval_14(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_14 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_12 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_14,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_to_device_12,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_12, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_to_layout_12,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_12, False)
    util_create_list_14 = [ttnn_reshape_14]
    return util_create_list_14


def main_const_eval_15(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_15 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_13 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_15,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_to_device_13,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_13, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_to_layout_13,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_13, False)
    util_create_list_15 = [ttnn_reshape_15]
    return util_create_list_15


def main_const_eval_16(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_16 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_14 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_to_device_14,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_14, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_to_layout_14,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_14, False)
    util_create_list_16 = [ttnn_reshape_16]
    return util_create_list_16


def main_const_eval_17(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_17 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_15 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_17,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_to_device_15,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_15, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_to_layout_15,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_15, False)
    util_create_list_17 = [ttnn_reshape_17]
    return util_create_list_17


def main_const_eval_18(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_18 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_16 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_18,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_to_device_16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_16, False)
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_to_layout_16,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_16, False)
    util_create_list_18 = [ttnn_reshape_18]
    return util_create_list_18


def main_const_eval_19(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_19 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_17 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_19,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_to_device_17,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_17, False)
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_to_layout_17,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_17, False)
    util_create_list_19 = [ttnn_reshape_19]
    return util_create_list_19


def main_const_eval_20(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_20 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_18 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_20,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_to_device_18,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_18, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_to_layout_18,
        [1, 4, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_18, False)
    util_create_list_20 = [ttnn_reshape_20]
    return util_create_list_20


def main_const_eval_21(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_21 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_19 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_21,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_to_device_19,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_19, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_to_layout_19,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_19, False)
    util_create_list_21 = [ttnn_reshape_21]
    return util_create_list_21


def main_const_eval_22(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_22 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_20 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_22,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_to_device_20,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_20, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_to_layout_20,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_20, False)
    util_create_list_22 = [ttnn_reshape_22]
    return util_create_list_22


def main_const_eval_23(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_23 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_21 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_23,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_to_device_21,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_21, False)
    ttnn_reshape_23 = ttnn.reshape(
        ttnn_to_layout_21,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_21, False)
    util_create_list_23 = [ttnn_reshape_23]
    return util_create_list_23


def main_const_eval_24(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_24 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_22 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_24,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_22 = ttnn.to_layout(
        ttnn_to_device_22,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_22, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_to_layout_22,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    util_create_list_24 = [ttnn_reshape_24]
    return util_create_list_24


def main_const_eval_25(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_25 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_23 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_25,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_23 = ttnn.to_layout(
        ttnn_to_device_23,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_23, False)
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_to_layout_23,
        [1, 2, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    util_create_list_25 = [ttnn_reshape_25]
    return util_create_list_25


def main_const_eval_26(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_26 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_24 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_26,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_to_device_24,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_24, False)
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_to_layout_24,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_24, False)
    util_create_list_26 = [ttnn_reshape_26]
    return util_create_list_26


def main_const_eval_27(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_27 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_25 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_27,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_to_device_25,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_25, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_to_layout_25,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    util_create_list_27 = [ttnn_reshape_27]
    return util_create_list_27


def main_const_eval_28(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_28 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_26 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_28,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_to_device_26,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_26, False)
    ttnn_reshape_28 = ttnn.reshape(
        ttnn_to_layout_26,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_26, False)
    util_create_list_28 = [ttnn_reshape_28]
    return util_create_list_28


def main_const_eval_29(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_29 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_27 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_to_device_27,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_27, False)
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_to_layout_27,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_27, False)
    util_create_list_29 = [ttnn_reshape_29]
    return util_create_list_29


def main_const_eval_30(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_30 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_28 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_to_device_28,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_28, False)
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_to_layout_28,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_28, False)
    util_create_list_30 = [ttnn_reshape_30]
    return util_create_list_30


def main_const_eval_31(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_31 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_29 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_to_device_29,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_29, False)
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_to_layout_29,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_29, False)
    util_create_list_31 = [ttnn_reshape_31]
    return util_create_list_31


def main_const_eval_32(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_32 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_30 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_to_device_30,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_30, False)
    ttnn_reshape_32 = ttnn.reshape(
        ttnn_to_layout_30,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_30, False)
    util_create_list_32 = [ttnn_reshape_32]
    return util_create_list_32


def main_const_eval_33(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_33 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_31 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_33,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_to_device_31,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_31, False)
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_to_layout_31,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_31, False)
    util_create_list_33 = [ttnn_reshape_33]
    return util_create_list_33


def main_const_eval_34(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_34 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_32 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_34,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_to_device_32,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_32, False)
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_to_layout_32,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    util_create_list_34 = [ttnn_reshape_34]
    return util_create_list_34


def main_const_eval_35(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_35 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_33 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_35,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_to_device_33,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_33, False)
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_to_layout_33,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_33, False)
    util_create_list_35 = [ttnn_reshape_35]
    return util_create_list_35


def main_const_eval_36(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_36 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_34 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_36,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_to_device_34,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_34, False)
    ttnn_reshape_36 = ttnn.reshape(
        ttnn_to_layout_34,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_34, False)
    util_create_list_36 = [ttnn_reshape_36]
    return util_create_list_36


def main_const_eval_37():
    utils_DeviceGetter_get_device_37 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 4, 40, 40]),
        fill_value=5.65625,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_37,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_37 = [ttnn_full_1]
    return util_create_list_37


def main_const_eval_38(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_38 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_35 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_38,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_to_device_35,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_35, False)
    ttnn_reshape_37 = ttnn.reshape(
        ttnn_to_layout_35,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_35, False)
    util_create_list_38 = [ttnn_reshape_37]
    return util_create_list_38


def main_const_eval_39(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_39 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_36 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_39,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_to_device_36,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_36, False)
    ttnn_reshape_38 = ttnn.reshape(
        ttnn_to_layout_36,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_36, False)
    util_create_list_39 = [ttnn_reshape_38]
    return util_create_list_39


def main_const_eval_40(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_40 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_37 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_40,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_to_device_37,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_37, False)
    ttnn_reshape_39 = ttnn.reshape(
        ttnn_to_layout_37,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_37, False)
    util_create_list_40 = [ttnn_reshape_39]
    return util_create_list_40


def main_const_eval_41(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_41 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_38 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_41,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_to_device_38,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_38, False)
    ttnn_reshape_40 = ttnn.reshape(
        ttnn_to_layout_38,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_38, False)
    util_create_list_41 = [ttnn_reshape_40]
    return util_create_list_41


def main_const_eval_42(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_42 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_39 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_42,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_to_device_39,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_39, False)
    ttnn_reshape_41 = ttnn.reshape(
        ttnn_to_layout_39,
        [1, 8, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_39, False)
    util_create_list_42 = [ttnn_reshape_41]
    return util_create_list_42


def main_const_eval_43(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_43 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_40 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_43,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_to_device_40,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_40, False)
    ttnn_reshape_42 = ttnn.reshape(
        ttnn_to_layout_40,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_40, False)
    util_create_list_43 = [ttnn_reshape_42]
    return util_create_list_43


def main_const_eval_44(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_44 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_41 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_44,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_to_device_41,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_41, False)
    ttnn_reshape_43 = ttnn.reshape(
        ttnn_to_layout_41,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_41, False)
    util_create_list_44 = [ttnn_reshape_43]
    return util_create_list_44


def main_const_eval_45(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_45 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_42 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_45,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_to_device_42,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_42, False)
    ttnn_reshape_44 = ttnn.reshape(
        ttnn_to_layout_42,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_42, False)
    util_create_list_45 = [ttnn_reshape_44]
    return util_create_list_45


def main_const_eval_46(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_46 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_43 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_46,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_to_device_43,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_43, False)
    ttnn_reshape_45 = ttnn.reshape(
        ttnn_to_layout_43,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_43, False)
    util_create_list_46 = [ttnn_reshape_45]
    return util_create_list_46


def main_const_eval_47(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_47 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_44 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_47,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_to_device_44,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_44, False)
    ttnn_reshape_46 = ttnn.reshape(
        ttnn_to_layout_44,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_44, False)
    util_create_list_47 = [ttnn_reshape_46]
    return util_create_list_47


def main_const_eval_48(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_48 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_45 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_48,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_to_device_45,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_45, False)
    ttnn_reshape_47 = ttnn.reshape(
        ttnn_to_layout_45,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_45, False)
    util_create_list_48 = [ttnn_reshape_47]
    return util_create_list_48


def main_const_eval_49(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_49 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_46 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_49,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_to_device_46,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_46, False)
    ttnn_reshape_48 = ttnn.reshape(
        ttnn_to_layout_46,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    util_create_list_49 = [ttnn_reshape_48]
    return util_create_list_49


def main_const_eval_50(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_50 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_47 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_50,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_to_device_47,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_47, False)
    ttnn_reshape_49 = ttnn.reshape(
        ttnn_to_layout_47,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_47, False)
    util_create_list_50 = [ttnn_reshape_49]
    return util_create_list_50


def main_const_eval_51(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_51 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_48 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_51,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_to_device_48,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_48, False)
    ttnn_reshape_50 = ttnn.reshape(
        ttnn_to_layout_48,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_48, False)
    util_create_list_51 = [ttnn_reshape_50]
    return util_create_list_51


def main_const_eval_52(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_52 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_49 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_52,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_49 = ttnn.to_layout(
        ttnn_to_device_49,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_49, False)
    ttnn_reshape_51 = ttnn.reshape(
        ttnn_to_layout_49,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_49, False)
    util_create_list_52 = [ttnn_reshape_51]
    return util_create_list_52


def main_const_eval_53(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_53 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_50 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_53,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_to_device_50,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_50, False)
    ttnn_reshape_52 = ttnn.reshape(
        ttnn_to_layout_50,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_50, False)
    util_create_list_53 = [ttnn_reshape_52]
    return util_create_list_53


def main_const_eval_54(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_54 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_51 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_54,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_to_device_51,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_51, False)
    ttnn_reshape_53 = ttnn.reshape(
        ttnn_to_layout_51,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_51, False)
    util_create_list_54 = [ttnn_reshape_53]
    return util_create_list_54


def main_const_eval_55(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_55 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_52 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_55,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_to_device_52,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_52, False)
    ttnn_reshape_54 = ttnn.reshape(
        ttnn_to_layout_52,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_52, False)
    util_create_list_55 = [ttnn_reshape_54]
    return util_create_list_55


def main_const_eval_56(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_56 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_53 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_56,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_53 = ttnn.to_layout(
        ttnn_to_device_53,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_53, False)
    ttnn_reshape_55 = ttnn.reshape(
        ttnn_to_layout_53,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_53, False)
    util_create_list_56 = [ttnn_reshape_55]
    return util_create_list_56


def main_const_eval_57(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_57 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_54 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_57,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_54 = ttnn.to_layout(
        ttnn_to_device_54,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_54, False)
    ttnn_reshape_56 = ttnn.reshape(
        ttnn_to_layout_54,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_54, False)
    util_create_list_57 = [ttnn_reshape_56]
    return util_create_list_57


def main_const_eval_58(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_58 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_55 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_58,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_55 = ttnn.to_layout(
        ttnn_to_device_55,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_55, False)
    ttnn_reshape_57 = ttnn.reshape(
        ttnn_to_layout_55,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_55, False)
    util_create_list_58 = [ttnn_reshape_57]
    return util_create_list_58


def main_const_eval_59(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_59 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_56 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_59,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_56 = ttnn.to_layout(
        ttnn_to_device_56,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_56, False)
    ttnn_reshape_58 = ttnn.reshape(
        ttnn_to_layout_56,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_56, False)
    util_create_list_59 = [ttnn_reshape_58]
    return util_create_list_59


def main_const_eval_60(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_60 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_57 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_60,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_57 = ttnn.to_layout(
        ttnn_to_device_57,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_57, False)
    ttnn_reshape_59 = ttnn.reshape(
        ttnn_to_layout_57,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_57, False)
    util_create_list_60 = [ttnn_reshape_59]
    return util_create_list_60


def main_const_eval_61(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_61 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_58 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_61,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_58 = ttnn.to_layout(
        ttnn_to_device_58,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_58, False)
    ttnn_reshape_60 = ttnn.reshape(
        ttnn_to_layout_58,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_58, False)
    util_create_list_61 = [ttnn_reshape_60]
    return util_create_list_61


def main_const_eval_62(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_62 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_59 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_62,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_59 = ttnn.to_layout(
        ttnn_to_device_59,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_59, False)
    ttnn_reshape_61 = ttnn.reshape(
        ttnn_to_layout_59,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_59, False)
    util_create_list_62 = [ttnn_reshape_61]
    return util_create_list_62


def main_const_eval_63(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_63 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_60 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_63,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_60 = ttnn.to_layout(
        ttnn_to_device_60,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_60, False)
    ttnn_reshape_62 = ttnn.reshape(
        ttnn_to_layout_60,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_60, False)
    util_create_list_63 = [ttnn_reshape_62]
    return util_create_list_63


def main_const_eval_64(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_64 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_61 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_64,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_61 = ttnn.to_layout(
        ttnn_to_device_61,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_61, False)
    ttnn_reshape_63 = ttnn.reshape(
        ttnn_to_layout_61,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_61, False)
    util_create_list_64 = [ttnn_reshape_63]
    return util_create_list_64


def main_const_eval_65(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_65 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_62 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_65,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_62 = ttnn.to_layout(
        ttnn_to_device_62,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_62, False)
    ttnn_reshape_64 = ttnn.reshape(
        ttnn_to_layout_62,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_62, False)
    util_create_list_65 = [ttnn_reshape_64]
    return util_create_list_65


def main_const_eval_66(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_66 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_63 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_66,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_63 = ttnn.to_layout(
        ttnn_to_device_63,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_63, False)
    ttnn_reshape_65 = ttnn.reshape(
        ttnn_to_layout_63,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_63, False)
    util_create_list_66 = [ttnn_reshape_65]
    return util_create_list_66


def main_const_eval_67(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_67 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_64 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_67,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_64 = ttnn.to_layout(
        ttnn_to_device_64,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_64, False)
    ttnn_reshape_66 = ttnn.reshape(
        ttnn_to_layout_64,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_64, False)
    ttnn_repeat_2 = ttnn.repeat(ttnn_reshape_66, ttnn.Shape([1, 3, 1]))
    ttnn.deallocate(ttnn_reshape_66, False)
    util_create_list_67 = [ttnn_repeat_2]
    return util_create_list_67


def main_const_eval_68(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_68 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_65 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_68,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_65 = ttnn.to_layout(
        ttnn_to_device_65,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_65, False)
    ttnn_reshape_67 = ttnn.reshape(
        ttnn_to_layout_65,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_65, False)
    util_create_list_68 = [ttnn_reshape_67]
    return util_create_list_68


def main_const_eval_69(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_69 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_66 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_69,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_66 = ttnn.to_layout(
        ttnn_to_device_66,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_66, False)
    ttnn_reshape_68 = ttnn.reshape(
        ttnn_to_layout_66,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_66, False)
    util_create_list_69 = [ttnn_reshape_68]
    return util_create_list_69


def main_const_eval_70(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_70 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_67 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_70,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_67 = ttnn.to_layout(
        ttnn_to_device_67,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_67, False)
    ttnn_reshape_69 = ttnn.reshape(
        ttnn_to_layout_67,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_67, False)
    util_create_list_70 = [ttnn_reshape_69]
    return util_create_list_70


def main_const_eval_71(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_71 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_68 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_71,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_68 = ttnn.to_layout(
        ttnn_to_device_68,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_68, False)
    ttnn_reshape_70 = ttnn.reshape(
        ttnn_to_layout_68,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_68, False)
    util_create_list_71 = [ttnn_reshape_70]
    return util_create_list_71


def main_const_eval_72(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_72 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_69 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_72,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_69 = ttnn.to_layout(
        ttnn_to_device_69,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_69, False)
    ttnn_reshape_71 = ttnn.reshape(
        ttnn_to_layout_69,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_69, False)
    util_create_list_72 = [ttnn_reshape_71]
    return util_create_list_72


def main_const_eval_73(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_73 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_70 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_73,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_70 = ttnn.to_layout(
        ttnn_to_device_70,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_70, False)
    ttnn_reshape_72 = ttnn.reshape(
        ttnn_to_layout_70,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_70, False)
    util_create_list_73 = [ttnn_reshape_72]
    return util_create_list_73


def main_const_eval_74(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_74 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_71 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_74,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_71 = ttnn.to_layout(
        ttnn_to_device_71,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_71, False)
    ttnn_reshape_73 = ttnn.reshape(
        ttnn_to_layout_71,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_71, False)
    util_create_list_74 = [ttnn_reshape_73]
    return util_create_list_74


def main_const_eval_75(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_75 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_72 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_75,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_72 = ttnn.to_layout(
        ttnn_to_device_72,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_72, False)
    ttnn_reshape_74 = ttnn.reshape(
        ttnn_to_layout_72,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_72, False)
    util_create_list_75 = [ttnn_reshape_74]
    return util_create_list_75


def main_const_eval_76(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_76 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_73 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_76,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_73 = ttnn.to_layout(
        ttnn_to_device_73,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_73, False)
    ttnn_reshape_75 = ttnn.reshape(
        ttnn_to_layout_73,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_73, False)
    util_create_list_76 = [ttnn_reshape_75]
    return util_create_list_76


def main_const_eval_77(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_77 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_74 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_77,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_74 = ttnn.to_layout(
        ttnn_to_device_74,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_74, False)
    ttnn_reshape_76 = ttnn.reshape(
        ttnn_to_layout_74,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_74, False)
    util_create_list_77 = [ttnn_reshape_76]
    return util_create_list_77


def main_const_eval_78(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_78 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_75 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_78,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_75 = ttnn.to_layout(
        ttnn_to_device_75,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_75, False)
    ttnn_reshape_77 = ttnn.reshape(
        ttnn_to_layout_75,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_75, False)
    util_create_list_78 = [ttnn_reshape_77]
    return util_create_list_78


def main_const_eval_79(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_79 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_76 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_79,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_76 = ttnn.to_layout(
        ttnn_to_device_76,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_76, False)
    ttnn_reshape_78 = ttnn.reshape(
        ttnn_to_layout_76,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_76, False)
    util_create_list_79 = [ttnn_reshape_78]
    return util_create_list_79


def main_const_eval_80(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_80 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_77 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_80,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_77 = ttnn.to_layout(
        ttnn_to_device_77,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_77, False)
    ttnn_reshape_79 = ttnn.reshape(
        ttnn_to_layout_77,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_77, False)
    ttnn_repeat_3 = ttnn.repeat(ttnn_reshape_79, ttnn.Shape([1, 3, 1]))
    ttnn.deallocate(ttnn_reshape_79, False)
    util_create_list_80 = [ttnn_repeat_3]
    return util_create_list_80


def main_const_eval_81(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_81 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_78 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_81,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_78 = ttnn.to_layout(
        ttnn_to_device_78,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_78, False)
    ttnn_reshape_80 = ttnn.reshape(
        ttnn_to_layout_78,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_78, False)
    util_create_list_81 = [ttnn_reshape_80]
    return util_create_list_81


def main_const_eval_82(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_82 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_79 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_82,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_79 = ttnn.to_layout(
        ttnn_to_device_79,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_79, False)
    ttnn_reshape_81 = ttnn.reshape(
        ttnn_to_layout_79,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_79, False)
    util_create_list_82 = [ttnn_reshape_81]
    return util_create_list_82


def main_const_eval_83(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_83 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_80 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_83,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_80 = ttnn.to_layout(
        ttnn_to_device_80,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_80, False)
    ttnn_reshape_82 = ttnn.reshape(
        ttnn_to_layout_80,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_80, False)
    util_create_list_83 = [ttnn_reshape_82]
    return util_create_list_83


def main_const_eval_84(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_84 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_81 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_84,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_81 = ttnn.to_layout(
        ttnn_to_device_81,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_81, False)
    ttnn_reshape_83 = ttnn.reshape(
        ttnn_to_layout_81,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_81, False)
    util_create_list_84 = [ttnn_reshape_83]
    return util_create_list_84


def main_const_eval_85(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_85 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_82 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_85,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_82 = ttnn.to_layout(
        ttnn_to_device_82,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_82, False)
    ttnn_reshape_84 = ttnn.reshape(
        ttnn_to_layout_82,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_82, False)
    util_create_list_85 = [ttnn_reshape_84]
    return util_create_list_85


def main_const_eval_86(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_86 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_83 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_86,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_83 = ttnn.to_layout(
        ttnn_to_device_83,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_83, False)
    ttnn_reshape_85 = ttnn.reshape(
        ttnn_to_layout_83,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_83, False)
    util_create_list_86 = [ttnn_reshape_85]
    return util_create_list_86


def main_const_eval_87(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_87 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_84 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_87,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_84 = ttnn.to_layout(
        ttnn_to_device_84,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_84, False)
    ttnn_reshape_86 = ttnn.reshape(
        ttnn_to_layout_84,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_84, False)
    util_create_list_87 = [ttnn_reshape_86]
    return util_create_list_87


def main_const_eval_88(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_88 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_85 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_88,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_85 = ttnn.to_layout(
        ttnn_to_device_85,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_85, False)
    ttnn_reshape_87 = ttnn.reshape(
        ttnn_to_layout_85,
        [1, 4, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_85, False)
    util_create_list_88 = [ttnn_reshape_87]
    return util_create_list_88


def main_const_eval_89(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_89 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_86 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_89,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_86 = ttnn.to_layout(
        ttnn_to_device_86,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_86, False)
    ttnn_reshape_88 = ttnn.reshape(
        ttnn_to_layout_86,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_86, False)
    util_create_list_89 = [ttnn_reshape_88]
    return util_create_list_89


def main_const_eval_90(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_90 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_87 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_90,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_87 = ttnn.to_layout(
        ttnn_to_device_87,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_87, False)
    ttnn_reshape_89 = ttnn.reshape(
        ttnn_to_layout_87,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_87, False)
    util_create_list_90 = [ttnn_reshape_89]
    return util_create_list_90


def main_const_eval_91(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_91 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_88 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_91,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_88 = ttnn.to_layout(
        ttnn_to_device_88,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_88, False)
    ttnn_reshape_90 = ttnn.reshape(
        ttnn_to_layout_88,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_88, False)
    util_create_list_91 = [ttnn_reshape_90]
    return util_create_list_91


def main_const_eval_92(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_92 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_89 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_92,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_89 = ttnn.to_layout(
        ttnn_to_device_89,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_89, False)
    ttnn_reshape_91 = ttnn.reshape(
        ttnn_to_layout_89,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_89, False)
    util_create_list_92 = [ttnn_reshape_91]
    return util_create_list_92


def main_const_eval_93(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_93 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_90 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_93,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_90 = ttnn.to_layout(
        ttnn_to_device_90,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_90, False)
    ttnn_reshape_92 = ttnn.reshape(
        ttnn_to_layout_90,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_90, False)
    util_create_list_93 = [ttnn_reshape_92]
    return util_create_list_93


def main_const_eval_94():
    utils_DeviceGetter_get_device_94 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 2, 80, 80]),
        fill_value=5.65625,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_94,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_94 = [ttnn_full_2]
    return util_create_list_94


def main_const_eval_95():
    utils_DeviceGetter_get_device_95 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_2 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        ],
        [40],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_95,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_Tensor_3 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.5,
            31.0,
            31.5,
            32.0,
            32.5,
            33.0,
            33.5,
            34.0,
            34.5,
            35.0,
            35.5,
            36.0,
            36.5,
            37.0,
            37.5,
            38.0,
            38.5,
            39.0,
            39.5,
        ],
        [80],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_95,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_floor_1 = ttnn.floor(
        ttnn_Tensor_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_Tensor_3, False)
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_floor_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_floor_1, False)
    ttnn_reshape_93 = ttnn.reshape(
        ttnn_typecast_5,
        [80, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_reshape_93,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_93, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_typecast_6,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_permute_2,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_reshape_94 = ttnn.reshape(
        ttnn_Tensor_2,
        [1, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_Tensor_2, False)
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_reshape_94,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_94, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_typecast_8,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_8, False)
    ttnn_typecast_9 = ttnn.typecast(
        ttnn_permute_3,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_eq_1 = ttnn.eq(
        ttnn_typecast_7,
        ttnn_typecast_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn.deallocate(ttnn_typecast_7, False)
    util_create_list_95 = [ttnn_eq_1]
    return util_create_list_95


def main_const_eval_96(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_96 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_91 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_96,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_91 = ttnn.to_layout(
        ttnn_to_device_91,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_91, False)
    ttnn_reshape_95 = ttnn.reshape(
        ttnn_to_layout_91,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_91, False)
    util_create_list_96 = [ttnn_reshape_95]
    return util_create_list_96


def main_const_eval_97(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_97 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_92 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_97,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_92 = ttnn.to_layout(
        ttnn_to_device_92,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_92, False)
    ttnn_reshape_96 = ttnn.reshape(
        ttnn_to_layout_92,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_92, False)
    util_create_list_97 = [ttnn_reshape_96]
    return util_create_list_97


def main_const_eval_98(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_98 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_93 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_98,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_93 = ttnn.to_layout(
        ttnn_to_device_93,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_93, False)
    ttnn_reshape_97 = ttnn.reshape(
        ttnn_to_layout_93,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_93, False)
    util_create_list_98 = [ttnn_reshape_97]
    return util_create_list_98


def main_const_eval_99(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_99 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_94 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_99,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_94 = ttnn.to_layout(
        ttnn_to_device_94,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_94, False)
    ttnn_reshape_98 = ttnn.reshape(
        ttnn_to_layout_94,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_94, False)
    util_create_list_99 = [ttnn_reshape_98]
    return util_create_list_99


def main_const_eval_100(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_100 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_95 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_100,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_95 = ttnn.to_layout(
        ttnn_to_device_95,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_95, False)
    ttnn_reshape_99 = ttnn.reshape(
        ttnn_to_layout_95,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_95, False)
    util_create_list_100 = [ttnn_reshape_99]
    return util_create_list_100


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None
CACHED_main_const_eval_2 = None
CACHED_main_const_eval_3 = None
CACHED_main_const_eval_4 = None
CACHED_main_const_eval_5 = None
CACHED_main_const_eval_6 = None
CACHED_main_const_eval_7 = None
CACHED_main_const_eval_8 = None
CACHED_main_const_eval_9 = None
CACHED_main_const_eval_10 = None
CACHED_main_const_eval_11 = None
CACHED_main_const_eval_12 = None
CACHED_main_const_eval_13 = None
CACHED_main_const_eval_14 = None
CACHED_main_const_eval_15 = None
CACHED_main_const_eval_16 = None
CACHED_main_const_eval_17 = None
CACHED_main_const_eval_18 = None
CACHED_main_const_eval_19 = None
CACHED_main_const_eval_20 = None
CACHED_main_const_eval_21 = None
CACHED_main_const_eval_22 = None
CACHED_main_const_eval_23 = None
CACHED_main_const_eval_24 = None
CACHED_main_const_eval_25 = None
CACHED_main_const_eval_26 = None
CACHED_main_const_eval_27 = None
CACHED_main_const_eval_28 = None
CACHED_main_const_eval_29 = None
CACHED_main_const_eval_30 = None
CACHED_main_const_eval_31 = None
CACHED_main_const_eval_32 = None
CACHED_main_const_eval_33 = None
CACHED_main_const_eval_34 = None
CACHED_main_const_eval_35 = None
CACHED_main_const_eval_36 = None
CACHED_main_const_eval_37 = None
CACHED_main_const_eval_38 = None
CACHED_main_const_eval_39 = None
CACHED_main_const_eval_40 = None
CACHED_main_const_eval_41 = None
CACHED_main_const_eval_42 = None
CACHED_main_const_eval_43 = None
CACHED_main_const_eval_44 = None
CACHED_main_const_eval_45 = None
CACHED_main_const_eval_46 = None
CACHED_main_const_eval_47 = None
CACHED_main_const_eval_48 = None
CACHED_main_const_eval_49 = None
CACHED_main_const_eval_50 = None
CACHED_main_const_eval_51 = None
CACHED_main_const_eval_52 = None
CACHED_main_const_eval_53 = None
CACHED_main_const_eval_54 = None
CACHED_main_const_eval_55 = None
CACHED_main_const_eval_56 = None
CACHED_main_const_eval_57 = None
CACHED_main_const_eval_58 = None
CACHED_main_const_eval_59 = None
CACHED_main_const_eval_60 = None
CACHED_main_const_eval_61 = None
CACHED_main_const_eval_62 = None
CACHED_main_const_eval_63 = None
CACHED_main_const_eval_64 = None
CACHED_main_const_eval_65 = None
CACHED_main_const_eval_66 = None
CACHED_main_const_eval_67 = None
CACHED_main_const_eval_68 = None
CACHED_main_const_eval_69 = None
CACHED_main_const_eval_70 = None
CACHED_main_const_eval_71 = None
CACHED_main_const_eval_72 = None
CACHED_main_const_eval_73 = None
CACHED_main_const_eval_74 = None
CACHED_main_const_eval_75 = None
CACHED_main_const_eval_76 = None
CACHED_main_const_eval_77 = None
CACHED_main_const_eval_78 = None
CACHED_main_const_eval_79 = None
CACHED_main_const_eval_80 = None
CACHED_main_const_eval_81 = None
CACHED_main_const_eval_82 = None
CACHED_main_const_eval_83 = None
CACHED_main_const_eval_84 = None
CACHED_main_const_eval_85 = None
CACHED_main_const_eval_86 = None
CACHED_main_const_eval_87 = None
CACHED_main_const_eval_88 = None
CACHED_main_const_eval_89 = None
CACHED_main_const_eval_90 = None
CACHED_main_const_eval_91 = None
CACHED_main_const_eval_92 = None
CACHED_main_const_eval_93 = None
CACHED_main_const_eval_94 = None
CACHED_main_const_eval_95 = None
CACHED_main_const_eval_96 = None
CACHED_main_const_eval_97 = None
CACHED_main_const_eval_98 = None
CACHED_main_const_eval_99 = None
CACHED_main_const_eval_100 = None


def _main(input):
    global CACHED_main_const_eval_100
    global CACHED_main_const_eval_99
    global CACHED_main_const_eval_98
    global CACHED_main_const_eval_97
    global CACHED_main_const_eval_96
    global CACHED_main_const_eval_95
    global CACHED_main_const_eval_94
    global CACHED_main_const_eval_93
    global CACHED_main_const_eval_92
    global CACHED_main_const_eval_91
    global CACHED_main_const_eval_90
    global CACHED_main_const_eval_89
    global CACHED_main_const_eval_88
    global CACHED_main_const_eval_87
    global CACHED_main_const_eval_86
    global CACHED_main_const_eval_85
    global CACHED_main_const_eval_84
    global CACHED_main_const_eval_83
    global CACHED_main_const_eval_82
    global CACHED_main_const_eval_81
    global CACHED_main_const_eval_80
    global CACHED_main_const_eval_79
    global CACHED_main_const_eval_78
    global CACHED_main_const_eval_77
    global CACHED_main_const_eval_76
    global CACHED_main_const_eval_75
    global CACHED_main_const_eval_74
    global CACHED_main_const_eval_73
    global CACHED_main_const_eval_72
    global CACHED_main_const_eval_71
    global CACHED_main_const_eval_70
    global CACHED_main_const_eval_69
    global CACHED_main_const_eval_68
    global CACHED_main_const_eval_67
    global CACHED_main_const_eval_66
    global CACHED_main_const_eval_65
    global CACHED_main_const_eval_64
    global CACHED_main_const_eval_63
    global CACHED_main_const_eval_62
    global CACHED_main_const_eval_61
    global CACHED_main_const_eval_60
    global CACHED_main_const_eval_59
    global CACHED_main_const_eval_58
    global CACHED_main_const_eval_57
    global CACHED_main_const_eval_56
    global CACHED_main_const_eval_55
    global CACHED_main_const_eval_54
    global CACHED_main_const_eval_53
    global CACHED_main_const_eval_52
    global CACHED_main_const_eval_51
    global CACHED_main_const_eval_50
    global CACHED_main_const_eval_49
    global CACHED_main_const_eval_48
    global CACHED_main_const_eval_47
    global CACHED_main_const_eval_46
    global CACHED_main_const_eval_45
    global CACHED_main_const_eval_44
    global CACHED_main_const_eval_43
    global CACHED_main_const_eval_42
    global CACHED_main_const_eval_41
    global CACHED_main_const_eval_40
    global CACHED_main_const_eval_39
    global CACHED_main_const_eval_38
    global CACHED_main_const_eval_37
    global CACHED_main_const_eval_36
    global CACHED_main_const_eval_35
    global CACHED_main_const_eval_34
    global CACHED_main_const_eval_33
    global CACHED_main_const_eval_32
    global CACHED_main_const_eval_31
    global CACHED_main_const_eval_30
    global CACHED_main_const_eval_29
    global CACHED_main_const_eval_28
    global CACHED_main_const_eval_27
    global CACHED_main_const_eval_26
    global CACHED_main_const_eval_25
    global CACHED_main_const_eval_24
    global CACHED_main_const_eval_23
    global CACHED_main_const_eval_22
    global CACHED_main_const_eval_21
    global CACHED_main_const_eval_20
    global CACHED_main_const_eval_19
    global CACHED_main_const_eval_18
    global CACHED_main_const_eval_17
    global CACHED_main_const_eval_16
    global CACHED_main_const_eval_15
    global CACHED_main_const_eval_14
    global CACHED_main_const_eval_13
    global CACHED_main_const_eval_12
    global CACHED_main_const_eval_11
    global CACHED_main_const_eval_10
    global CACHED_main_const_eval_9
    global CACHED_main_const_eval_8
    global CACHED_main_const_eval_7
    global CACHED_main_const_eval_6
    global CACHED_main_const_eval_5
    global CACHED_main_const_eval_4
    global CACHED_main_const_eval_3
    global CACHED_main_const_eval_2
    global CACHED_main_const_eval_1
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    input_3 = input[3]
    input_4 = input[4]
    input_5 = input[5]
    input_6 = input[6]
    input_7 = input[7]
    input_8 = input[8]
    input_9 = input[9]
    input_10 = input[10]
    input_11 = input[11]
    input_12 = input[12]
    input_13 = input[13]
    input_14 = input[14]
    input_15 = input[15]
    input_16 = input[16]
    input_17 = input[17]
    input_18 = input[18]
    input_19 = input[19]
    input_20 = input[20]
    input_21 = input[21]
    input_22 = input[22]
    input_23 = input[23]
    input_24 = input[24]
    input_25 = input[25]
    input_26 = input[26]
    input_27 = input[27]
    input_28 = input[28]
    input_29 = input[29]
    input_30 = input[30]
    input_31 = input[31]
    input_32 = input[32]
    input_33 = input[33]
    input_34 = input[34]
    input_35 = input[35]
    input_36 = input[36]
    input_37 = input[37]
    input_38 = input[38]
    input_39 = input[39]
    input_40 = input[40]
    input_41 = input[41]
    input_42 = input[42]
    input_43 = input[43]
    input_44 = input[44]
    input_45 = input[45]
    input_46 = input[46]
    input_47 = input[47]
    input_48 = input[48]
    input_49 = input[49]
    input_50 = input[50]
    input_51 = input[51]
    input_52 = input[52]
    input_53 = input[53]
    input_54 = input[54]
    input_55 = input[55]
    input_56 = input[56]
    input_57 = input[57]
    input_58 = input[58]
    input_59 = input[59]
    input_60 = input[60]
    input_61 = input[61]
    input_62 = input[62]
    input_63 = input[63]
    input_64 = input[64]
    input_65 = input[65]
    input_66 = input[66]
    input_67 = input[67]
    input_68 = input[68]
    input_69 = input[69]
    input_70 = input[70]
    input_71 = input[71]
    input_72 = input[72]
    input_73 = input[73]
    input_74 = input[74]
    input_75 = input[75]
    input_76 = input[76]
    input_77 = input[77]
    input_78 = input[78]
    input_79 = input[79]
    input_80 = input[80]
    input_81 = input[81]
    input_82 = input[82]
    input_83 = input[83]
    input_84 = input[84]
    input_85 = input[85]
    input_86 = input[86]
    input_87 = input[87]
    input_88 = input[88]
    input_89 = input[89]
    input_90 = input[90]
    input_91 = input[91]
    input_92 = input[92]
    input_93 = input[93]
    input_94 = input[94]
    input_95 = input[95]
    input_96 = input[96]
    input_97 = input[97]
    input_98 = input[98]
    input_99 = input[99]
    input_100 = input[100]
    input_101 = input[101]
    input_102 = input[102]
    input_103 = input[103]
    input_104 = input[104]
    input_105 = input[105]
    input_106 = input[106]
    input_107 = input[107]
    input_108 = input[108]
    input_109 = input[109]
    input_110 = input[110]
    input_111 = input[111]
    input_112 = input[112]
    input_113 = input[113]
    input_114 = input[114]
    input_115 = input[115]
    input_116 = input[116]
    input_117 = input[117]
    input_118 = input[118]
    input_119 = input[119]
    input_120 = input[120]
    input_121 = input[121]
    input_122 = input[122]
    input_123 = input[123]
    input_124 = input[124]
    input_125 = input[125]
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(const_0, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_1 = main_const_eval_1
    util_create_list_101 = [input_12]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(const_1, util_create_list_101, CACHED_main_const_eval_1)
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_2 = main_const_eval_2
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(const_2, CACHED_main_const_eval_2)
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapperZeroArg_1
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_3 = main_const_eval_3
    util_create_list_102 = [input_34]
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(const_3, util_create_list_102, CACHED_main_const_eval_3)
    CACHED_main_const_eval_3 = utils_constEvalFuncWrapper_1
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_4 = main_const_eval_4
    util_create_list_103 = [input_99]
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(const_4, util_create_list_103, CACHED_main_const_eval_4)
    CACHED_main_const_eval_4 = utils_constEvalFuncWrapper_2
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_5 = main_const_eval_5
    util_create_list_104 = [input_103]
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(const_5, util_create_list_104, CACHED_main_const_eval_5)
    CACHED_main_const_eval_5 = utils_constEvalFuncWrapper_3
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_6 = main_const_eval_6
    util_create_list_105 = [input_50]
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(const_6, util_create_list_105, CACHED_main_const_eval_6)
    CACHED_main_const_eval_6 = utils_constEvalFuncWrapper_4
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    const_7 = main_const_eval_7
    util_create_list_106 = [input_69]
    utils_constEvalFuncWrapper_5 = utils.constEvalFuncWrapper(const_7, util_create_list_106, CACHED_main_const_eval_7)
    CACHED_main_const_eval_7 = utils_constEvalFuncWrapper_5
    utils_constEvalFuncWrapper_5_0 = utils_constEvalFuncWrapper_5[0]
    const_8 = main_const_eval_8
    util_create_list_107 = [input_79]
    utils_constEvalFuncWrapper_6 = utils.constEvalFuncWrapper(const_8, util_create_list_107, CACHED_main_const_eval_8)
    CACHED_main_const_eval_8 = utils_constEvalFuncWrapper_6
    utils_constEvalFuncWrapper_6_0 = utils_constEvalFuncWrapper_6[0]
    const_9 = main_const_eval_9
    util_create_list_108 = [input_28]
    utils_constEvalFuncWrapper_7 = utils.constEvalFuncWrapper(const_9, util_create_list_108, CACHED_main_const_eval_9)
    CACHED_main_const_eval_9 = utils_constEvalFuncWrapper_7
    utils_constEvalFuncWrapper_7_0 = utils_constEvalFuncWrapper_7[0]
    const_10 = main_const_eval_10
    util_create_list_109 = [input_101]
    utils_constEvalFuncWrapper_8 = utils.constEvalFuncWrapper(const_10, util_create_list_109, CACHED_main_const_eval_10)
    CACHED_main_const_eval_10 = utils_constEvalFuncWrapper_8
    utils_constEvalFuncWrapper_8_0 = utils_constEvalFuncWrapper_8[0]
    const_11 = main_const_eval_11
    util_create_list_110 = [input_117]
    utils_constEvalFuncWrapper_9 = utils.constEvalFuncWrapper(const_11, util_create_list_110, CACHED_main_const_eval_11)
    CACHED_main_const_eval_11 = utils_constEvalFuncWrapper_9
    utils_constEvalFuncWrapper_9_0 = utils_constEvalFuncWrapper_9[0]
    const_12 = main_const_eval_12
    util_create_list_111 = [input_6]
    utils_constEvalFuncWrapper_10 = utils.constEvalFuncWrapper(
        const_12, util_create_list_111, CACHED_main_const_eval_12
    )
    CACHED_main_const_eval_12 = utils_constEvalFuncWrapper_10
    utils_constEvalFuncWrapper_10_0 = utils_constEvalFuncWrapper_10[0]
    const_13 = main_const_eval_13
    util_create_list_112 = [input_116]
    utils_constEvalFuncWrapper_11 = utils.constEvalFuncWrapper(
        const_13, util_create_list_112, CACHED_main_const_eval_13
    )
    CACHED_main_const_eval_13 = utils_constEvalFuncWrapper_11
    utils_constEvalFuncWrapper_11_0 = utils_constEvalFuncWrapper_11[0]
    const_14 = main_const_eval_14
    util_create_list_113 = [input_85]
    utils_constEvalFuncWrapper_12 = utils.constEvalFuncWrapper(
        const_14, util_create_list_113, CACHED_main_const_eval_14
    )
    CACHED_main_const_eval_14 = utils_constEvalFuncWrapper_12
    utils_constEvalFuncWrapper_12_0 = utils_constEvalFuncWrapper_12[0]
    const_15 = main_const_eval_15
    util_create_list_114 = [input_80]
    utils_constEvalFuncWrapper_13 = utils.constEvalFuncWrapper(
        const_15, util_create_list_114, CACHED_main_const_eval_15
    )
    CACHED_main_const_eval_15 = utils_constEvalFuncWrapper_13
    utils_constEvalFuncWrapper_13_0 = utils_constEvalFuncWrapper_13[0]
    const_16 = main_const_eval_16
    util_create_list_115 = [input_71]
    utils_constEvalFuncWrapper_14 = utils.constEvalFuncWrapper(
        const_16, util_create_list_115, CACHED_main_const_eval_16
    )
    CACHED_main_const_eval_16 = utils_constEvalFuncWrapper_14
    utils_constEvalFuncWrapper_14_0 = utils_constEvalFuncWrapper_14[0]
    const_17 = main_const_eval_17
    util_create_list_116 = [input_11]
    utils_constEvalFuncWrapper_15 = utils.constEvalFuncWrapper(
        const_17, util_create_list_116, CACHED_main_const_eval_17
    )
    CACHED_main_const_eval_17 = utils_constEvalFuncWrapper_15
    utils_constEvalFuncWrapper_15_0 = utils_constEvalFuncWrapper_15[0]
    const_18 = main_const_eval_18
    util_create_list_117 = [input_107]
    utils_constEvalFuncWrapper_16 = utils.constEvalFuncWrapper(
        const_18, util_create_list_117, CACHED_main_const_eval_18
    )
    CACHED_main_const_eval_18 = utils_constEvalFuncWrapper_16
    utils_constEvalFuncWrapper_16_0 = utils_constEvalFuncWrapper_16[0]
    const_19 = main_const_eval_19
    util_create_list_118 = [input_1]
    utils_constEvalFuncWrapper_17 = utils.constEvalFuncWrapper(
        const_19, util_create_list_118, CACHED_main_const_eval_19
    )
    CACHED_main_const_eval_19 = utils_constEvalFuncWrapper_17
    utils_constEvalFuncWrapper_17_0 = utils_constEvalFuncWrapper_17[0]
    const_20 = main_const_eval_20
    util_create_list_119 = [input_30]
    utils_constEvalFuncWrapper_18 = utils.constEvalFuncWrapper(
        const_20, util_create_list_119, CACHED_main_const_eval_20
    )
    CACHED_main_const_eval_20 = utils_constEvalFuncWrapper_18
    utils_constEvalFuncWrapper_18_0 = utils_constEvalFuncWrapper_18[0]
    const_21 = main_const_eval_21
    util_create_list_120 = [input_39]
    utils_constEvalFuncWrapper_19 = utils.constEvalFuncWrapper(
        const_21, util_create_list_120, CACHED_main_const_eval_21
    )
    CACHED_main_const_eval_21 = utils_constEvalFuncWrapper_19
    utils_constEvalFuncWrapper_19_0 = utils_constEvalFuncWrapper_19[0]
    const_22 = main_const_eval_22
    util_create_list_121 = [input_10]
    utils_constEvalFuncWrapper_20 = utils.constEvalFuncWrapper(
        const_22, util_create_list_121, CACHED_main_const_eval_22
    )
    CACHED_main_const_eval_22 = utils_constEvalFuncWrapper_20
    utils_constEvalFuncWrapper_20_0 = utils_constEvalFuncWrapper_20[0]
    const_23 = main_const_eval_23
    util_create_list_122 = [input_81]
    utils_constEvalFuncWrapper_21 = utils.constEvalFuncWrapper(
        const_23, util_create_list_122, CACHED_main_const_eval_23
    )
    CACHED_main_const_eval_23 = utils_constEvalFuncWrapper_21
    utils_constEvalFuncWrapper_21_0 = utils_constEvalFuncWrapper_21[0]
    const_24 = main_const_eval_24
    util_create_list_123 = [input_91]
    utils_constEvalFuncWrapper_22 = utils.constEvalFuncWrapper(
        const_24, util_create_list_123, CACHED_main_const_eval_24
    )
    CACHED_main_const_eval_24 = utils_constEvalFuncWrapper_22
    utils_constEvalFuncWrapper_22_0 = utils_constEvalFuncWrapper_22[0]
    const_25 = main_const_eval_25
    util_create_list_124 = [input_5]
    utils_constEvalFuncWrapper_23 = utils.constEvalFuncWrapper(
        const_25, util_create_list_124, CACHED_main_const_eval_25
    )
    CACHED_main_const_eval_25 = utils_constEvalFuncWrapper_23
    utils_constEvalFuncWrapper_23_0 = utils_constEvalFuncWrapper_23[0]
    const_26 = main_const_eval_26
    util_create_list_125 = [input_9]
    utils_constEvalFuncWrapper_24 = utils.constEvalFuncWrapper(
        const_26, util_create_list_125, CACHED_main_const_eval_26
    )
    CACHED_main_const_eval_26 = utils_constEvalFuncWrapper_24
    utils_constEvalFuncWrapper_24_0 = utils_constEvalFuncWrapper_24[0]
    const_27 = main_const_eval_27
    util_create_list_126 = [input_61]
    utils_constEvalFuncWrapper_25 = utils.constEvalFuncWrapper(
        const_27, util_create_list_126, CACHED_main_const_eval_27
    )
    CACHED_main_const_eval_27 = utils_constEvalFuncWrapper_25
    utils_constEvalFuncWrapper_25_0 = utils_constEvalFuncWrapper_25[0]
    const_28 = main_const_eval_28
    util_create_list_127 = [input_78]
    utils_constEvalFuncWrapper_26 = utils.constEvalFuncWrapper(
        const_28, util_create_list_127, CACHED_main_const_eval_28
    )
    CACHED_main_const_eval_28 = utils_constEvalFuncWrapper_26
    utils_constEvalFuncWrapper_26_0 = utils_constEvalFuncWrapper_26[0]
    const_29 = main_const_eval_29
    util_create_list_128 = [input_22]
    utils_constEvalFuncWrapper_27 = utils.constEvalFuncWrapper(
        const_29, util_create_list_128, CACHED_main_const_eval_29
    )
    CACHED_main_const_eval_29 = utils_constEvalFuncWrapper_27
    utils_constEvalFuncWrapper_27_0 = utils_constEvalFuncWrapper_27[0]
    const_30 = main_const_eval_30
    util_create_list_129 = [input_108]
    utils_constEvalFuncWrapper_28 = utils.constEvalFuncWrapper(
        const_30, util_create_list_129, CACHED_main_const_eval_30
    )
    CACHED_main_const_eval_30 = utils_constEvalFuncWrapper_28
    utils_constEvalFuncWrapper_28_0 = utils_constEvalFuncWrapper_28[0]
    const_31 = main_const_eval_31
    util_create_list_130 = [input_76]
    utils_constEvalFuncWrapper_29 = utils.constEvalFuncWrapper(
        const_31, util_create_list_130, CACHED_main_const_eval_31
    )
    CACHED_main_const_eval_31 = utils_constEvalFuncWrapper_29
    utils_constEvalFuncWrapper_29_0 = utils_constEvalFuncWrapper_29[0]
    const_32 = main_const_eval_32
    util_create_list_131 = [input_14]
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(
        const_32, util_create_list_131, CACHED_main_const_eval_32
    )
    CACHED_main_const_eval_32 = utils_constEvalFuncWrapper_30
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    const_33 = main_const_eval_33
    util_create_list_132 = [input_46]
    utils_constEvalFuncWrapper_31 = utils.constEvalFuncWrapper(
        const_33, util_create_list_132, CACHED_main_const_eval_33
    )
    CACHED_main_const_eval_33 = utils_constEvalFuncWrapper_31
    utils_constEvalFuncWrapper_31_0 = utils_constEvalFuncWrapper_31[0]
    const_34 = main_const_eval_34
    util_create_list_133 = [input_25]
    utils_constEvalFuncWrapper_32 = utils.constEvalFuncWrapper(
        const_34, util_create_list_133, CACHED_main_const_eval_34
    )
    CACHED_main_const_eval_34 = utils_constEvalFuncWrapper_32
    utils_constEvalFuncWrapper_32_0 = utils_constEvalFuncWrapper_32[0]
    const_35 = main_const_eval_35
    util_create_list_134 = [input_88]
    utils_constEvalFuncWrapper_33 = utils.constEvalFuncWrapper(
        const_35, util_create_list_134, CACHED_main_const_eval_35
    )
    CACHED_main_const_eval_35 = utils_constEvalFuncWrapper_33
    utils_constEvalFuncWrapper_33_0 = utils_constEvalFuncWrapper_33[0]
    const_36 = main_const_eval_36
    util_create_list_135 = [input_56]
    utils_constEvalFuncWrapper_34 = utils.constEvalFuncWrapper(
        const_36, util_create_list_135, CACHED_main_const_eval_36
    )
    CACHED_main_const_eval_36 = utils_constEvalFuncWrapper_34
    utils_constEvalFuncWrapper_34_0 = utils_constEvalFuncWrapper_34[0]
    const_37 = main_const_eval_37
    utils_constEvalFuncWrapperZeroArg_2 = utils.constEvalFuncWrapperZeroArg(const_37, CACHED_main_const_eval_37)
    CACHED_main_const_eval_37 = utils_constEvalFuncWrapperZeroArg_2
    utils_constEvalFuncWrapperZeroArg_2_0 = utils_constEvalFuncWrapperZeroArg_2[0]
    const_38 = main_const_eval_38
    util_create_list_136 = [input_21]
    utils_constEvalFuncWrapper_35 = utils.constEvalFuncWrapper(
        const_38, util_create_list_136, CACHED_main_const_eval_38
    )
    CACHED_main_const_eval_38 = utils_constEvalFuncWrapper_35
    utils_constEvalFuncWrapper_35_0 = utils_constEvalFuncWrapper_35[0]
    const_39 = main_const_eval_39
    util_create_list_137 = [input_20]
    utils_constEvalFuncWrapper_36 = utils.constEvalFuncWrapper(
        const_39, util_create_list_137, CACHED_main_const_eval_39
    )
    CACHED_main_const_eval_39 = utils_constEvalFuncWrapper_36
    utils_constEvalFuncWrapper_36_0 = utils_constEvalFuncWrapper_36[0]
    const_40 = main_const_eval_40
    util_create_list_138 = [input_26]
    utils_constEvalFuncWrapper_37 = utils.constEvalFuncWrapper(
        const_40, util_create_list_138, CACHED_main_const_eval_40
    )
    CACHED_main_const_eval_40 = utils_constEvalFuncWrapper_37
    utils_constEvalFuncWrapper_37_0 = utils_constEvalFuncWrapper_37[0]
    const_41 = main_const_eval_41
    util_create_list_139 = [input_55]
    utils_constEvalFuncWrapper_38 = utils.constEvalFuncWrapper(
        const_41, util_create_list_139, CACHED_main_const_eval_41
    )
    CACHED_main_const_eval_41 = utils_constEvalFuncWrapper_38
    utils_constEvalFuncWrapper_38_0 = utils_constEvalFuncWrapper_38[0]
    const_42 = main_const_eval_42
    util_create_list_140 = [input_98]
    utils_constEvalFuncWrapper_39 = utils.constEvalFuncWrapper(
        const_42, util_create_list_140, CACHED_main_const_eval_42
    )
    CACHED_main_const_eval_42 = utils_constEvalFuncWrapper_39
    utils_constEvalFuncWrapper_39_0 = utils_constEvalFuncWrapper_39[0]
    const_43 = main_const_eval_43
    util_create_list_141 = [input_51]
    utils_constEvalFuncWrapper_40 = utils.constEvalFuncWrapper(
        const_43, util_create_list_141, CACHED_main_const_eval_43
    )
    CACHED_main_const_eval_43 = utils_constEvalFuncWrapper_40
    utils_constEvalFuncWrapper_40_0 = utils_constEvalFuncWrapper_40[0]
    const_44 = main_const_eval_44
    util_create_list_142 = [input_44]
    utils_constEvalFuncWrapper_41 = utils.constEvalFuncWrapper(
        const_44, util_create_list_142, CACHED_main_const_eval_44
    )
    CACHED_main_const_eval_44 = utils_constEvalFuncWrapper_41
    utils_constEvalFuncWrapper_41_0 = utils_constEvalFuncWrapper_41[0]
    const_45 = main_const_eval_45
    util_create_list_143 = [input_38]
    utils_constEvalFuncWrapper_42 = utils.constEvalFuncWrapper(
        const_45, util_create_list_143, CACHED_main_const_eval_45
    )
    CACHED_main_const_eval_45 = utils_constEvalFuncWrapper_42
    utils_constEvalFuncWrapper_42_0 = utils_constEvalFuncWrapper_42[0]
    const_46 = main_const_eval_46
    util_create_list_144 = [input_36]
    utils_constEvalFuncWrapper_43 = utils.constEvalFuncWrapper(
        const_46, util_create_list_144, CACHED_main_const_eval_46
    )
    CACHED_main_const_eval_46 = utils_constEvalFuncWrapper_43
    utils_constEvalFuncWrapper_43_0 = utils_constEvalFuncWrapper_43[0]
    const_47 = main_const_eval_47
    util_create_list_145 = [input_96]
    utils_constEvalFuncWrapper_44 = utils.constEvalFuncWrapper(
        const_47, util_create_list_145, CACHED_main_const_eval_47
    )
    CACHED_main_const_eval_47 = utils_constEvalFuncWrapper_44
    utils_constEvalFuncWrapper_44_0 = utils_constEvalFuncWrapper_44[0]
    const_48 = main_const_eval_48
    util_create_list_146 = [input_74]
    utils_constEvalFuncWrapper_45 = utils.constEvalFuncWrapper(
        const_48, util_create_list_146, CACHED_main_const_eval_48
    )
    CACHED_main_const_eval_48 = utils_constEvalFuncWrapper_45
    utils_constEvalFuncWrapper_45_0 = utils_constEvalFuncWrapper_45[0]
    const_49 = main_const_eval_49
    util_create_list_147 = [input_35]
    utils_constEvalFuncWrapper_46 = utils.constEvalFuncWrapper(
        const_49, util_create_list_147, CACHED_main_const_eval_49
    )
    CACHED_main_const_eval_49 = utils_constEvalFuncWrapper_46
    utils_constEvalFuncWrapper_46_0 = utils_constEvalFuncWrapper_46[0]
    const_50 = main_const_eval_50
    util_create_list_148 = [input_58]
    utils_constEvalFuncWrapper_47 = utils.constEvalFuncWrapper(
        const_50, util_create_list_148, CACHED_main_const_eval_50
    )
    CACHED_main_const_eval_50 = utils_constEvalFuncWrapper_47
    utils_constEvalFuncWrapper_47_0 = utils_constEvalFuncWrapper_47[0]
    const_51 = main_const_eval_51
    util_create_list_149 = [input_63]
    utils_constEvalFuncWrapper_48 = utils.constEvalFuncWrapper(
        const_51, util_create_list_149, CACHED_main_const_eval_51
    )
    CACHED_main_const_eval_51 = utils_constEvalFuncWrapper_48
    utils_constEvalFuncWrapper_48_0 = utils_constEvalFuncWrapper_48[0]
    const_52 = main_const_eval_52
    util_create_list_150 = [input_70]
    utils_constEvalFuncWrapper_49 = utils.constEvalFuncWrapper(
        const_52, util_create_list_150, CACHED_main_const_eval_52
    )
    CACHED_main_const_eval_52 = utils_constEvalFuncWrapper_49
    utils_constEvalFuncWrapper_49_0 = utils_constEvalFuncWrapper_49[0]
    const_53 = main_const_eval_53
    util_create_list_151 = [input_112]
    utils_constEvalFuncWrapper_50 = utils.constEvalFuncWrapper(
        const_53, util_create_list_151, CACHED_main_const_eval_53
    )
    CACHED_main_const_eval_53 = utils_constEvalFuncWrapper_50
    utils_constEvalFuncWrapper_50_0 = utils_constEvalFuncWrapper_50[0]
    const_54 = main_const_eval_54
    util_create_list_152 = [input_106]
    utils_constEvalFuncWrapper_51 = utils.constEvalFuncWrapper(
        const_54, util_create_list_152, CACHED_main_const_eval_54
    )
    CACHED_main_const_eval_54 = utils_constEvalFuncWrapper_51
    utils_constEvalFuncWrapper_51_0 = utils_constEvalFuncWrapper_51[0]
    const_55 = main_const_eval_55
    util_create_list_153 = [input_15]
    utils_constEvalFuncWrapper_52 = utils.constEvalFuncWrapper(
        const_55, util_create_list_153, CACHED_main_const_eval_55
    )
    CACHED_main_const_eval_55 = utils_constEvalFuncWrapper_52
    utils_constEvalFuncWrapper_52_0 = utils_constEvalFuncWrapper_52[0]
    const_56 = main_const_eval_56
    util_create_list_154 = [input_111]
    utils_constEvalFuncWrapper_53 = utils.constEvalFuncWrapper(
        const_56, util_create_list_154, CACHED_main_const_eval_56
    )
    CACHED_main_const_eval_56 = utils_constEvalFuncWrapper_53
    utils_constEvalFuncWrapper_53_0 = utils_constEvalFuncWrapper_53[0]
    const_57 = main_const_eval_57
    util_create_list_155 = [input_113]
    utils_constEvalFuncWrapper_54 = utils.constEvalFuncWrapper(
        const_57, util_create_list_155, CACHED_main_const_eval_57
    )
    CACHED_main_const_eval_57 = utils_constEvalFuncWrapper_54
    utils_constEvalFuncWrapper_54_0 = utils_constEvalFuncWrapper_54[0]
    const_58 = main_const_eval_58
    util_create_list_156 = [input_109]
    utils_constEvalFuncWrapper_55 = utils.constEvalFuncWrapper(
        const_58, util_create_list_156, CACHED_main_const_eval_58
    )
    CACHED_main_const_eval_58 = utils_constEvalFuncWrapper_55
    utils_constEvalFuncWrapper_55_0 = utils_constEvalFuncWrapper_55[0]
    const_59 = main_const_eval_59
    util_create_list_157 = [input_45]
    utils_constEvalFuncWrapper_56 = utils.constEvalFuncWrapper(
        const_59, util_create_list_157, CACHED_main_const_eval_59
    )
    CACHED_main_const_eval_59 = utils_constEvalFuncWrapper_56
    utils_constEvalFuncWrapper_56_0 = utils_constEvalFuncWrapper_56[0]
    const_60 = main_const_eval_60
    util_create_list_158 = [input_60]
    utils_constEvalFuncWrapper_57 = utils.constEvalFuncWrapper(
        const_60, util_create_list_158, CACHED_main_const_eval_60
    )
    CACHED_main_const_eval_60 = utils_constEvalFuncWrapper_57
    utils_constEvalFuncWrapper_57_0 = utils_constEvalFuncWrapper_57[0]
    const_61 = main_const_eval_61
    util_create_list_159 = [input_41]
    utils_constEvalFuncWrapper_58 = utils.constEvalFuncWrapper(
        const_61, util_create_list_159, CACHED_main_const_eval_61
    )
    CACHED_main_const_eval_61 = utils_constEvalFuncWrapper_58
    utils_constEvalFuncWrapper_58_0 = utils_constEvalFuncWrapper_58[0]
    const_62 = main_const_eval_62
    util_create_list_160 = [input_27]
    utils_constEvalFuncWrapper_59 = utils.constEvalFuncWrapper(
        const_62, util_create_list_160, CACHED_main_const_eval_62
    )
    CACHED_main_const_eval_62 = utils_constEvalFuncWrapper_59
    utils_constEvalFuncWrapper_59_0 = utils_constEvalFuncWrapper_59[0]
    const_63 = main_const_eval_63
    util_create_list_161 = [input_2]
    utils_constEvalFuncWrapper_60 = utils.constEvalFuncWrapper(
        const_63, util_create_list_161, CACHED_main_const_eval_63
    )
    CACHED_main_const_eval_63 = utils_constEvalFuncWrapper_60
    utils_constEvalFuncWrapper_60_0 = utils_constEvalFuncWrapper_60[0]
    const_64 = main_const_eval_64
    util_create_list_162 = [input_95]
    utils_constEvalFuncWrapper_61 = utils.constEvalFuncWrapper(
        const_64, util_create_list_162, CACHED_main_const_eval_64
    )
    CACHED_main_const_eval_64 = utils_constEvalFuncWrapper_61
    utils_constEvalFuncWrapper_61_0 = utils_constEvalFuncWrapper_61[0]
    const_65 = main_const_eval_65
    util_create_list_163 = [input_124]
    utils_constEvalFuncWrapper_62 = utils.constEvalFuncWrapper(
        const_65, util_create_list_163, CACHED_main_const_eval_65
    )
    CACHED_main_const_eval_65 = utils_constEvalFuncWrapper_62
    utils_constEvalFuncWrapper_62_0 = utils_constEvalFuncWrapper_62[0]
    const_66 = main_const_eval_66
    util_create_list_164 = [input_0]
    utils_constEvalFuncWrapper_63 = utils.constEvalFuncWrapper(
        const_66, util_create_list_164, CACHED_main_const_eval_66
    )
    CACHED_main_const_eval_66 = utils_constEvalFuncWrapper_63
    utils_constEvalFuncWrapper_63_0 = utils_constEvalFuncWrapper_63[0]
    const_67 = main_const_eval_67
    util_create_list_165 = [input_66]
    utils_constEvalFuncWrapper_64 = utils.constEvalFuncWrapper(
        const_67, util_create_list_165, CACHED_main_const_eval_67
    )
    CACHED_main_const_eval_67 = utils_constEvalFuncWrapper_64
    utils_constEvalFuncWrapper_64_0 = utils_constEvalFuncWrapper_64[0]
    const_68 = main_const_eval_68
    util_create_list_166 = [input_53]
    utils_constEvalFuncWrapper_65 = utils.constEvalFuncWrapper(
        const_68, util_create_list_166, CACHED_main_const_eval_68
    )
    CACHED_main_const_eval_68 = utils_constEvalFuncWrapper_65
    utils_constEvalFuncWrapper_65_0 = utils_constEvalFuncWrapper_65[0]
    const_69 = main_const_eval_69
    util_create_list_167 = [input_3]
    utils_constEvalFuncWrapper_66 = utils.constEvalFuncWrapper(
        const_69, util_create_list_167, CACHED_main_const_eval_69
    )
    CACHED_main_const_eval_69 = utils_constEvalFuncWrapper_66
    utils_constEvalFuncWrapper_66_0 = utils_constEvalFuncWrapper_66[0]
    const_70 = main_const_eval_70
    util_create_list_168 = [input_123]
    utils_constEvalFuncWrapper_67 = utils.constEvalFuncWrapper(
        const_70, util_create_list_168, CACHED_main_const_eval_70
    )
    CACHED_main_const_eval_70 = utils_constEvalFuncWrapper_67
    utils_constEvalFuncWrapper_67_0 = utils_constEvalFuncWrapper_67[0]
    const_71 = main_const_eval_71
    util_create_list_169 = [input_16]
    utils_constEvalFuncWrapper_68 = utils.constEvalFuncWrapper(
        const_71, util_create_list_169, CACHED_main_const_eval_71
    )
    CACHED_main_const_eval_71 = utils_constEvalFuncWrapper_68
    utils_constEvalFuncWrapper_68_0 = utils_constEvalFuncWrapper_68[0]
    const_72 = main_const_eval_72
    util_create_list_170 = [input_33]
    utils_constEvalFuncWrapper_69 = utils.constEvalFuncWrapper(
        const_72, util_create_list_170, CACHED_main_const_eval_72
    )
    CACHED_main_const_eval_72 = utils_constEvalFuncWrapper_69
    utils_constEvalFuncWrapper_69_0 = utils_constEvalFuncWrapper_69[0]
    const_73 = main_const_eval_73
    util_create_list_171 = [input_102]
    utils_constEvalFuncWrapper_70 = utils.constEvalFuncWrapper(
        const_73, util_create_list_171, CACHED_main_const_eval_73
    )
    CACHED_main_const_eval_73 = utils_constEvalFuncWrapper_70
    utils_constEvalFuncWrapper_70_0 = utils_constEvalFuncWrapper_70[0]
    const_74 = main_const_eval_74
    util_create_list_172 = [input_68]
    utils_constEvalFuncWrapper_71 = utils.constEvalFuncWrapper(
        const_74, util_create_list_172, CACHED_main_const_eval_74
    )
    CACHED_main_const_eval_74 = utils_constEvalFuncWrapper_71
    utils_constEvalFuncWrapper_71_0 = utils_constEvalFuncWrapper_71[0]
    const_75 = main_const_eval_75
    util_create_list_173 = [input_75]
    utils_constEvalFuncWrapper_72 = utils.constEvalFuncWrapper(
        const_75, util_create_list_173, CACHED_main_const_eval_75
    )
    CACHED_main_const_eval_75 = utils_constEvalFuncWrapper_72
    utils_constEvalFuncWrapper_72_0 = utils_constEvalFuncWrapper_72[0]
    const_76 = main_const_eval_76
    util_create_list_174 = [input_104]
    utils_constEvalFuncWrapper_73 = utils.constEvalFuncWrapper(
        const_76, util_create_list_174, CACHED_main_const_eval_76
    )
    CACHED_main_const_eval_76 = utils_constEvalFuncWrapper_73
    utils_constEvalFuncWrapper_73_0 = utils_constEvalFuncWrapper_73[0]
    const_77 = main_const_eval_77
    util_create_list_175 = [input_119]
    utils_constEvalFuncWrapper_74 = utils.constEvalFuncWrapper(
        const_77, util_create_list_175, CACHED_main_const_eval_77
    )
    CACHED_main_const_eval_77 = utils_constEvalFuncWrapper_74
    utils_constEvalFuncWrapper_74_0 = utils_constEvalFuncWrapper_74[0]
    const_78 = main_const_eval_78
    util_create_list_176 = [input_114]
    utils_constEvalFuncWrapper_75 = utils.constEvalFuncWrapper(
        const_78, util_create_list_176, CACHED_main_const_eval_78
    )
    CACHED_main_const_eval_78 = utils_constEvalFuncWrapper_75
    utils_constEvalFuncWrapper_75_0 = utils_constEvalFuncWrapper_75[0]
    const_79 = main_const_eval_79
    util_create_list_177 = [input_57]
    utils_constEvalFuncWrapper_76 = utils.constEvalFuncWrapper(
        const_79, util_create_list_177, CACHED_main_const_eval_79
    )
    CACHED_main_const_eval_79 = utils_constEvalFuncWrapper_76
    utils_constEvalFuncWrapper_76_0 = utils_constEvalFuncWrapper_76[0]
    const_80 = main_const_eval_80
    util_create_list_178 = [input_31]
    utils_constEvalFuncWrapper_77 = utils.constEvalFuncWrapper(
        const_80, util_create_list_178, CACHED_main_const_eval_80
    )
    CACHED_main_const_eval_80 = utils_constEvalFuncWrapper_77
    utils_constEvalFuncWrapper_77_0 = utils_constEvalFuncWrapper_77[0]
    const_81 = main_const_eval_81
    util_create_list_179 = [input_86]
    utils_constEvalFuncWrapper_78 = utils.constEvalFuncWrapper(
        const_81, util_create_list_179, CACHED_main_const_eval_81
    )
    CACHED_main_const_eval_81 = utils_constEvalFuncWrapper_78
    utils_constEvalFuncWrapper_78_0 = utils_constEvalFuncWrapper_78[0]
    const_82 = main_const_eval_82
    util_create_list_180 = [input_89]
    utils_constEvalFuncWrapper_79 = utils.constEvalFuncWrapper(
        const_82, util_create_list_180, CACHED_main_const_eval_82
    )
    CACHED_main_const_eval_82 = utils_constEvalFuncWrapper_79
    utils_constEvalFuncWrapper_79_0 = utils_constEvalFuncWrapper_79[0]
    const_83 = main_const_eval_83
    util_create_list_181 = [input_84]
    utils_constEvalFuncWrapper_80 = utils.constEvalFuncWrapper(
        const_83, util_create_list_181, CACHED_main_const_eval_83
    )
    CACHED_main_const_eval_83 = utils_constEvalFuncWrapper_80
    utils_constEvalFuncWrapper_80_0 = utils_constEvalFuncWrapper_80[0]
    const_84 = main_const_eval_84
    util_create_list_182 = [input_62]
    utils_constEvalFuncWrapper_81 = utils.constEvalFuncWrapper(
        const_84, util_create_list_182, CACHED_main_const_eval_84
    )
    CACHED_main_const_eval_84 = utils_constEvalFuncWrapper_81
    utils_constEvalFuncWrapper_81_0 = utils_constEvalFuncWrapper_81[0]
    const_85 = main_const_eval_85
    util_create_list_183 = [input_83]
    utils_constEvalFuncWrapper_82 = utils.constEvalFuncWrapper(
        const_85, util_create_list_183, CACHED_main_const_eval_85
    )
    CACHED_main_const_eval_85 = utils_constEvalFuncWrapper_82
    utils_constEvalFuncWrapper_82_0 = utils_constEvalFuncWrapper_82[0]
    const_86 = main_const_eval_86
    util_create_list_184 = [input_90]
    utils_constEvalFuncWrapper_83 = utils.constEvalFuncWrapper(
        const_86, util_create_list_184, CACHED_main_const_eval_86
    )
    CACHED_main_const_eval_86 = utils_constEvalFuncWrapper_83
    utils_constEvalFuncWrapper_83_0 = utils_constEvalFuncWrapper_83[0]
    const_87 = main_const_eval_87
    util_create_list_185 = [input_43]
    utils_constEvalFuncWrapper_84 = utils.constEvalFuncWrapper(
        const_87, util_create_list_185, CACHED_main_const_eval_87
    )
    CACHED_main_const_eval_87 = utils_constEvalFuncWrapper_84
    utils_constEvalFuncWrapper_84_0 = utils_constEvalFuncWrapper_84[0]
    const_88 = main_const_eval_88
    util_create_list_186 = [input_65]
    utils_constEvalFuncWrapper_85 = utils.constEvalFuncWrapper(
        const_88, util_create_list_186, CACHED_main_const_eval_88
    )
    CACHED_main_const_eval_88 = utils_constEvalFuncWrapper_85
    utils_constEvalFuncWrapper_85_0 = utils_constEvalFuncWrapper_85[0]
    const_89 = main_const_eval_89
    util_create_list_187 = [input_52]
    utils_constEvalFuncWrapper_86 = utils.constEvalFuncWrapper(
        const_89, util_create_list_187, CACHED_main_const_eval_89
    )
    CACHED_main_const_eval_89 = utils_constEvalFuncWrapper_86
    utils_constEvalFuncWrapper_86_0 = utils_constEvalFuncWrapper_86[0]
    const_90 = main_const_eval_90
    util_create_list_188 = [input_121]
    utils_constEvalFuncWrapper_87 = utils.constEvalFuncWrapper(
        const_90, util_create_list_188, CACHED_main_const_eval_90
    )
    CACHED_main_const_eval_90 = utils_constEvalFuncWrapper_87
    utils_constEvalFuncWrapper_87_0 = utils_constEvalFuncWrapper_87[0]
    const_91 = main_const_eval_91
    util_create_list_189 = [input_40]
    utils_constEvalFuncWrapper_88 = utils.constEvalFuncWrapper(
        const_91, util_create_list_189, CACHED_main_const_eval_91
    )
    CACHED_main_const_eval_91 = utils_constEvalFuncWrapper_88
    utils_constEvalFuncWrapper_88_0 = utils_constEvalFuncWrapper_88[0]
    const_92 = main_const_eval_92
    util_create_list_190 = [input_73]
    utils_constEvalFuncWrapper_89 = utils.constEvalFuncWrapper(
        const_92, util_create_list_190, CACHED_main_const_eval_92
    )
    CACHED_main_const_eval_92 = utils_constEvalFuncWrapper_89
    utils_constEvalFuncWrapper_89_0 = utils_constEvalFuncWrapper_89[0]
    const_93 = main_const_eval_93
    util_create_list_191 = [input_17]
    utils_constEvalFuncWrapper_90 = utils.constEvalFuncWrapper(
        const_93, util_create_list_191, CACHED_main_const_eval_93
    )
    CACHED_main_const_eval_93 = utils_constEvalFuncWrapper_90
    utils_constEvalFuncWrapper_90_0 = utils_constEvalFuncWrapper_90[0]
    const_94 = main_const_eval_94
    utils_constEvalFuncWrapperZeroArg_3 = utils.constEvalFuncWrapperZeroArg(const_94, CACHED_main_const_eval_94)
    CACHED_main_const_eval_94 = utils_constEvalFuncWrapperZeroArg_3
    utils_constEvalFuncWrapperZeroArg_3_0 = utils_constEvalFuncWrapperZeroArg_3[0]
    const_95 = main_const_eval_95
    utils_constEvalFuncWrapperZeroArg_4 = utils.constEvalFuncWrapperZeroArg(const_95, CACHED_main_const_eval_95)
    CACHED_main_const_eval_95 = utils_constEvalFuncWrapperZeroArg_4
    utils_constEvalFuncWrapperZeroArg_4_0 = utils_constEvalFuncWrapperZeroArg_4[0]
    const_96 = main_const_eval_96
    util_create_list_192 = [input_118]
    utils_constEvalFuncWrapper_91 = utils.constEvalFuncWrapper(
        const_96, util_create_list_192, CACHED_main_const_eval_96
    )
    CACHED_main_const_eval_96 = utils_constEvalFuncWrapper_91
    utils_constEvalFuncWrapper_91_0 = utils_constEvalFuncWrapper_91[0]
    const_97 = main_const_eval_97
    util_create_list_193 = [input_19]
    utils_constEvalFuncWrapper_92 = utils.constEvalFuncWrapper(
        const_97, util_create_list_193, CACHED_main_const_eval_97
    )
    CACHED_main_const_eval_97 = utils_constEvalFuncWrapper_92
    utils_constEvalFuncWrapper_92_0 = utils_constEvalFuncWrapper_92[0]
    const_98 = main_const_eval_98
    util_create_list_194 = [input_122]
    utils_constEvalFuncWrapper_93 = utils.constEvalFuncWrapper(
        const_98, util_create_list_194, CACHED_main_const_eval_98
    )
    CACHED_main_const_eval_98 = utils_constEvalFuncWrapper_93
    utils_constEvalFuncWrapper_93_0 = utils_constEvalFuncWrapper_93[0]
    const_99 = main_const_eval_99
    util_create_list_195 = [input_93]
    utils_constEvalFuncWrapper_94 = utils.constEvalFuncWrapper(
        const_99, util_create_list_195, CACHED_main_const_eval_99
    )
    CACHED_main_const_eval_99 = utils_constEvalFuncWrapper_94
    utils_constEvalFuncWrapper_94_0 = utils_constEvalFuncWrapper_94[0]
    const_100 = main_const_eval_100
    util_create_list_196 = [input_94]
    utils_constEvalFuncWrapper_95 = utils.constEvalFuncWrapper(
        const_100, util_create_list_196, CACHED_main_const_eval_100
    )
    CACHED_main_const_eval_100 = utils_constEvalFuncWrapper_95
    utils_constEvalFuncWrapper_95_0 = utils_constEvalFuncWrapper_95[0]
    utils_DeviceGetter_get_device_101 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_96 = ttnn.to_layout(
        input_49,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_4 = ttnn.permute(
        ttnn_to_layout_96,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_96, False)
    ttnn_reshape_100 = ttnn.reshape(
        ttnn_permute_4,
        [10240, 20],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_4, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_100,
        utils_constEvalFuncWrapperZeroArg_0_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_100, False)
    ttnn_reshape_101 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 512, 20, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_permute_5 = ttnn.permute(
        ttnn_reshape_101,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_101, False)
    ttnn_reshape_102 = ttnn.reshape(
        ttnn_permute_5,
        [20480, 20],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_5, False)
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_reshape_102,
        utils_constEvalFuncWrapperZeroArg_0_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_102, False)
    ttnn_reshape_103 = ttnn.reshape(
        ttnn_matmul_1,
        [1, 512, 40, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn_permute_6 = ttnn.permute(
        ttnn_reshape_103,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_103, False)
    ttnn_to_layout_97 = ttnn.to_layout(
        input_48,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_48, False)
    ttnn_permute_7 = ttnn.permute(
        ttnn_to_layout_97,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_97, False)
    ttnn_reshape_104 = ttnn.reshape(
        ttnn_permute_6,
        [1, 1, 1600, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_6, False)
    ttnn_reshape_105 = ttnn.reshape(
        ttnn_permute_7,
        [1, 1, 1600, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_7, False)
    util_create_list_197 = [ttnn_reshape_104, ttnn_reshape_105]
    ttnn_concat_0 = ttnn.concat(
        util_create_list_197,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_105, False)
    ttnn.deallocate(ttnn_reshape_104, False)
    ttnn_to_layout_98 = ttnn.to_layout(
        ttnn_concat_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_98,
        weight_tensor=input_47,
        device=utils_DeviceGetter_get_device_101,
        in_channels=768,
        out_channels=256,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_98, False)
    ttnn_reshape_106 = ttnn.reshape(
        ttnn_conv2d_0,
        [1, 40, 40, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_8 = ttnn.permute(
        ttnn_reshape_106,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_106, False)
    ttnn_batch_norm_0 = ttnn.batch_norm(
        ttnn_permute_8,
        running_mean=utils_constEvalFuncWrapper_41_0,
        running_var=utils_constEvalFuncWrapper_84_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_31_0,
        bias=utils_constEvalFuncWrapper_56_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_8, False)
    ttnn_silu_0 = ttnn.silu(
        ttnn_batch_norm_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_0, False)
    ttnn_permute_9 = ttnn.permute(
        ttnn_silu_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_0, False)
    ttnn_reshape_107 = ttnn.reshape(
        ttnn_permute_9,
        [1, 1, 1600, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_9, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_reshape_107,
        [0, 0, 0, 128],
        [1, 1, 1600, 256],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_99 = ttnn.to_layout(
        ttnn_slice_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_conv2d_1 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_99,
        weight_tensor=input_42,
        device=utils_DeviceGetter_get_device_101,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_99, False)
    ttnn_reshape_108 = ttnn.reshape(
        ttnn_conv2d_1,
        [1, 40, 40, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_1, False)
    ttnn_permute_10 = ttnn.permute(
        ttnn_reshape_108,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_108, False)
    ttnn_batch_norm_1 = ttnn.batch_norm(
        ttnn_permute_10,
        running_mean=utils_constEvalFuncWrapper_19_0,
        running_var=utils_constEvalFuncWrapper_42_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_58_0,
        bias=utils_constEvalFuncWrapper_88_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_10, False)
    ttnn_silu_1 = ttnn.silu(
        ttnn_batch_norm_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_1, False)
    ttnn_permute_11 = ttnn.permute(
        ttnn_silu_1,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_1, False)
    ttnn_reshape_109 = ttnn.reshape(
        ttnn_permute_11,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_11, False)
    ttnn_to_layout_100 = ttnn.to_layout(
        ttnn_reshape_109,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_109, False)
    ttnn_conv2d_2 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_100,
        weight_tensor=input_37,
        device=utils_DeviceGetter_get_device_101,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_100, False)
    ttnn_reshape_110 = ttnn.reshape(
        ttnn_conv2d_2,
        [1, 40, 40, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_2, False)
    ttnn_permute_12 = ttnn.permute(
        ttnn_reshape_110,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_110, False)
    ttnn_batch_norm_2 = ttnn.batch_norm(
        ttnn_permute_12,
        running_mean=utils_constEvalFuncWrapper_1_0,
        running_var=utils_constEvalFuncWrapper_69_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_43_0,
        bias=utils_constEvalFuncWrapper_46_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_12, False)
    ttnn_silu_2 = ttnn.silu(
        ttnn_batch_norm_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_2, False)
    ttnn_reshape_111 = ttnn.reshape(
        ttnn_silu_2,
        [1, 4, 32, 40, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_101 = ttnn.to_layout(
        input_8,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_8, False)
    ttnn_reshape_112 = ttnn.reshape(
        ttnn_to_layout_101,
        [3, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_101, False)
    ttnn_matmul_2 = ttnn.matmul(
        ttnn_reshape_112,
        input_32,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_0 = ttnn.add(
        ttnn_matmul_2,
        utils_constEvalFuncWrapper_77_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn_reshape_113 = ttnn.reshape(
        ttnn_add_0,
        [1, 3, 4, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_permute_13 = ttnn.permute(
        ttnn_reshape_111,
        [0, 1, 3, 4, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_111, False)
    ttnn_permute_14 = ttnn.permute(
        ttnn_reshape_113,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_113, False)
    ttnn_reshape_114 = ttnn.reshape(
        ttnn_permute_13,
        [1, 4, 1600, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_13, False)
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_reshape_114,
        ttnn_permute_14,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_114, False)
    ttnn.deallocate(ttnn_permute_14, False)
    ttnn_reshape_115 = ttnn.reshape(
        ttnn_matmul_3,
        [1, 4, 40, 40, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn_slice_1 = ttnn.slice(
        ttnn_reshape_107,
        [0, 0, 0, 0],
        [1, 1, 1600, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_107, False)
    ttnn_permute_15 = ttnn.permute(
        ttnn_silu_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_116 = ttnn.reshape(
        ttnn_permute_15,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_15, False)
    ttnn_to_layout_102 = ttnn.to_layout(
        ttnn_reshape_116,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_116, False)
    ttnn_conv2d_3 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_102,
        weight_tensor=input_54,
        device=utils_DeviceGetter_get_device_101,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_102, False)
    ttnn_reshape_117 = ttnn.reshape(
        ttnn_conv2d_3,
        [1, 40, 40, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_3, False)
    ttnn_permute_16 = ttnn.permute(
        ttnn_reshape_117,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_117, False)
    ttnn_batch_norm_3 = ttnn.batch_norm(
        ttnn_permute_16,
        running_mean=utils_constEvalFuncWrapper_40_0,
        running_var=utils_constEvalFuncWrapper_4_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_65_0,
        bias=utils_constEvalFuncWrapper_86_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_16, False)
    ttnn_max_0 = ttnn.max(
        ttnn_reshape_115,
        [4],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_115, False)
    ttnn_divide_0 = ttnn.divide(
        ttnn_max_0,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_max_0, False)
    ttnn_add_1 = ttnn.add(
        ttnn_divide_0,
        utils_constEvalFuncWrapper_18_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_divide_0, False)
    ttnn_sigmoid_0 = ttnn.sigmoid(
        ttnn_add_1,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_reshape_118 = ttnn.reshape(
        ttnn_sigmoid_0,
        [1, 4, 1, 40, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sigmoid_0, False)
    ttnn_repeat_4 = ttnn.repeat(ttnn_reshape_118, ttnn.Shape([1, 1, 32, 1, 1]))
    ttnn.deallocate(ttnn_reshape_118, False)
    ttnn_reshape_119 = ttnn.reshape(
        ttnn_repeat_4,
        [1, 128, 40, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_4, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_batch_norm_3,
        ttnn_reshape_119,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_119, False)
    ttnn.deallocate(ttnn_batch_norm_3, False)
    ttnn_permute_17 = ttnn.permute(
        ttnn_multiply_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_reshape_120 = ttnn.reshape(
        ttnn_permute_17,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_17, False)
    ttnn_permute_18 = ttnn.permute(
        ttnn_silu_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_2, False)
    ttnn_reshape_121 = ttnn.reshape(
        ttnn_permute_18,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_18, False)
    util_create_list_198 = [
        ttnn_slice_1,
        ttnn_slice_0,
        ttnn_reshape_121,
        ttnn_reshape_120,
    ]
    ttnn_concat_1 = ttnn.concat(
        util_create_list_198,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_121, False)
    ttnn.deallocate(ttnn_reshape_120, False)
    ttnn.deallocate(ttnn_slice_1, False)
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_to_layout_103 = ttnn.to_layout(
        ttnn_concat_1,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_1, False)
    ttnn_conv2d_4 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_103,
        weight_tensor=input_29,
        device=utils_DeviceGetter_get_device_101,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_103, False)
    ttnn_reshape_122 = ttnn.reshape(
        ttnn_conv2d_4,
        [1, 40, 40, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_4, False)
    ttnn_permute_19 = ttnn.permute(
        ttnn_reshape_122,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_122, False)
    ttnn_batch_norm_4 = ttnn.batch_norm(
        ttnn_permute_19,
        running_mean=utils_constEvalFuncWrapper_37_0,
        running_var=utils_constEvalFuncWrapper_32_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_7_0,
        bias=utils_constEvalFuncWrapper_59_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_19, False)
    ttnn_silu_3 = ttnn.silu(
        ttnn_batch_norm_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_4, False)
    ttnn_permute_20 = ttnn.permute(
        ttnn_silu_3,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_123 = ttnn.reshape(
        ttnn_permute_20,
        [10240, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_20, False)
    ttnn_matmul_4 = ttnn.matmul(
        ttnn_reshape_123,
        utils_constEvalFuncWrapperZeroArg_4_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_123, False)
    ttnn_reshape_124 = ttnn.reshape(
        ttnn_matmul_4,
        [1, 256, 40, 80],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_4, False)
    ttnn_permute_21 = ttnn.permute(
        ttnn_reshape_124,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_124, False)
    ttnn_reshape_125 = ttnn.reshape(
        ttnn_permute_21,
        [20480, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_21, False)
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_reshape_125,
        utils_constEvalFuncWrapperZeroArg_4_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_125, False)
    ttnn_reshape_126 = ttnn.reshape(
        ttnn_matmul_5,
        [1, 256, 80, 80],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn_permute_22 = ttnn.permute(
        ttnn_reshape_126,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_126, False)
    ttnn_to_layout_104 = ttnn.to_layout(
        input_24,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_24, False)
    ttnn_permute_23 = ttnn.permute(
        ttnn_to_layout_104,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_104, False)
    ttnn_reshape_127 = ttnn.reshape(
        ttnn_permute_22,
        [1, 1, 6400, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_22, False)
    ttnn_reshape_128 = ttnn.reshape(
        ttnn_permute_23,
        [1, 1, 6400, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_23, False)
    util_create_list_199 = [ttnn_reshape_127, ttnn_reshape_128]
    ttnn_concat_2 = ttnn.concat(
        util_create_list_199,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_128, False)
    ttnn.deallocate(ttnn_reshape_127, False)
    ttnn_to_layout_105 = ttnn.to_layout(
        ttnn_concat_2,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_2, False)
    ttnn_conv2d_5 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_105,
        weight_tensor=input_23,
        device=utils_DeviceGetter_get_device_101,
        in_channels=384,
        out_channels=128,
        batch_size=1,
        input_height=80,
        input_width=80,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_105, False)
    ttnn_reshape_129 = ttnn.reshape(
        ttnn_conv2d_5,
        [1, 80, 80, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_5, False)
    ttnn_permute_24 = ttnn.permute(
        ttnn_reshape_129,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_129, False)
    ttnn_batch_norm_5 = ttnn.batch_norm(
        ttnn_permute_24,
        running_mean=utils_constEvalFuncWrapper_36_0,
        running_var=utils_constEvalFuncWrapper_92_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_27_0,
        bias=utils_constEvalFuncWrapper_35_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_24, False)
    ttnn_silu_4 = ttnn.silu(
        ttnn_batch_norm_5,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_5, False)
    ttnn_permute_25 = ttnn.permute(
        ttnn_silu_4,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_4, False)
    ttnn_reshape_130 = ttnn.reshape(
        ttnn_permute_25,
        [1, 1, 6400, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_25, False)
    ttnn_slice_2 = ttnn.slice(
        ttnn_reshape_130,
        [0, 0, 0, 64],
        [1, 1, 6400, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_106 = ttnn.to_layout(
        ttnn_slice_2,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_conv2d_6 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_106,
        weight_tensor=input_18,
        device=utils_DeviceGetter_get_device_101,
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=80,
        input_width=80,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_106, False)
    ttnn_reshape_131 = ttnn.reshape(
        ttnn_conv2d_6,
        [1, 80, 80, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_6, False)
    ttnn_permute_26 = ttnn.permute(
        ttnn_reshape_131,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_131, False)
    ttnn_batch_norm_6 = ttnn.batch_norm(
        ttnn_permute_26,
        running_mean=utils_constEvalFuncWrapper_52_0,
        running_var=utils_constEvalFuncWrapper_30_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_90_0,
        bias=utils_constEvalFuncWrapper_68_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_26, False)
    ttnn_silu_5 = ttnn.silu(
        ttnn_batch_norm_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_6, False)
    ttnn_permute_27 = ttnn.permute(
        ttnn_silu_5,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_5, False)
    ttnn_reshape_132 = ttnn.reshape(
        ttnn_permute_27,
        [1, 1, 6400, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_27, False)
    ttnn_to_layout_107 = ttnn.to_layout(
        ttnn_reshape_132,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_132, False)
    ttnn_conv2d_7 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_107,
        weight_tensor=input_13,
        device=utils_DeviceGetter_get_device_101,
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=80,
        input_width=80,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_107, False)
    ttnn_reshape_133 = ttnn.reshape(
        ttnn_conv2d_7,
        [1, 80, 80, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_7, False)
    ttnn_permute_28 = ttnn.permute(
        ttnn_reshape_133,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_133, False)
    ttnn_batch_norm_7 = ttnn.batch_norm(
        ttnn_permute_28,
        running_mean=utils_constEvalFuncWrapper_20_0,
        running_var=utils_constEvalFuncWrapper_24_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_0_0,
        bias=utils_constEvalFuncWrapper_15_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_28, False)
    ttnn_silu_6 = ttnn.silu(
        ttnn_batch_norm_7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_7, False)
    ttnn_reshape_134 = ttnn.reshape(
        ttnn_silu_6,
        [1, 2, 32, 80, 80],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_reshape_112,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_2 = ttnn.add(
        ttnn_matmul_6,
        utils_constEvalFuncWrapper_10_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn_reshape_135 = ttnn.reshape(
        ttnn_add_2,
        [1, 3, 2, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_2, False)
    ttnn_permute_29 = ttnn.permute(
        ttnn_reshape_134,
        [0, 1, 3, 4, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_134, False)
    ttnn_permute_30 = ttnn.permute(
        ttnn_reshape_135,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_135, False)
    ttnn_reshape_136 = ttnn.reshape(
        ttnn_permute_29,
        [1, 2, 6400, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_29, False)
    ttnn_matmul_7 = ttnn.matmul(
        ttnn_reshape_136,
        ttnn_permute_30,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_136, False)
    ttnn.deallocate(ttnn_permute_30, False)
    ttnn_reshape_137 = ttnn.reshape(
        ttnn_matmul_7,
        [1, 2, 80, 80, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn_slice_3 = ttnn.slice(
        ttnn_reshape_130,
        [0, 0, 0, 0],
        [1, 1, 6400, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_130, False)
    ttnn_permute_31 = ttnn.permute(
        ttnn_silu_6,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_138 = ttnn.reshape(
        ttnn_permute_31,
        [1, 1, 6400, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_31, False)
    ttnn_to_layout_108 = ttnn.to_layout(
        ttnn_reshape_138,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_138, False)
    ttnn_conv2d_8 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_108,
        weight_tensor=input_59,
        device=utils_DeviceGetter_get_device_101,
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=80,
        input_width=80,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_108, False)
    ttnn_reshape_139 = ttnn.reshape(
        ttnn_conv2d_8,
        [1, 80, 80, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_8, False)
    ttnn_permute_32 = ttnn.permute(
        ttnn_reshape_139,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_139, False)
    ttnn_batch_norm_8 = ttnn.batch_norm(
        ttnn_permute_32,
        running_mean=utils_constEvalFuncWrapper_34_0,
        running_var=utils_constEvalFuncWrapper_38_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_47_0,
        bias=utils_constEvalFuncWrapper_76_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_32, False)
    ttnn_max_1 = ttnn.max(
        ttnn_reshape_137,
        [4],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_137, False)
    ttnn_divide_1 = ttnn.divide(
        ttnn_max_1,
        utils_constEvalFuncWrapperZeroArg_3_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_max_1, False)
    ttnn_add_3 = ttnn.add(
        ttnn_divide_1,
        utils_constEvalFuncWrapper_23_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_divide_1, False)
    ttnn_sigmoid_1 = ttnn.sigmoid(
        ttnn_add_3,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_reshape_140 = ttnn.reshape(
        ttnn_sigmoid_1,
        [1, 2, 1, 80, 80],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sigmoid_1, False)
    ttnn_repeat_5 = ttnn.repeat(ttnn_reshape_140, ttnn.Shape([1, 1, 32, 1, 1]))
    ttnn.deallocate(ttnn_reshape_140, False)
    ttnn_reshape_141 = ttnn.reshape(
        ttnn_repeat_5,
        [1, 64, 80, 80],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_5, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_batch_norm_8,
        ttnn_reshape_141,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_141, False)
    ttnn.deallocate(ttnn_batch_norm_8, False)
    ttnn_permute_33 = ttnn.permute(
        ttnn_multiply_1,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_reshape_142 = ttnn.reshape(
        ttnn_permute_33,
        [1, 1, 6400, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_33, False)
    ttnn_permute_34 = ttnn.permute(
        ttnn_silu_6,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_6, False)
    ttnn_reshape_143 = ttnn.reshape(
        ttnn_permute_34,
        [1, 1, 6400, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_34, False)
    util_create_list_200 = [
        ttnn_slice_3,
        ttnn_slice_2,
        ttnn_reshape_143,
        ttnn_reshape_142,
    ]
    ttnn_concat_3 = ttnn.concat(
        util_create_list_200,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_143, False)
    ttnn.deallocate(ttnn_reshape_142, False)
    ttnn.deallocate(ttnn_slice_3, False)
    ttnn.deallocate(ttnn_slice_2, False)
    ttnn_to_layout_109 = ttnn.to_layout(
        ttnn_concat_3,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_3, False)
    ttnn_conv2d_9 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_109,
        weight_tensor=input_4,
        device=utils_DeviceGetter_get_device_101,
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=80,
        input_width=80,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_109, False)
    ttnn_reshape_144 = ttnn.reshape(
        ttnn_conv2d_9,
        [1, 80, 80, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_9, False)
    ttnn_permute_35 = ttnn.permute(
        ttnn_reshape_144,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_144, False)
    ttnn_batch_norm_9 = ttnn.batch_norm(
        ttnn_permute_35,
        running_mean=utils_constEvalFuncWrapper_17_0,
        running_var=utils_constEvalFuncWrapper_63_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_66_0,
        bias=utils_constEvalFuncWrapper_60_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_35, False)
    ttnn_silu_7 = ttnn.silu(
        ttnn_batch_norm_9,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_9, False)
    ttnn_permute_36 = ttnn.permute(
        ttnn_silu_7,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_145 = ttnn.reshape(
        ttnn_permute_36,
        [1, 1, 6400, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_36, False)
    ttnn_to_layout_110 = ttnn.to_layout(
        ttnn_reshape_145,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_145, False)
    ttnn_conv2d_10 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_110,
        weight_tensor=input_87,
        device=utils_DeviceGetter_get_device_101,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=80,
        input_width=80,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_110, False)
    ttnn_reshape_146 = ttnn.reshape(
        ttnn_conv2d_10,
        [1, 40, 40, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_10, False)
    ttnn_permute_37 = ttnn.permute(
        ttnn_reshape_146,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_146, False)
    ttnn_batch_norm_10 = ttnn.batch_norm(
        ttnn_permute_37,
        running_mean=utils_constEvalFuncWrapper_80_0,
        running_var=utils_constEvalFuncWrapper_82_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_78_0,
        bias=utils_constEvalFuncWrapper_12_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_37, False)
    ttnn_silu_8 = ttnn.silu(
        ttnn_batch_norm_10,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_10, False)
    ttnn_permute_38 = ttnn.permute(
        ttnn_silu_8,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_8, False)
    ttnn_reshape_147 = ttnn.reshape(
        ttnn_permute_38,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_38, False)
    ttnn_permute_39 = ttnn.permute(
        ttnn_silu_3,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_3, False)
    ttnn_reshape_148 = ttnn.reshape(
        ttnn_permute_39,
        [1, 1, 1600, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_39, False)
    util_create_list_201 = [ttnn_reshape_147, ttnn_reshape_148]
    ttnn_concat_4 = ttnn.concat(
        util_create_list_201,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_148, False)
    ttnn.deallocate(ttnn_reshape_147, False)
    ttnn_to_layout_111 = ttnn.to_layout(
        ttnn_concat_4,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_4, False)
    ttnn_conv2d_11 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_111,
        weight_tensor=input_82,
        device=utils_DeviceGetter_get_device_101,
        in_channels=384,
        out_channels=256,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_111, False)
    ttnn_reshape_149 = ttnn.reshape(
        ttnn_conv2d_11,
        [1, 40, 40, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_11, False)
    ttnn_permute_40 = ttnn.permute(
        ttnn_reshape_149,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_149, False)
    ttnn_batch_norm_11 = ttnn.batch_norm(
        ttnn_permute_40,
        running_mean=utils_constEvalFuncWrapper_6_0,
        running_var=utils_constEvalFuncWrapper_26_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_21_0,
        bias=utils_constEvalFuncWrapper_13_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_40, False)
    ttnn_silu_9 = ttnn.silu(
        ttnn_batch_norm_11,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_11, False)
    ttnn_permute_41 = ttnn.permute(
        ttnn_silu_9,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_9, False)
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_permute_41,
        [1, 1, 1600, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_41, False)
    ttnn_slice_4 = ttnn.slice(
        ttnn_reshape_150,
        [0, 0, 0, 128],
        [1, 1, 1600, 256],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_112 = ttnn.to_layout(
        ttnn_slice_4,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_conv2d_12 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_112,
        weight_tensor=input_77,
        device=utils_DeviceGetter_get_device_101,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_112, False)
    ttnn_reshape_151 = ttnn.reshape(
        ttnn_conv2d_12,
        [1, 40, 40, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_12, False)
    ttnn_permute_42 = ttnn.permute(
        ttnn_reshape_151,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_151, False)
    ttnn_batch_norm_12 = ttnn.batch_norm(
        ttnn_permute_42,
        running_mean=utils_constEvalFuncWrapper_45_0,
        running_var=utils_constEvalFuncWrapper_89_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_29_0,
        bias=utils_constEvalFuncWrapper_72_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_42, False)
    ttnn_silu_10 = ttnn.silu(
        ttnn_batch_norm_12,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_12, False)
    ttnn_permute_43 = ttnn.permute(
        ttnn_silu_10,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_10, False)
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_permute_43,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_43, False)
    ttnn_to_layout_113 = ttnn.to_layout(
        ttnn_reshape_152,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_152, False)
    ttnn_conv2d_13 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_113,
        weight_tensor=input_72,
        device=utils_DeviceGetter_get_device_101,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_113, False)
    ttnn_reshape_153 = ttnn.reshape(
        ttnn_conv2d_13,
        [1, 40, 40, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_13, False)
    ttnn_permute_44 = ttnn.permute(
        ttnn_reshape_153,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_153, False)
    ttnn_batch_norm_13 = ttnn.batch_norm(
        ttnn_permute_44,
        running_mean=utils_constEvalFuncWrapper_5_0,
        running_var=utils_constEvalFuncWrapper_71_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_14_0,
        bias=utils_constEvalFuncWrapper_49_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_44, False)
    ttnn_silu_11 = ttnn.silu(
        ttnn_batch_norm_13,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_13, False)
    ttnn_reshape_154 = ttnn.reshape(
        ttnn_silu_11,
        [1, 4, 32, 40, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_reshape_112,
        input_67,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_4 = ttnn.add(
        ttnn_matmul_8,
        utils_constEvalFuncWrapper_64_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_8, False)
    ttnn_reshape_155 = ttnn.reshape(
        ttnn_add_4,
        [1, 3, 4, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_permute_45 = ttnn.permute(
        ttnn_reshape_154,
        [0, 1, 3, 4, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_154, False)
    ttnn_permute_46 = ttnn.permute(
        ttnn_reshape_155,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_155, False)
    ttnn_reshape_156 = ttnn.reshape(
        ttnn_permute_45,
        [1, 4, 1600, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_45, False)
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_reshape_156,
        ttnn_permute_46,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_156, False)
    ttnn.deallocate(ttnn_permute_46, False)
    ttnn_reshape_157 = ttnn.reshape(
        ttnn_matmul_9,
        [1, 4, 40, 40, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_9, False)
    ttnn_slice_5 = ttnn.slice(
        ttnn_reshape_150,
        [0, 0, 0, 0],
        [1, 1, 1600, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_150, False)
    ttnn_permute_47 = ttnn.permute(
        ttnn_silu_11,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_158 = ttnn.reshape(
        ttnn_permute_47,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_47, False)
    ttnn_to_layout_114 = ttnn.to_layout(
        ttnn_reshape_158,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_158, False)
    ttnn_conv2d_14 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_114,
        weight_tensor=input_92,
        device=utils_DeviceGetter_get_device_101,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_114, False)
    ttnn_reshape_159 = ttnn.reshape(
        ttnn_conv2d_14,
        [1, 40, 40, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_14, False)
    ttnn_permute_48 = ttnn.permute(
        ttnn_reshape_159,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_159, False)
    ttnn_batch_norm_14 = ttnn.batch_norm(
        ttnn_permute_48,
        running_mean=utils_constEvalFuncWrapper_79_0,
        running_var=utils_constEvalFuncWrapper_33_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_22_0,
        bias=utils_constEvalFuncWrapper_83_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_48, False)
    ttnn_max_2 = ttnn.max(
        ttnn_reshape_157,
        [4],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_157, False)
    ttnn_divide_2 = ttnn.divide(
        ttnn_max_2,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_max_2, False)
    ttnn_add_5 = ttnn.add(
        ttnn_divide_2,
        utils_constEvalFuncWrapper_85_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_divide_2, False)
    ttnn_sigmoid_2 = ttnn.sigmoid(
        ttnn_add_5,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_reshape_160 = ttnn.reshape(
        ttnn_sigmoid_2,
        [1, 4, 1, 40, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sigmoid_2, False)
    ttnn_repeat_6 = ttnn.repeat(ttnn_reshape_160, ttnn.Shape([1, 1, 32, 1, 1]))
    ttnn.deallocate(ttnn_reshape_160, False)
    ttnn_reshape_161 = ttnn.reshape(
        ttnn_repeat_6,
        [1, 128, 40, 40],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_6, False)
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_batch_norm_14,
        ttnn_reshape_161,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_161, False)
    ttnn.deallocate(ttnn_batch_norm_14, False)
    ttnn_permute_49 = ttnn.permute(
        ttnn_multiply_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn_reshape_162 = ttnn.reshape(
        ttnn_permute_49,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_49, False)
    ttnn_permute_50 = ttnn.permute(
        ttnn_silu_11,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_11, False)
    ttnn_reshape_163 = ttnn.reshape(
        ttnn_permute_50,
        [1, 1, 1600, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_50, False)
    util_create_list_202 = [
        ttnn_slice_5,
        ttnn_slice_4,
        ttnn_reshape_163,
        ttnn_reshape_162,
    ]
    ttnn_concat_5 = ttnn.concat(
        util_create_list_202,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_163, False)
    ttnn.deallocate(ttnn_reshape_162, False)
    ttnn.deallocate(ttnn_slice_5, False)
    ttnn.deallocate(ttnn_slice_4, False)
    ttnn_to_layout_115 = ttnn.to_layout(
        ttnn_concat_5,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_5, False)
    ttnn_conv2d_15 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_115,
        weight_tensor=input_64,
        device=utils_DeviceGetter_get_device_101,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_115, False)
    ttnn_reshape_164 = ttnn.reshape(
        ttnn_conv2d_15,
        [1, 40, 40, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_15, False)
    ttnn_permute_51 = ttnn.permute(
        ttnn_reshape_164,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_164, False)
    ttnn_batch_norm_15 = ttnn.batch_norm(
        ttnn_permute_51,
        running_mean=utils_constEvalFuncWrapper_25_0,
        running_var=utils_constEvalFuncWrapper_57_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_48_0,
        bias=utils_constEvalFuncWrapper_81_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_51, False)
    ttnn_silu_12 = ttnn.silu(
        ttnn_batch_norm_15,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_15, False)
    ttnn_permute_52 = ttnn.permute(
        ttnn_silu_12,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_165 = ttnn.reshape(
        ttnn_permute_52,
        [1, 1, 1600, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_52, False)
    ttnn_to_layout_116 = ttnn.to_layout(
        ttnn_reshape_165,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_165, False)
    ttnn_conv2d_16 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_116,
        weight_tensor=input_120,
        device=utils_DeviceGetter_get_device_101,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=40,
        input_width=40,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_116, False)
    ttnn_reshape_166 = ttnn.reshape(
        ttnn_conv2d_16,
        [1, 20, 20, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_16, False)
    ttnn_permute_53 = ttnn.permute(
        ttnn_reshape_166,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_166, False)
    ttnn_batch_norm_16 = ttnn.batch_norm(
        ttnn_permute_53,
        running_mean=utils_constEvalFuncWrapper_9_0,
        running_var=utils_constEvalFuncWrapper_11_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_74_0,
        bias=utils_constEvalFuncWrapper_91_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_53, False)
    ttnn_silu_13 = ttnn.silu(
        ttnn_batch_norm_16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_16, False)
    ttnn_permute_54 = ttnn.permute(
        ttnn_silu_13,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_13, False)
    ttnn_reshape_167 = ttnn.reshape(
        ttnn_permute_54,
        [1, 1, 400, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_54, False)
    ttnn_to_layout_117 = ttnn.to_layout(
        input_49,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_49, False)
    ttnn_permute_55 = ttnn.permute(
        ttnn_to_layout_117,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_117, False)
    ttnn_reshape_168 = ttnn.reshape(
        ttnn_permute_55,
        [1, 1, 400, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_55, False)
    util_create_list_203 = [ttnn_reshape_167, ttnn_reshape_168]
    ttnn_concat_6 = ttnn.concat(
        util_create_list_203,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_168, False)
    ttnn.deallocate(ttnn_reshape_167, False)
    ttnn_to_layout_118 = ttnn.to_layout(
        ttnn_concat_6,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_6, False)
    ttnn_conv2d_17 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_118,
        weight_tensor=input_115,
        device=utils_DeviceGetter_get_device_101,
        in_channels=768,
        out_channels=512,
        batch_size=1,
        input_height=20,
        input_width=20,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_118, False)
    ttnn_reshape_169 = ttnn.reshape(
        ttnn_conv2d_17,
        [1, 20, 20, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_17, False)
    ttnn_permute_56 = ttnn.permute(
        ttnn_reshape_169,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_169, False)
    ttnn_batch_norm_17 = ttnn.batch_norm(
        ttnn_permute_56,
        running_mean=utils_constEvalFuncWrapper_50_0,
        running_var=utils_constEvalFuncWrapper_53_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_75_0,
        bias=utils_constEvalFuncWrapper_54_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_56, False)
    ttnn_silu_14 = ttnn.silu(
        ttnn_batch_norm_17,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_17, False)
    ttnn_permute_57 = ttnn.permute(
        ttnn_silu_14,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_14, False)
    ttnn_reshape_170 = ttnn.reshape(
        ttnn_permute_57,
        [1, 1, 400, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_57, False)
    ttnn_slice_6 = ttnn.slice(
        ttnn_reshape_170,
        [0, 0, 0, 256],
        [1, 1, 400, 512],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_119 = ttnn.to_layout(
        ttnn_slice_6,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_conv2d_18 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_119,
        weight_tensor=input_110,
        device=utils_DeviceGetter_get_device_101,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=20,
        input_width=20,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_119, False)
    ttnn_reshape_171 = ttnn.reshape(
        ttnn_conv2d_18,
        [1, 20, 20, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_18, False)
    ttnn_permute_58 = ttnn.permute(
        ttnn_reshape_171,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_171, False)
    ttnn_batch_norm_18 = ttnn.batch_norm(
        ttnn_permute_58,
        running_mean=utils_constEvalFuncWrapper_16_0,
        running_var=utils_constEvalFuncWrapper_51_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_55_0,
        bias=utils_constEvalFuncWrapper_28_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_58, False)
    ttnn_silu_15 = ttnn.silu(
        ttnn_batch_norm_18,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_18, False)
    ttnn_permute_59 = ttnn.permute(
        ttnn_silu_15,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_15, False)
    ttnn_reshape_172 = ttnn.reshape(
        ttnn_permute_59,
        [1, 1, 400, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_59, False)
    ttnn_to_layout_120 = ttnn.to_layout(
        ttnn_reshape_172,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_172, False)
    ttnn_conv2d_19 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_120,
        weight_tensor=input_105,
        device=utils_DeviceGetter_get_device_101,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=20,
        input_width=20,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_120, False)
    ttnn_reshape_173 = ttnn.reshape(
        ttnn_conv2d_19,
        [1, 20, 20, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_19, False)
    ttnn_permute_60 = ttnn.permute(
        ttnn_reshape_173,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_173, False)
    ttnn_batch_norm_19 = ttnn.batch_norm(
        ttnn_permute_60,
        running_mean=utils_constEvalFuncWrapper_70_0,
        running_var=utils_constEvalFuncWrapper_8_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_73_0,
        bias=utils_constEvalFuncWrapper_3_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_60, False)
    ttnn_silu_16 = ttnn.silu(
        ttnn_batch_norm_19,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_19, False)
    ttnn_reshape_174 = ttnn.reshape(
        ttnn_silu_16,
        [1, 8, 32, 20, 20],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_reshape_112,
        input_100,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_112, False)
    ttnn_add_6 = ttnn.add(
        ttnn_matmul_10,
        utils_constEvalFuncWrapper_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn_reshape_175 = ttnn.reshape(
        ttnn_add_6,
        [1, 3, 8, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_permute_61 = ttnn.permute(
        ttnn_reshape_174,
        [0, 1, 3, 4, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_174, False)
    ttnn_permute_62 = ttnn.permute(
        ttnn_reshape_175,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_175, False)
    ttnn_reshape_176 = ttnn.reshape(
        ttnn_permute_61,
        [1, 8, 400, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_61, False)
    ttnn_matmul_11 = ttnn.matmul(
        ttnn_reshape_176,
        ttnn_permute_62,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_176, False)
    ttnn.deallocate(ttnn_permute_62, False)
    ttnn_reshape_177 = ttnn.reshape(
        ttnn_matmul_11,
        [1, 8, 20, 20, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_11, False)
    ttnn_slice_7 = ttnn.slice(
        ttnn_reshape_170,
        [0, 0, 0, 0],
        [1, 1, 400, 256],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_170, False)
    ttnn_permute_63 = ttnn.permute(
        ttnn_silu_16,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_178 = ttnn.reshape(
        ttnn_permute_63,
        [1, 1, 400, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_63, False)
    ttnn_to_layout_121 = ttnn.to_layout(
        ttnn_reshape_178,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_178, False)
    ttnn_conv2d_20 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_121,
        weight_tensor=input_125,
        device=utils_DeviceGetter_get_device_101,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=20,
        input_width=20,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_121, False)
    ttnn_reshape_179 = ttnn.reshape(
        ttnn_conv2d_20,
        [1, 20, 20, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_20, False)
    ttnn_permute_64 = ttnn.permute(
        ttnn_reshape_179,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_179, False)
    ttnn_batch_norm_20 = ttnn.batch_norm(
        ttnn_permute_64,
        running_mean=utils_constEvalFuncWrapper_93_0,
        running_var=utils_constEvalFuncWrapper_87_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_62_0,
        bias=utils_constEvalFuncWrapper_67_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_64, False)
    ttnn_max_3 = ttnn.max(
        ttnn_reshape_177,
        [4],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_177, False)
    ttnn_divide_3 = ttnn.divide(
        ttnn_max_3,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_max_3, False)
    ttnn_add_7 = ttnn.add(
        ttnn_divide_3,
        utils_constEvalFuncWrapper_39_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_divide_3, False)
    ttnn_sigmoid_3 = ttnn.sigmoid(
        ttnn_add_7,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_reshape_180 = ttnn.reshape(
        ttnn_sigmoid_3,
        [1, 8, 1, 20, 20],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sigmoid_3, False)
    ttnn_repeat_7 = ttnn.repeat(ttnn_reshape_180, ttnn.Shape([1, 1, 32, 1, 1]))
    ttnn.deallocate(ttnn_reshape_180, False)
    ttnn_reshape_181 = ttnn.reshape(
        ttnn_repeat_7,
        [1, 256, 20, 20],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_7, False)
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_batch_norm_20,
        ttnn_reshape_181,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_181, False)
    ttnn.deallocate(ttnn_batch_norm_20, False)
    ttnn_permute_65 = ttnn.permute(
        ttnn_multiply_3,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn_reshape_182 = ttnn.reshape(
        ttnn_permute_65,
        [1, 1, 400, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_65, False)
    ttnn_permute_66 = ttnn.permute(
        ttnn_silu_16,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_silu_16, False)
    ttnn_reshape_183 = ttnn.reshape(
        ttnn_permute_66,
        [1, 1, 400, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_66, False)
    util_create_list_204 = [
        ttnn_slice_7,
        ttnn_slice_6,
        ttnn_reshape_183,
        ttnn_reshape_182,
    ]
    ttnn_concat_7 = ttnn.concat(
        util_create_list_204,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_183, False)
    ttnn.deallocate(ttnn_reshape_182, False)
    ttnn.deallocate(ttnn_slice_7, False)
    ttnn.deallocate(ttnn_slice_6, False)
    ttnn_to_layout_122 = ttnn.to_layout(
        ttnn_concat_7,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_7, False)
    ttnn_conv2d_21 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_122,
        weight_tensor=input_97,
        device=utils_DeviceGetter_get_device_101,
        in_channels=1024,
        out_channels=512,
        batch_size=1,
        input_height=20,
        input_width=20,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_122, False)
    ttnn_reshape_184 = ttnn.reshape(
        ttnn_conv2d_21,
        [1, 20, 20, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_21, False)
    ttnn_permute_67 = ttnn.permute(
        ttnn_reshape_184,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_184, False)
    ttnn_batch_norm_21 = ttnn.batch_norm(
        ttnn_permute_67,
        running_mean=utils_constEvalFuncWrapper_95_0,
        running_var=utils_constEvalFuncWrapper_94_0,
        training=False,
        eps=0.0010000000474974513,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_44_0,
        bias=utils_constEvalFuncWrapper_61_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_67, False)
    ttnn_silu_17 = ttnn.silu(
        ttnn_batch_norm_21,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_batch_norm_21, False)
    util_create_list_205 = [ttnn_silu_7, ttnn_silu_12, ttnn_silu_17]
    return util_create_list_205


def load_inputs_for__main():
    utils_DeviceGetter_get_device_102 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_1 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_2 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_3 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_4 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_5 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_6 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_7 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg7.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_9 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg9.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_10 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg10.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_11 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_12 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg12.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_13 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg13.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_14 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_15 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg15.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_16 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg16.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_17 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_18 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg18.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_19 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg19.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_20 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg20.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_21 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg21.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_22 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg22.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_23 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg23.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_24 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg24.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_25 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_26 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg26.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_27 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_28 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg28.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_29 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg29.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_30 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg30.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_31 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg31.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_32 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg32.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_33 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg33.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_34 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg34.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_35 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg35.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_36 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_37 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg37.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_38 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg38.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_39 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg39.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_40 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg40.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_41 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg41.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_42 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_43 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg43.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_44 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg44.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_45 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg45.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_46 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg46.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_47 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg47.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_48 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg48.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_49 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg49.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_50 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg50.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_51 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg51.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_52 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg52.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_53 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg53.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_54 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg54.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_55 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg55.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_56 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg56.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_57 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg57.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_58 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg58.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_59 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg59.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_60 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg60.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_61 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg61.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_62 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg62.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_63 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg63.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_64 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg64.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_65 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg65.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_66 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg66.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_67 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg67.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_68 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg68.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_69 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg69.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_70 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg70.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_71 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg71.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_72 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_73 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg73.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_74 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg74.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_75 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg75.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_76 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg76.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_77 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg77.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_78 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg78.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_79 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg79.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_80 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg80.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_81 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg81.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_82 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg82.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_83 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg83.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_84 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg84.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_85 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg85.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_86 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg86.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_87 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg87.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_88 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg88.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_89 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg89.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_90 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg90.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_91 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg91.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_92 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg92.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_93 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg93.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_94 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg94.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_95 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg95.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_96 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg96.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_97 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg97.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_98 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg98.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_99 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg99.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_100 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg100.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_102,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_101 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg101.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_102 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg102.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_103 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg103.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_104 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg104.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_105 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg105.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_106 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg106.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_107 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg107.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_108 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg108.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_109 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg109.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_110 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg110.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_111 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg111.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_112 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg112.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_113 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg113.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_114 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg114.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_115 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg115.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_116 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg116.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_117 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg117.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_118 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg118.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_119 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg119.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_120 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg120.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_121 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg121.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_122 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg122.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_123 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg123.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_124 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg124.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_125 = utils.load_tensor(
        "yoloworld_pcc_drop/small_640_variant_neck/tensors/arg125.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    util_create_list_206 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
        utils_load_tensor_10,
        utils_load_tensor_11,
        utils_load_tensor_12,
        utils_load_tensor_13,
        utils_load_tensor_14,
        utils_load_tensor_15,
        utils_load_tensor_16,
        utils_load_tensor_17,
        utils_load_tensor_18,
        utils_load_tensor_19,
        utils_load_tensor_20,
        utils_load_tensor_21,
        utils_load_tensor_22,
        utils_load_tensor_23,
        utils_load_tensor_24,
        utils_load_tensor_25,
        utils_load_tensor_26,
        utils_load_tensor_27,
        utils_load_tensor_28,
        utils_load_tensor_29,
        utils_load_tensor_30,
        utils_load_tensor_31,
        utils_load_tensor_32,
        utils_load_tensor_33,
        utils_load_tensor_34,
        utils_load_tensor_35,
        utils_load_tensor_36,
        utils_load_tensor_37,
        utils_load_tensor_38,
        utils_load_tensor_39,
        utils_load_tensor_40,
        utils_load_tensor_41,
        utils_load_tensor_42,
        utils_load_tensor_43,
        utils_load_tensor_44,
        utils_load_tensor_45,
        utils_load_tensor_46,
        utils_load_tensor_47,
        utils_load_tensor_48,
        utils_load_tensor_49,
        utils_load_tensor_50,
        utils_load_tensor_51,
        utils_load_tensor_52,
        utils_load_tensor_53,
        utils_load_tensor_54,
        utils_load_tensor_55,
        utils_load_tensor_56,
        utils_load_tensor_57,
        utils_load_tensor_58,
        utils_load_tensor_59,
        utils_load_tensor_60,
        utils_load_tensor_61,
        utils_load_tensor_62,
        utils_load_tensor_63,
        utils_load_tensor_64,
        utils_load_tensor_65,
        utils_load_tensor_66,
        utils_load_tensor_67,
        utils_load_tensor_68,
        utils_load_tensor_69,
        utils_load_tensor_70,
        utils_load_tensor_71,
        utils_load_tensor_72,
        utils_load_tensor_73,
        utils_load_tensor_74,
        utils_load_tensor_75,
        utils_load_tensor_76,
        utils_load_tensor_77,
        utils_load_tensor_78,
        utils_load_tensor_79,
        utils_load_tensor_80,
        utils_load_tensor_81,
        utils_load_tensor_82,
        utils_load_tensor_83,
        utils_load_tensor_84,
        utils_load_tensor_85,
        utils_load_tensor_86,
        utils_load_tensor_87,
        utils_load_tensor_88,
        utils_load_tensor_89,
        utils_load_tensor_90,
        utils_load_tensor_91,
        utils_load_tensor_92,
        utils_load_tensor_93,
        utils_load_tensor_94,
        utils_load_tensor_95,
        utils_load_tensor_96,
        utils_load_tensor_97,
        utils_load_tensor_98,
        utils_load_tensor_99,
        utils_load_tensor_100,
        utils_load_tensor_101,
        utils_load_tensor_102,
        utils_load_tensor_103,
        utils_load_tensor_104,
        utils_load_tensor_105,
        utils_load_tensor_106,
        utils_load_tensor_107,
        utils_load_tensor_108,
        utils_load_tensor_109,
        utils_load_tensor_110,
        utils_load_tensor_111,
        utils_load_tensor_112,
        utils_load_tensor_113,
        utils_load_tensor_114,
        utils_load_tensor_115,
        utils_load_tensor_116,
        utils_load_tensor_117,
        utils_load_tensor_118,
        utils_load_tensor_119,
        utils_load_tensor_120,
        utils_load_tensor_121,
        utils_load_tensor_122,
        utils_load_tensor_123,
        utils_load_tensor_124,
        utils_load_tensor_125,
    ]
    return util_create_list_206


class ReduceLayers(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.reduce_layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def forward(self, x, idx: int = 0):
        return self.reduce_layers[idx](x)

    def __getitem__(self, idx):
        return self.reduce_layers[idx]


class UpsampleLayers(nn.Module):
    def __init__(self, num_layers: int = 2, scale_factor: int = 2):
        super().__init__()
        self.upsample_layers = nn.ModuleList(
            [nn.Upsample(scale_factor=scale_factor, mode="nearest") for _ in range(num_layers)]
        )

    def __getitem__(self, idx):
        return self.upsample_layers[idx]

    def forward(self, x, idx: int):
        return self.upsample_layers[idx](x)


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, activation=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

        self.batch_norm2d = nn.BatchNorm2d(
            out_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True
        )

        self.activate = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm2d(x)
        x = self.activate(x)
        return x


class DarknetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, add_identity=True):
        super().__init__()
        self.add_identity = add_identity and (in_channels == out_channels)

        # Bottleneck conv layers
        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class MaxSigmoidAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_heads: int = 1,
        with_scale: bool = False,
        use_einsum: bool = True,
    ) -> None:
        super().__init__()

        assert (
            out_channels % num_heads == 0 and embed_channels % num_heads == 0
        ), "out_channels and embed_channels should be divisible by num_heads."

        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        # Optional embedding conv
        self.embed_conv = (
            ConvModule(
                in_channels,
                embed_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=False,
            )
            if embed_channels != in_channels
            else None
        )

        # Guide projection
        self.guide_fc = nn.Linear(guide_channels, embed_channels)

        # Attention bias & scale
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if with_scale else 1.0

        # Output projection (always ConvModule)
        self.project_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            activation=False,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        B, _, H, W = x.shape

        # Guide embedding
        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)

        # Feature embedding
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        # Attention computation
        if self.use_einsum:
            attn_weight = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, -1)

        # Max over guide dimension
        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        # Apply attention
        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)

        return x


class CSPLayerWithTwoConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
    ) -> None:
        super().__init__()

        self.mid_channels = int(out_channels * expand_ratio)

        # First split conv
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
        )

        # Bottleneck blocks
        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                add_identity=add_identity,
            )
            for _ in range(num_blocks)
        )

        # Final fusion conv
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.main_conv(x)

        # Split into two parts
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), dim=1))

        # Sequential bottlenecks
        x_main.extend(block(x_main[-1]) for block in self.blocks)

        # Fuse
        return self.final_conv(torch.cat(x_main, dim=1))


class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        num_heads: int = 1,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        with_scale: bool = False,
        add_identity: bool = True,
        use_einsum: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
            num_blocks=num_blocks,
            add_identity=add_identity,
        )

        self.final_conv = ConvModule(
            (3 + num_blocks) * self.mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
        )

        # Attention block
        self.attn_block = MaxSigmoidAttnBlock(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            use_einsum=use_einsum,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        x_main = self.main_conv(x)

        # Split
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), dim=1))

        # Bottlenecks
        x_main.extend(block(x_main[-1]) for block in self.blocks)

        # Attention branch
        x_main.append(self.attn_block(x_main[-1], guide))

        # Fuse
        return self.final_conv(torch.cat(x_main, dim=1))


def make_round(value, factor):
    return max(int(round(value * factor)), 1)


def make_divisible(value, factor):
    return max(int(round(value * factor)), 1)


class YOLOWorldPAFPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        guide_channels: int,
        embed_channels: List[int],
        num_heads: List[int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        freeze_all: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor

        num_levels = len(in_channels)

        self.num_blocks = make_round(num_csp_blocks, deepen_factor)
        self.upsample_feats_cat_first = True

        # --------------------
        # Reduce / Upsample
        # --------------------
        self.reduce_layers = ReduceLayers(num_levels)
        self.upsample_layers = UpsampleLayers(num_levels - 1)

        # --------------------
        # Top-down layers
        # --------------------
        self.top_down_layers = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            in_ch = make_divisible(
                in_channels[idx] + in_channels[idx - 1],
                widen_factor,
            )
            out_ch = make_divisible(
                out_channels[idx - 1],
                widen_factor,
            )

            self.top_down_layers.append(
                MaxSigmoidCSPLayerWithTwoConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    guide_channels=guide_channels,
                    embed_channels=make_round(embed_channels[idx - 1], widen_factor),
                    num_heads=make_round(num_heads[idx - 1], widen_factor),
                    num_blocks=self.num_blocks,
                    add_identity=False,
                )
            )

        # --------------------
        # Downsample layers
        # --------------------
        self.downsample_layers = nn.ModuleList()
        for i in range(num_levels - 1):
            ch = make_divisible(out_channels[i], widen_factor)
            self.downsample_layers.append(
                ConvModule(
                    ch,
                    ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )

        # --------------------
        # Bottom-up layers
        # --------------------
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(num_levels - 1):
            in_ch = make_divisible(
                out_channels[idx] + out_channels[idx + 1],
                widen_factor,
            )
            out_ch = make_divisible(
                out_channels[idx + 1],
                widen_factor,
            )

            self.bottom_up_layers.append(
                MaxSigmoidCSPLayerWithTwoConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    guide_channels=guide_channels,
                    embed_channels=make_round(embed_channels[idx + 1], widen_factor),
                    num_heads=make_round(num_heads[idx + 1], widen_factor),
                    num_blocks=self.num_blocks,
                    add_identity=False,
                )
            )

        # --------------------
        # Output layers
        # --------------------
        self.out_layers = nn.ModuleList([nn.Identity() for _ in range(num_levels)])

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> Tuple[Tensor, ...]:
        assert len(img_feats) == len(self.in_channels)

        # --------------------
        # Reduce
        # --------------------
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # --------------------
        # Top-down path
        # --------------------
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]

            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)

            if self.upsample_feats_cat_first:
                top_down_in = torch.cat([upsample_feat, feat_low], dim=1)
            else:
                top_down_in = torch.cat([feat_low, upsample_feat], dim=1)

            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](top_down_in, txt_feats)

            inner_outs.insert(0, inner_out)

        # --------------------
        # Bottom-up path
        # --------------------
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], dim=1),
                txt_feats,
            )
            outs.append(out)

        # --------------------
        # Output
        # --------------------
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)


class Neck(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.neck = model

    def forward(self, f1, f2, f3, txt_feats):
        img_feats = [f1, f2, f3]
        return self.neck(img_feats, txt_feats)


def test_ttnn_yoloworld_neck_module(
    guide_channels=512,
    in_channels=[256, 512, 1024],
    out_channels=[256, 512, 1024],
    embed_channels=[128, 256, 512],
    num_heads=[4, 8, 16],
    deepen_factor=0.33,
    widen_factor=0.5,
):
    # ttnn
    load_inputs = load_inputs_for__main()
    ttnn_out = _main(load_inputs)
    ttnn_out[0] = ttnn.to_torch(ttnn_out[0])
    ttnn_out[1] = ttnn.to_torch(ttnn_out[1])
    ttnn_out[2] = ttnn.to_torch(ttnn_out[2])
    logger.info("ttnn output shapes are", ttnn_out[0].shape, ttnn_out[1].shape, ttnn_out[2].shape)

    # cpu(torch)
    custom_model = YOLOWorldPAFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        guide_channels=guide_channels,
        embed_channels=embed_channels,
        num_heads=num_heads,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    )
    custom_model.eval()
    ckpt = torch.load("neck_only.pt", map_location="cpu")
    custom_model.load_state_dict(ckpt, strict=False)
    custom_model.to(torch.bfloat16)
    inputs = torch.load("small_640_inputs.pt", map_location="cpu")
    f1 = inputs["img_feats_0"]  # f1
    f2 = inputs["img_feats_1"]  # f2
    f3 = inputs["img_feats_2"]  # f3
    txt_feats = inputs["txt_feats"]  # txt feats
    cpu_module = Neck(custom_model)
    cpu_out = cpu_module(f1, f2, f3, txt_feats)
    logger.info(f"cpu output shapes are", cpu_out[0].shape, cpu_out[1].shape, cpu_out[2].shape)
    pcc_values0 = comp_pcc(ttnn_out[0], cpu_out[0])
    pcc_values1 = comp_pcc(ttnn_out[1], cpu_out[1])
    pcc_values2 = comp_pcc(ttnn_out[2], cpu_out[2])
    logger.info("values are", pcc_values0, pcc_values1, pcc_values2)  # 0.99,0.99,0.97
