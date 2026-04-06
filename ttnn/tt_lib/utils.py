# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import struct

import torch
import numpy as np

from typing_extensions import deprecated


def _nearest_32(x):
    return math.ceil(x / 32) * 32


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def pad_activation(x):
    """
    This function pads an activation with 0s as a pre-preprocessing step to tilization.

    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"
    if len(x.shape) == 1:  # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, : x.shape[0]] = x
    elif len(x.shape) == 2:  # (batch, num features)
        padded_tensor = torch.zeros(x.shape[0], 1, 32, nearest_32(x.shape[1]))
        padded_tensor[:, 0, 0, : x.shape[1]] = x
    elif len(x.shape) == 3:  # (batch, num features y, num features x)
        padded_tensor = torch.zeros(x.shape[0], 1, nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., 0, : x.shape[-2], : x.shape[-1]] = x
    else:  # (batch, num channels, num features y, num features x)
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., : x.shape[-2], : x.shape[-1]] = x
    return padded_tensor


def pad_weight(x):
    """
    This function pads a weight/bias with 0s as a pre-preprocessing step to tilization.

    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"

    if len(x.shape) == 1:  # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, : x.shape[0]] = x
    elif len(x.shape) == 2:  # (r_features, c_features)
        padded_tensor = torch.zeros(1, 1, nearest_32(x.shape[0]), nearest_32(x.shape[1]))
        padded_tensor[:, 0, : x.shape[0], : x.shape[1]] = x
    else:
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., : x.shape[-2], : x.shape[-1]] = x

    return padded_tensor


def convert_weights_2d_matrix(weights, w_shape):
    """
    :param weights: Input PyTorch Tensor
    :type weights: class:`torch.Tensor`
    """
    ret_shape = [1, 1, w_shape[0], w_shape[1] * w_shape[2] * w_shape[3]]
    if isinstance(weights, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for k in range(w_shape[0]):
        for r in range(w_shape[2]):
            for s in range(w_shape[3]):
                for c in range(w_shape[1]):
                    ret[idx] = weights[k][c][r][s]
                    idx += 1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape).transpose(2, 3)


def convert_act_2d_matrix(activation, kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x):
    """
    :param activation: Input PyTorch Tensor
    :type activation: class:`torch.Tensor`
    """
    N = activation.shape[0]
    C = activation.shape[1]
    H = activation.shape[2]
    W = activation.shape[3]

    OH = (int)((H - kernel_y + 2 * pad_y) // stride_y) + 1
    OW = ((W - kernel_x + 2 * pad_x) // stride_x) + 1
    nrows = OH * OW
    ncols = C * kernel_x * kernel_y
    ret_shape = [1, N, nrows, ncols]
    if isinstance(activation, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for n in range(N):
        for h in range(-1 * pad_y, H + pad_y - kernel_y + 1, stride_y):
            for w in range(-1 * pad_x, W + pad_x - kernel_x + 1, stride_x):
                for r in range(kernel_y):
                    for s in range(kernel_x):
                        for c in range(C):
                            h_offs = h + r
                            w_offs = w + s
                            pad = h_offs < 0 or h_offs >= H or w_offs < 0 or w_offs >= W
                            ret[idx] = 0 if pad else activation[n][c][h_offs][w_offs]
                            idx += 1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape)


@deprecated("PyTorch data is handled automatically in tensor infra. This function does nothing now:")
def tilize(x):
    return x


@deprecated("PyTorch data is handled automatically in tensor infra. This function does nothing now:")
def tilize_to_list(x):
    """
    Returns a flattened list of the tensor
    """
    return tilize(x).reshape(-1).tolist()


@deprecated("PyTorch data is handled automatically in tensor infra. This function does nothing now:")
def untilize(x):
    return x


def print_diff_argmax(a, b, annotation=""):
    """
    Prints out the value of both tensors at a point where the absolute difference is the largest.
    """
    absdiff = (a - b).abs()
    argmax = absdiff.argmax().item()
    diff = absdiff.reshape(-1)[argmax]
    rela = a.abs() / (torch.max(a.abs(), b.abs()))
    relb = b.abs() / (torch.max(a.abs(), b.abs()))
    HT = a.shape[-2] // 32
    WT = a.shape[-1] // 32
    hwt = argmax // 1024
    wt = hwt % WT
    ht = hwt // WT
    h = (argmax % 1024) // 32
    w = (argmax % 1024) % 32
    print("Abs diff=", diff, " at ", argmax, " --- ", annotation, "HTWT=", ht, wt, "HW=", h, w)
    print("  (a=", a.reshape(-1)[argmax].item(), ")")
    print("  (b=", b.reshape(-1)[argmax].item(), ")")
    print("  Rel a=", rela.reshape(-1)[argmax], " at ", argmax)
    print("  Rel b=", relb.reshape(-1)[argmax], " at ", argmax)
    return diff.item()


def tt2torch(ttx):
    """
    Converts an llbuda tiled tensor to torch tensor.
    """
    tt_out = ttx.cpu()
    torch_out = untilize(tt_out.to_torch())
    return torch_out


def tt2torch_rm(ttx):
    """
    Converts an llbuda row-major tensor to torch tensor.
    """
    tt_out = ttx.cpu()
    torch_out = tt_out.to_torch()
    return torch_out


def divup(a, b):
    return (a + b - 1) // b


def roundup(a, b):
    result = divup(a, b) * b
    return result


def roundup32(a):
    return roundup(a, 32)


def float_to_bits(x):
    s = struct.pack(">f", x)
    return struct.unpack(">l", s)[0]


def read_conv_act_into_mm_act_block(
    conv_act, act_address_map_index, address_map, address_map_this_block_size, act_block_h, act_block_w
):
    mm_act_block_shape = [1, 1, act_block_h * 32, act_block_w * 32]
    mm_act_block_size = act_block_h * act_block_w * 1024
    mm_act_block = torch.zeros(mm_act_block_size, dtype=torch.bfloat16).float()
    for i in range(0, address_map_this_block_size, 4):
        src_address = address_map[act_address_map_index]
        dst_address = address_map[act_address_map_index + 1]
        read_size = address_map[act_address_map_index + 2]
        pad = address_map[act_address_map_index + 3]
        for s in range(read_size):
            assert dst_address + s < mm_act_block_size
            if pad:
                mm_act_block[dst_address + s] = 0
            else:
                assert src_address + s < len(conv_act)
                mm_act_block[dst_address + s] = conv_act[src_address + s]
        act_address_map_index += 4
    return (mm_act_block.reshape(mm_act_block_shape), act_address_map_index)


def read_conv_weight_into_mm_weight_block(
    conv_weight,
    weight_address_map_index,
    weight_address_map,
    weight_address_map_this_block_size,
    weight_block_h,
    weight_block_w,
):
    mm_weight_block_shape = [1, 1, weight_block_h * 32, weight_block_w * 32]
    mm_weight_block_size = weight_block_h * weight_block_w * 1024
    mm_weight_block = torch.zeros(mm_weight_block_size, dtype=torch.bfloat16).float()
    for i in range(0, weight_address_map_this_block_size, 4):
        src_address = weight_address_map[weight_address_map_index]
        dst_address = weight_address_map[weight_address_map_index + 1]
        read_size = weight_address_map[weight_address_map_index + 2]
        pad = weight_address_map[weight_address_map_index + 3]
        for s in range(read_size):
            assert dst_address + s < mm_weight_block_size
            if pad:
                mm_weight_block[dst_address + s] = 0
            else:
                assert src_address + s < len(conv_weight)
                mm_weight_block[dst_address + s] = conv_weight[src_address + s]
        weight_address_map_index += 4
    return (mm_weight_block.reshape(mm_weight_block_shape), weight_address_map_index)


def blocked_mm_with_conv_act(
    conv_act,
    mm_weight,
    act_address_map,
    weight_address_map,
    num_blocks_act_h,
    num_blocks_act_w,
    num_blocks_weight_w,
    act_block_h,
    act_block_w,
    weight_block_w,
):
    # act refers to conv activation tensor
    # weight refers to conv weight tensor
    mm_output_shape = [1, 1, num_blocks_act_h * act_block_h * 32, num_blocks_weight_w * weight_block_w * 32]
    ret = torch.zeros(mm_output_shape, dtype=torch.bfloat16).float()
    mm_output_block_shape = [1, 1, act_block_h * 32, weight_block_w * 32]
    act_address_map_index = 0
    weight_address_map_index = 0
    weight_block_h = act_block_w
    num_groups = act_address_map[act_address_map_index]
    assert num_groups == num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w
    weight_num_groups = act_address_map[weight_address_map_index]
    assert weight_num_groups == num_groups
    act_address_map_index += 1
    weight_address_map_index += 1
    for block_act_h in range(num_blocks_act_h):
        # Reset weight (weight) to the starting tile in this column
        for block_weight_w in range(num_blocks_weight_w):
            output_block = torch.zeros(mm_output_block_shape, dtype=torch.bfloat16).float()
            for block_act_w in range(num_blocks_act_w):
                address_map_this_block_size = act_address_map[act_address_map_index]
                act_address_map_index += 1
                weight_address_map_this_block_size = weight_address_map[weight_address_map_index]
                weight_address_map_index += 1
                (mm_act_block, act_address_map_index) = read_conv_act_into_mm_act_block(
                    conv_act,
                    act_address_map_index,
                    act_address_map,
                    address_map_this_block_size,
                    act_block_h,
                    act_block_w,
                )
                (mm_weight_block, weight_address_map_index) = read_conv_weight_into_mm_weight_block(
                    mm_weight,
                    weight_address_map_index,
                    weight_address_map,
                    weight_address_map_this_block_size,
                    weight_block_h,
                    weight_block_w,
                )
                # Untilize weight block (this CPU reference does matmul on untilized blocks)
                mm_weight_block = untilize(mm_weight_block)
                for out_h_block in range(act_block_h * 32):
                    for out_w_block in range(weight_block_w * 32):
                        output_block[0][0][out_h_block][out_w_block] += torch.dot(
                            mm_act_block[0, 0, out_h_block, :].reshape(-1),
                            mm_weight_block[0, 0, :, out_w_block].reshape(-1),
                        )
            start_oh = block_act_h * act_block_h * 32
            start_ow = block_weight_w * weight_block_w * 32
            end_oh = start_oh + (act_block_h * 32)
            end_ow = start_ow + (weight_block_w * 32)
            ret[0, 0, start_oh:end_oh, start_ow:end_ow] = output_block

    return ret


def is_close(a, b, rtol=8e-2, atol=8e-2, max_mag=4.0, max_mag_fraction=0.02):
    """
    An improved variant of isclose, taking into account max magnitude in the sum, with logging.
    """
    absdiff = (a - b).abs()
    reldiff1 = (a.abs() / b.abs()) - 1.0
    reldiff2 = (a.abs() + 1.0) / (b.abs() + 1.0) - 1.0  # in case b.abs() is 0
    reldiff_or = torch.logical_or(reldiff1.abs() < rtol, reldiff2.abs() < rtol)
    max_mag_ok = absdiff < max_mag * max_mag_fraction

    or_abs_rel = torch.logical_or(absdiff < atol, reldiff_or)
    or_abs_rel = torch.logical_or(or_abs_rel, max_mag_ok)
    debug_index = or_abs_rel.to(torch.int32).argmin().item()
    if not or_abs_rel.reshape(-1)[debug_index]:
        print("****   is_close mismatch at index=", debug_index)
        print(a.reshape(-1)[debug_index])
        print(b.reshape(-1)[debug_index])
        print("****    reldiff1=", reldiff1.reshape(-1)[debug_index])
        print("****    reldiff2=", reldiff2.reshape(-1)[debug_index])
        print("****    absdiff=", absdiff.reshape(-1)[debug_index])
        HT = a.shape[-2] // 32
        WT = a.shape[-1] // 32
        hwt = debug_index // 1024
        wt = hwt % WT
        ht = hwt // WT
        h = (debug_index % 1024) // 32
        w = (debug_index % 1024) % 32
        print("****    at ", debug_index, " --- ", "HTWT=", ht, wt, "HW=", h, w)
    return torch.all(or_abs_rel)


def find_closest_largest_divisor(num: int, start_divisor: int):
    divisor = start_divisor
    while num % divisor != 0:
        divisor = divisor - 1
    return divisor


def find_closest_largest_divisor_with_num_padding(num: int, start_divisor: int):
    divisor = start_divisor
    padded_num = _nearest_y(num, divisor)
    while (padded_num - num) >= (int)(padded_num / divisor):
        divisor = divisor - 1
        padded_num = _nearest_y(num, divisor)
    return divisor
