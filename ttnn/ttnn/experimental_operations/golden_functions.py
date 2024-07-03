# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

# set golden functions


def _golden_function(input_tensor, *args, **kwargs):
    import torch

    return torch.exp(input_tensor)


ttnn.attach_golden_function(ttnn.experimental.tensor.exp, _golden_function)


def _golden_function_matmul(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    ret = input_tensor_a.float() @ input_tensor_b.float()
    if "bias" in kwargs:
        ret += kwargs["bias"]
    if (
        "program_config" in kwargs
        and hasattr(kwargs["program_config"], "fused_activation")
        and kwargs["program_config"].fused_activation is not None
    ):
        activation = kwargs["program_config"].fused_activation.op_type.name
        if activation == "GELU":
            ret = torch.nn.functional.gelu(ret)
        elif activation == "RELU":
            ret = torch.nn.functional.relu(ret)
        elif activation == "SILU":
            ret = torch.nn.functional.silu(ret)
        else:
            raise RuntimeError(f"{activation} is not supported as activation function")
    return ret


ttnn.attach_golden_function(ttnn.experimental.operations.primary.matmul, _golden_function_matmul)


def _golden_function(
    input_tensor,
    kv_input_tensor,
    *,
    num_q_heads,
    num_kv_heads,
    transpose_k_heads=True,
    **_,
):
    import torch

    if num_kv_heads is None:
        num_kv_heads = num_q_heads

    batch_size, Z, sequence_size, hidden_size = input_tensor.shape
    head_size = hidden_size // num_q_heads

    query = torch.reshape(input_tensor, (batch_size, sequence_size, num_q_heads, head_size))
    query = torch.permute(query, (0, 2, 1, 3)).contiguous().clone()

    batch_size, Z, sequence_size, hidden_size = kv_input_tensor.shape
    head_size = hidden_size // num_kv_heads // 2
    split_tensors = kv_input_tensor.split(kv_input_tensor.shape[-1] // (2 * num_kv_heads), dim=-1)
    key = torch.concat(split_tensors[::2], dim=-1)
    value = torch.concat(split_tensors[1::2], dim=-1)

    key = torch.reshape(key, (batch_size, sequence_size, num_kv_heads, head_size))
    value = torch.reshape(value, (batch_size, sequence_size, num_kv_heads, head_size))

    key = torch.permute(key, (0, 2, 1, 3)).contiguous().clone()
    value = torch.permute(value, (0, 2, 1, 3)).contiguous().clone()
    if transpose_k_heads:
        key = torch.permute(key, (0, 1, 3, 2)).contiguous().clone()

    return query, key, value


ttnn.attach_golden_function(ttnn.experimental.tensor.create_qkv_heads_from_separate_tensors, _golden_function)


def _golden_function(input_tensor, scalar, attention_mask, *args, **kwargs):
    import torch

    input_tensor = input_tensor.float()
    input_tensor = input_tensor * scalar
    if attention_mask is not None:
        input_tensor = input_tensor + attention_mask
    ret = torch.softmax(input_tensor, dim=-1)
    return ret


ttnn.attach_golden_function(
    ttnn.experimental.operations.primary.transformers.scale_mask_softmax_in_place, _golden_function
)


def _golden_function(input_tensor, *args, **kwargs):
    import torch

    input_tensor = input_tensor.float()
    ret = torch.softmax(input_tensor, dim=-1)
    return ret


ttnn.attach_golden_function(ttnn.experimental.operations.primary.softmax_in_place, _golden_function)


def _golden_function(tensor, starts, stops, *args, **kwargs):
    import torch

    for dim, (start, stop) in enumerate(zip(starts, stops)):
        tensor = torch.index_select(tensor, dim, torch.arange(start, stop + 1))
    return tensor


ttnn.attach_golden_function(ttnn.experimental.tensor.unpad, _golden_function)


def _golden_function(tensor, grid_size, shard_spec, num_slices, slice, *args, **kwargs):
    tensor = tensor.reshape(1, 1, -1, tensor.shape[-1])
    slice_size = tensor.shape[-2] // num_slices
    start = slice * slice_size
    stop = start + slice_size
    tensor = tensor[:, :, start:stop, :]
    return tensor


ttnn.attach_golden_function(ttnn.experimental.tensor.interleaved_to_sharded_partial, _golden_function)


def _golden_function(in0, in1, op, dir, *args, **kwargs):
    in0 = in0.reshape((in1.shape[0], 1, -1, in0.shape[-1]))
    if op == ttnn.experimental.tensor.BcastOpMath.ADD:
        res = in0 + in1
    elif op == ttnn.experimental.tensor.BcastOpMath.SUB:
        res = in0 - in1
    elif op == ttnn.experimental.tensor.BcastOpMath.MUL:
        res = in0 * in1
    return res


ttnn.attach_golden_function(ttnn.experimental.tensor.bcast, _golden_function)


def _nop_golden_function(input_tensor, *args, **kwargs):
    return input_tensor


ttnn.attach_golden_function(ttnn.experimental.tensor.interleaved_to_sharded, _nop_golden_function)
ttnn.attach_golden_function(ttnn.experimental.tensor.reshard, _nop_golden_function)
ttnn.attach_golden_function(ttnn.experimental.tensor.tilize, _nop_golden_function)
