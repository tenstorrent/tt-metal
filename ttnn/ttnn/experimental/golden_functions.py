# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

import ttnn.experimental

if not ttnn.CONFIG.enable_fast_runtime_mode:
    # set golden functions

    def _golden_function(input_tensor, *args, **kwargs):
        return torch.exp(input_tensor)

    ttnn.experimental.tensor.exp.golden_function = _golden_function

    def _golden_function_matmul(input_tensor_a, input_tensor_b, *args, **kwargs):
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

    ttnn.experimental.operations.primary.matmul.golden_function = _golden_function_matmul

    def _golden_function(input_tensor, scalar, attention_mask, *args, **kwargs):
        input_tensor = input_tensor.float()
        input_tensor = input_tensor * scalar
        if attention_mask is not None:
            input_tensor = input_tensor + attention_mask
        ret = torch.softmax(input_tensor, dim=-1)
        return ret

    ttnn.experimental.operations.primary.transformers.scale_mask_softmax_in_place.golden_function = _golden_function

    def _golden_function(tensor, starts, stops, *args, **kwargs):
        for dim, (start, stop) in enumerate(zip(starts, stops)):
            tensor = torch.index_select(tensor, dim, torch.arange(start, stop + 1))
        return tensor

    ttnn.experimental.tensor.unpad.golden_function = _golden_function

    def _golden_function(tensor, grid_size, shard_spec, num_slices, slice, *args, **kwargs):
        tensor = tensor.reshape(1, 1, -1, tensor.shape[-1])
        slice_size = tensor.shape[-2] // num_slices
        start = slice * slice_size
        stop = start + slice_size
        tensor = tensor[:, :, start:stop, :]
        return tensor

    ttnn.experimental.tensor.interleaved_to_sharded_partial.golden_function = _golden_function

    def _nop_golden_function(input_tensor, *args, **kwargs):
        return input_tensor

    ttnn.experimental.tensor.sharded_to_interleaved.golden_function = _nop_golden_function
    ttnn.experimental.tensor.interleaved_to_sharded.golden_function = _nop_golden_function
    ttnn.experimental.tensor.reshard.golden_function = _nop_golden_function
    ttnn.experimental.tensor.tilize.golden_function = _nop_golden_function
    ttnn.experimental.tensor.sharded_to_interleaved_partial.golden_function = _nop_golden_function
