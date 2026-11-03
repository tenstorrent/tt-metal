# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import ttnn
from ttnn.decorators import get_golden_function
from ttnn.operations.activations import get_golden_function_for_activation
from ttnn.operations.golden_common import golden_to_output_dtype

MatmulProgramConfig = ttnn._ttnn.operations.matmul.MatmulProgramConfig
MatmulMultiCoreReuseProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseProgramConfig
MatmulMultiCoreReuseMultiCastProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastProgramConfig
MatmulMultiCoreReuseMultiCast1DProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCast1DProgramConfig
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig = (
    ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
)
MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig = (
    ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig
)
MatmulParams = ttnn._ttnn.operations.matmul.MatmulParams
MatmulInputs = ttnn._ttnn.operations.matmul.MatmulInputs
MatmulDeviceOperation = ttnn._ttnn.operations.matmul.MatmulDeviceOperation
create_matmul_attributes = ttnn._ttnn.operations.matmul.create_matmul_attributes
matmul_select_program_factory = ttnn._ttnn.operations.matmul.matmul_select_program_factory


def _golden_function(
    input_tensor_a,
    input_tensor_b,
    transpose_a=False,
    transpose_b=False,
    *,
    bias=None,
    activation=None,
    program_config=None,
    dtype=None,
    **kwargs,
):
    import torch

    if transpose_a:
        input_tensor_a = input_tensor_a.transpose(-1, -2)
    if transpose_b:
        input_tensor_b = input_tensor_b.transpose(-1, -2)
    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    # First check if there is a fused activation in the program config
    if program_config is not None and hasattr(program_config, "fused_activation") and program_config.fused_activation:
        program_config_activation = program_config.fused_activation.op_type
        output_tensor = get_golden_function_for_activation(program_config_activation)(output_tensor)

    # Do the composite op activation function if it is requested
    elif activation is not None:
        output_tensor = get_golden_function_for_activation(activation)(output_tensor)

    while len(output_tensor.shape) > len(input_tensor_a.shape):
        output_tensor = output_tensor.squeeze(0)
    return golden_to_output_dtype(output_tensor, dtype)


ttnn.attach_golden_function(
    ttnn.matmul,
    golden_function=_golden_function,
)


def _golden_function(
    input_tensor_a,
    input_tensor_b,
    transpose_a=False,
    transpose_b=False,
    *,
    bias=None,
    program_config=None,
    activation=None,
    dtype=None,
    **kwargs,
):
    import torch

    if transpose_a:
        input_tensor_a = input_tensor_a.transpose(-1, -2)
    if transpose_b:
        input_tensor_b = input_tensor_b.transpose(-1, -2)
    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    if bias is not None:
        if len(bias) == 2:
            if bias.shape[0] != 1:
                raise RuntimeError(f"bias must be a 1D tensor")
            bias = bias[0]
        output_tensor += bias

    # First check if there is a fused activation in the program config
    if program_config is not None and hasattr(program_config, "fused_activation") and program_config.fused_activation:
        program_config_activation = program_config.fused_activation.op_type
        output_tensor = get_golden_function_for_activation(program_config_activation)(output_tensor)

    # Do the composite op activation function if it is requested
    elif activation is not None:
        output_tensor = get_golden_function_for_activation(activation)(output_tensor)

    while len(output_tensor.shape) > len(input_tensor_a.shape):
        output_tensor = output_tensor.squeeze(0)
    return golden_to_output_dtype(output_tensor, dtype)


ttnn.attach_golden_function(
    ttnn.linear,
    golden_function=_golden_function,
)


def _golden_function(
    input_tensor,
    mat1_tensor,
    mat2_tensor,
    alpha=1.0,
    beta=1.0,
    out_tensor=None,
    *,
    dtype=None,
    _ttnn_output_tensor_dtype=None,
    **kwargs,
):
    import torch

    if beta == 0:
        # TTNN intentionally ignores the addend when beta is zero, including an otherwise invalid addend shape.
        result = alpha * torch.matmul(mat1_tensor, mat2_tensor)
    else:
        result = torch.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha=alpha, beta=beta)
    result = golden_to_output_dtype(result, dtype if dtype is not None else _ttnn_output_tensor_dtype)
    if out_tensor is not None:
        out_tensor.copy_(result)
        return out_tensor
    return result


def _preprocess_addmm_golden_inputs(function_args, function_kwargs):
    """Record the preallocated output dtype, which fixes the stored values when dtype is omitted.
    Default preprocessing converts that tensor to Torch and loses block-float dtypes.
    """

    golden_args, golden_kwargs = ttnn.decorators.default_preprocess_golden_function_inputs(
        function_args, function_kwargs
    )
    golden_kwargs["_ttnn_output_tensor_dtype"] = getattr(function_kwargs.get("optional_output_tensor"), "dtype", None)
    return golden_args, golden_kwargs


ttnn.attach_golden_function(
    ttnn.addmm,
    golden_function=_golden_function,
    preprocess_golden_function_inputs=_preprocess_addmm_golden_inputs,
)


def _golden_function_matmul_batched_weights(input_tensor_a, input_tensors_b, *_, dtype=None, **__):
    import torch

    # One input tensor a multiplied against each of the batched weight tensors b.
    return [
        golden_to_output_dtype(torch.matmul(input_tensor_a, b.to(input_tensor_a.dtype)), dtype) for b in input_tensors_b
    ]


ttnn.attach_golden_function(ttnn.matmul_batched_weights, golden_function=_golden_function_matmul_batched_weights)


def _pairwise_group_matmul(a, b):
    """Multiply every batch block of A with every group block of B; the result has A's batch dims, then B's."""

    import torch

    a_batch, b_batch = a.shape[:-2], b.shape[:-2]
    a_exp = a.reshape(*a_batch, *([1] * len(b_batch)), *a.shape[-2:])
    b_exp = b.reshape(*([1] * len(a_batch)), *b_batch, *b.shape[-2:])
    return torch.matmul(a_exp, b_exp.to(a.dtype))


def _sparse_matmul_golden_result(
    input_tensor_a,
    input_tensor_b,
    *,
    sparsity,
    is_input_a_sparse=False,
    is_input_b_sparse=True,
    nnz=None,
    indices=None,
    optional_output_tensor=None,
    **__,
):
    import torch

    a, b = input_tensor_a, input_tensor_b
    if not is_input_a_sparse and not is_input_b_sparse:
        raise ValueError("sparse_matmul requires at least one sparse input")
    if indices is not None:
        # Indexed mode gathers the listed groups of B into a compact group axis, in index order,
        # and never reads sparsity. A sparse A is already compact, with one block per listed group.
        b_selected = b.index_select(-3, indices.reshape(-1).to(torch.int64)).to(a.dtype)
        if is_input_a_sparse:
            return torch.matmul(a, b_selected)
        return _pairwise_group_matmul(a, b_selected)

    if is_input_a_sparse:
        dense = torch.matmul(a, b.to(a.dtype))
    else:
        # Dense-A/sparse-B mode forms every pair from A's batch dims and B's sparse-group dims.
        dense = _pairwise_group_matmul(a, b)
    mask = (sparsity != 0).reshape(dense.shape[:-2])
    expanded_output = dense * mask.unsqueeze(-1).unsqueeze(-1).to(dense.dtype)

    if optional_output_tensor is not None and optional_output_tensor.shape != expanded_output.shape:
        compact_shape = (1, nnz, a.shape[-2], b.shape[-1])
        if nnz is None or tuple(optional_output_tensor.shape) != compact_shape:
            raise ValueError(f"sparse_matmul output shape {tuple(optional_output_tensor.shape)} is not supported")
        # A compact output of shape [1, nnz, M, N] packs the active results in sparsity scan order.
        active_results = expanded_output.reshape(-1, *expanded_output.shape[-2:])[mask.reshape(-1)]
        return active_results.reshape(compact_shape)
    return expanded_output


def _golden_function_sparse_matmul(input_tensor_a, input_tensor_b, *, dtype=None, **kwargs):
    return golden_to_output_dtype(_sparse_matmul_golden_result(input_tensor_a, input_tensor_b, **kwargs), dtype)


ttnn.attach_golden_function(ttnn.sparse_matmul, golden_function=_golden_function_sparse_matmul)


ttnn.Tensor.__matmul__ = lambda self, *args, **kwargs: ttnn.matmul(self, *args, **kwargs)


__all__ = []
