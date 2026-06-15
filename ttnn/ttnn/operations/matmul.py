# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn.decorators import REGISTERED_OPERATIONS
from ttnn.operations.activations import get_golden_function_for_activation

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
MatmulMultiCoreReuseOptimizedProgramFactory = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseOptimizedProgramFactory
create_matmul_attributes = ttnn._ttnn.operations.matmul.create_matmul_attributes
matmul_select_program_factory = ttnn._ttnn.operations.matmul.matmul_select_program_factory


def _matmul_golden_function(
    input_tensor_a,
    input_tensor_b,
    transpose_a=False,
    transpose_b=False,
    *,
    bias=None,
    activation=None,
    program_config=None,
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
    return output_tensor


ttnn.attach_golden_function(ttnn.matmul, golden_function=_matmul_golden_function)


def _linear_golden_function(
    input_tensor_a,
    input_tensor_b,
    transpose_a=False,
    transpose_b=False,
    *,
    bias=None,
    program_config=None,
    activation=None,
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
    return output_tensor


ttnn.attach_golden_function(ttnn.linear, golden_function=_linear_golden_function)


def _addmm_golden_function(input_tensor, mat1_tensor, mat2_tensor, alpha=1.0, beta=1.0, out_tensor=None, **kwargs):
    import torch

    return torch.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha=alpha, beta=beta, out=out_tensor)


ttnn.attach_golden_function(ttnn.addmm, golden_function=_addmm_golden_function)


_CPP_MATMUL = ttnn.matmul
_CPP_LINEAR = ttnn.linear
REGISTERED_OPERATIONS.operations.discard(_CPP_MATMUL)
REGISTERED_OPERATIONS.operations.discard(_CPP_LINEAR)
_AUTO_CONFIG_DOC_SUFFIX = """

Additional Keyword Args:
    auto_config (bool, optional): Enables measured auto-selection and persistent caching for matmul recipes.
        Defaults to `True`. If explicit low-level placement/program configuration arguments such as
        `program_config`, `core_grid`, `compute_kernel_config`, or `output_tile` are supplied,
        auto-configuration is bypassed and the requested configuration is used directly.
"""
_MATMUL_DOC = f"{_CPP_MATMUL.__doc__}{_AUTO_CONFIG_DOC_SUFFIX}"
_LINEAR_DOC = f"{_CPP_LINEAR.__doc__}{_AUTO_CONFIG_DOC_SUFFIX}"


@ttnn.register_python_operation(name="ttnn.matmul", golden_function=_matmul_golden_function, doc=_MATMUL_DOC)
def _matmul_wrapper(
    input_tensor_a,
    input_tensor_b,
    *,
    transpose_a=False,
    transpose_b=False,
    memory_config=None,
    dtype=None,
    program_config=None,
    activation=None,
    compute_kernel_config=None,
    core_grid=None,
    output_tile=None,
    optional_output_tensor=None,
    global_cb=None,
    sub_device_id=None,
    auto_config=True,
):
    from ttnn._experimental.auto_config.matmul import dispatch_matmul

    return dispatch_matmul(
        base_operation=_CPP_MATMUL,
        input_tensor_a=input_tensor_a,
        input_tensor_b=input_tensor_b,
        bias=None,
        is_linear=False,
        auto_config=auto_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        output_tile=output_tile,
        optional_output_tensor=optional_output_tensor,
        global_cb=global_cb,
        sub_device_id=sub_device_id,
    )


@ttnn.register_python_operation(name="ttnn.linear", golden_function=_linear_golden_function, doc=_LINEAR_DOC)
def _linear_wrapper(
    input_tensor_a,
    input_tensor_b,
    *,
    bias=None,
    transpose_a=False,
    transpose_b=False,
    memory_config=None,
    dtype=None,
    program_config=None,
    activation=None,
    compute_kernel_config=None,
    core_grid=None,
    output_tile=None,
    optional_output_tensor=None,
    global_cb=None,
    sub_device_id=None,
    auto_config=True,
):
    from ttnn._experimental.auto_config.matmul import dispatch_matmul

    return dispatch_matmul(
        base_operation=_CPP_LINEAR,
        input_tensor_a=input_tensor_a,
        input_tensor_b=input_tensor_b,
        bias=bias,
        is_linear=True,
        auto_config=auto_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        output_tile=output_tile,
        optional_output_tensor=optional_output_tensor,
        global_cb=global_cb,
        sub_device_id=sub_device_id,
    )


ttnn.Tensor.__matmul__ = lambda self, *args, **kwargs: ttnn.matmul(self, *args, **kwargs)


__all__ = []
