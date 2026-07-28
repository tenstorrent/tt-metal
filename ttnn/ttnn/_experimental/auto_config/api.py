# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Public auto-configuration matmul / linear entrypoints.

Exposed as ``ttnn.experimental.auto_config.matmul`` and
``ttnn.experimental.auto_config.linear``.

These are standalone utility functions: they run measured auto-configuration
(candidate selection, tuning, and persistent caching) and dispatch to the stock
kernel-backed ``ttnn.matmul`` / ``ttnn.linear`` as the base operation. They do
NOT modify the public ``ttnn.matmul`` / ``ttnn.linear`` entrypoints -- callers
opt in explicitly by calling these functions. Passing ``auto_config=False`` (or
supplying explicit low-level placement / program-configuration arguments such as
``program_config``, ``core_grid``, ``compute_kernel_config`` or ``output_tile``)
bypasses tuning and runs the base op directly, so the result is identical to a
stock ``ttnn.matmul`` / ``ttnn.linear`` call.
"""

import ttnn


def matmul(
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
    queue_id=None,
    cq_id=None,
):
    from ttnn._experimental.auto_config._selector import dispatch_matmul

    dispatch_kwargs = {}
    if queue_id is not None:
        dispatch_kwargs["queue_id"] = queue_id
    elif cq_id is not None:
        dispatch_kwargs["cq_id"] = cq_id

    return dispatch_matmul(
        base_operation=ttnn.matmul,
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
        **dispatch_kwargs,
    )


def linear(
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
    queue_id=None,
    cq_id=None,
):
    from ttnn._experimental.auto_config._selector import dispatch_matmul

    dispatch_kwargs = {}
    if queue_id is not None:
        dispatch_kwargs["queue_id"] = queue_id
    elif cq_id is not None:
        dispatch_kwargs["cq_id"] = cq_id

    return dispatch_matmul(
        base_operation=ttnn.linear,
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
        **dispatch_kwargs,
    )
