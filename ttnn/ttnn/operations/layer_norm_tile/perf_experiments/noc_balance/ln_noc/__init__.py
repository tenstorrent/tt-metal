# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Private copy of layer_norm_tile with a per-core NoC plan (noc_balance bake-off)."""

import ttnn

from ttnn.operations.layer_norm_tile import validate

import importlib.util
import os
import sys
from pathlib import Path

from .program_descriptor import create_program_descriptor, select_blocking

if os.environ.get("NB_IMPL") == "graduate":
    # The exact host change proposed for the op (graduate/), kernels unchanged.
    _spec = importlib.util.spec_from_file_location(
        "nb_graduate", Path(__file__).parent.parent / "graduate" / "layer_norm_tile_program_descriptor.py"
    )
    _grad = importlib.util.module_from_spec(_spec)
    sys.modules["nb_graduate"] = _grad
    _spec.loader.exec_module(_grad)
    _grad.NOC_BALANCE = os.environ.get("NB_BALANCE", "1") == "1"
    select_blocking = _grad.select_blocking

    def create_program_descriptor(*args, plan=None, **kwargs):
        return _grad.create_program_descriptor(*args, **kwargs)


def layer_norm(
    input_tensor,
    gamma,
    beta,
    residual_input_tensor=None,
    *,
    epsilon=1e-5,
    compute_kernel_config,
    memory_config=None,
    plan=None,
):
    validate(
        input_tensor,
        gamma,
        beta,
        residual_input_tensor,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config if memory_config is not None else input_tensor.memory_config(),
    )
    blocking = select_blocking(
        input_tensor,
        residual=residual_input_tensor,
        fp32_dest_acc_en=bool(compute_kernel_config.fp32_dest_acc_en),
    )
    assert blocking is not None
    resolved = None if plan is None else plan(input_tensor, residual_input_tensor, blocking)
    descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        beta,
        residual_input_tensor,
        output_tensor,
        blocking,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        plan=resolved,
    )
    io_tensors = [input_tensor]
    if residual_input_tensor is not None:
        io_tensors.append(residual_input_tensor)
    io_tensors += [gamma, beta, output_tensor]
    return ttnn.generic_op(io_tensors, descriptor)
