# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc


def run_elt_binary_mul_with_sub_devices(
    batch, num_heads, seq_len, dim, dtype, in_mem_config, out_mem_config, device, mesh_mapper, mesh_composer, pcc
):
    input_shape = [batch, num_heads, seq_len, dim]
    in0 = torch.randn(input_shape)
    in1 = torch.randn(input_shape)
    in0_ref = in0.clone()
    in1_ref = in1.clone()
    in0_t = ttnn.from_torch(
        in0,
        device=device,
        mesh_mapper=mesh_mapper,
        dtype=dtype,
        memory_config=in_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )
    in1_t = ttnn.from_torch(
        in1,
        device=device,
        mesh_mapper=mesh_mapper,
        dtype=dtype,
        memory_config=in_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )
    out_t = ttnn.mul(
        in0_t,
        in1_t,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat8_b,
        memory_config=out_mem_config,
    )
    tt_out = ttnn.to_torch(out_t, mesh_composer=mesh_composer)
    torch_silu = torch.nn.SiLU()
    passing, output = comp_pcc(tt_out, torch.mul(torch_silu(in0_ref), in1_ref), pcc)
    logger.info(output)
    assert passing
