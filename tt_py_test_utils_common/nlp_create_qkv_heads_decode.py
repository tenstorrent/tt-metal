# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
from tt_py_test_utils_common.comparison_funcs import comp_pcc


def run_test_create_min_width_shard(
    device,
    batch,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    overlap_coregrid,
    batch_offset=None,
    slice_size=None,
    sub_core_grids=None,
):
    # Split Heads
    if not overlap_coregrid and (slice_size >= 32 if slice_size is not None else batch >= 32):
        # Test with smaller batch size for CI to pass on devices not utilizing full coregrid
        pytest.skip(
            "Skipping tests for batch_per_device>=32 for non-overlapping coregrid as CI device does not support full coregrid"
        )
    seq_len = 1
    total_heads = n_local_heads + n_local_kv_heads * 2
    total_cores = total_heads * head_dim // 32

    device_core_grid_size = device.compute_with_storage_grid_size()
    device_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device_core_grid_size.x - 1, device_core_grid_size.y - 1),
            ),
        }
    )
    if sub_core_grids is not None:
        sub_core_grids_bounds = sub_core_grids.bounding_box()
        if (
            sub_core_grids_bounds.start.x < 0
            or sub_core_grids_bounds.start.y < 0
            or sub_core_grids_bounds.end.x >= device_core_grid_size.x
            or sub_core_grids_bounds.end.y >= device_core_grid_size.y
        ):
            pytest.skip("Sub core grid is out of bounds")
        device_core_grid = sub_core_grids

    device_core_grid_start = device_core_grid.bounding_box().start
    input_shard_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        device_core_grid_start, total_cores, device_core_grid, True
    )

    CREATE_HEAD_INPUT_SHARD_SPEC = ttnn.ShardSpec(
        input_shard_core_grid,
        [
            32,
            32,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, CREATE_HEAD_INPUT_SHARD_SPEC
    )

    CREATE_HEAD_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            device_core_grid,
            [
                32,
                head_dim,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Prepare tt input
    proj_output = torch.rand(1, seq_len, batch, head_dim * total_heads)
    proj_output_tt = ttnn.from_torch(
        proj_output, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=CREATE_HEAD_INPUT_MEMCFG
    )

    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        overlap_qk_coregrid=overlap_coregrid,
        batch_offset=batch_offset,
        slice_size=slice_size,
        memory_config=CREATE_HEAD_OUTPUT_MEMCFG,
    )
    logger.debug(f"q_heads_tt: {q_heads_tt.shape}, {q_heads_tt.memory_config()}")
    logger.debug(f"k_heads_tt: {k_heads_tt.shape}, {k_heads_tt.memory_config()}")
    logger.debug(f"v_heads_tt: {v_heads_tt.shape}, {v_heads_tt.memory_config()}")

    if batch_offset is None and slice_size is None:
        batch_offset = 0
        slice_size = batch
    else:
        if isinstance(batch_offset, ttnn.Tensor):
            # convert ttnn.Tensor to torch tensor
            tensor = ttnn.to_torch(batch_offset)
            batch_offset = tensor[0]
        batch = slice_size
    q_heads_torch = proj_output[:, :, batch_offset : batch_offset + slice_size, : head_dim * n_local_heads].view(
        seq_len, batch, n_local_heads, head_dim
    )
    k_heads_torch = proj_output[
        :,
        :,
        batch_offset : batch_offset + slice_size,
        head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads),
    ].view(seq_len, batch, n_local_kv_heads, head_dim)
    v_heads_torch = proj_output[
        :, :, batch_offset : batch_offset + slice_size, head_dim * (n_local_heads + n_local_kv_heads) :
    ].view(seq_len, batch, n_local_kv_heads, head_dim)

    # compare
    q_heads_tt_cpu = ttnn.to_torch(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch)
    logger.debug(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = ttnn.to_torch(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch)
    logger.debug(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = ttnn.to_torch(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch)
    logger.debug(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v
