# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3 kw-fold path (HEIGHT_SHARDED TILE, stride 1, padding == dilation, W % 32 == 0, with bias), forced on.

The selection gate (`_kw_fold_pays`) takes kw-fold only where it measured faster, which none of the other unit shapes
reach, so this file forces it to cover the path itself:
- column rounds in both forms: whole-K (all three rounds resident) and K-blocked (rounds streamed, re-streamed per
  M-block), with weight servers and without;
- Q_1 packed as the overwrite the Q_0 / Q_2 rounds accumulate onto, across several M-blocks per core;
- the final pass's bias add from the reader's decoded bias rows: bf8_b and bf16 bias tiles;
- dilation 2, C_out off a multiple of the plane width, a ragged last core, HiFi4;
- the fused side rounds (2-tile n-groups: Q_0 | Q_2 as one 1 x 4 pass) on K-blocked weights, one and several n-groups.
"""

import importlib

import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")

# (id, H, W, C_in, C_out, dilation, activation dtype, weight dtype, fidelity, relu, cores)
CASES = [
    ("n1_bf8", 32, 64, 64, 32, 1, "bf8", "bf8", "LoFi", True, 16),
    ("bf16_bias_kblocked", 32, 64, 128, 128, 1, "bf16", "bf16", "LoFi", True, 16),
    ("bf16_act_bf8_w_hifi4", 32, 64, 64, 64, 1, "bf16", "bf8", "HiFi4", False, 16),
    ("d2_bf8", 64, 64, 64, 64, 2, "bf8", "bf8", "LoFi", True, 32),
    ("cout48_bf8", 32, 96, 64, 48, 1, "bf8", "bf8", "LoFi", True, 24),
    ("multiblock_bf8", 64, 128, 64, 64, 1, "bf8", "bf8", "LoFi", True, 8),
    ("kblocked_multiblock_hifi4", 64, 128, 128, 128, 1, "bf8", "bf8", "HiFi4", True, 8),
    ("ragged_bf8", 48, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 20),
    ("many_blocks_whole_k", 128, 256, 32, 32, 1, "bf8", "bf8", "LoFi", True, 4),
    ("many_blocks_kblocked", 64, 128, 128, 128, 1, "bf8", "bf8", "LoFi", True, 8),
    # Fused side rounds (2-tile n-groups) on K-blocked weights: one n-group (N = 2) and five (N = 10).
    ("kblocked_n2", 32, 64, 256, 64, 1, "bf8", "bf8", "LoFi", True, 16),
    ("kblocked_n10", 32, 128, 64, 320, 1, "bf8", "bf8", "LoFi", True, 32),
]
DTYPES = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_conv2d_3x3_kw_fold_forced(case, device, monkeypatch):
    import torch

    _, H, W, c_in, c_out, d, act, wdt, fidelity, relu, cores = case
    dtype, weight_dtype = DTYPES[act], DTYPES[wdt]
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            [per_core * 32, c_in],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    taken = []
    kw_fold_mode = _pd.kw_fold_mode
    monkeypatch.setattr(_pd, "_kw_fold_pays", lambda *a: True)
    monkeypatch.setattr(_pd, "kw_fold_mode", lambda *a: taken.append(kw_fold_mode(*a)) or taken[-1])

    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    torch_weight = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    kwargs = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True
    )
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_output = conv2d_3x3(
        tt_input,
        prepare_conv2d_weights(torch_weight, **kwargs),
        input_height=H,
        input_width=W,
        bias=prepare_conv2d_bias(torch_bias, **kwargs),
        padding=d,
        stride=1,
        dilation=d,
        activation="relu" if relu else None,
        compute_kernel_config=compute_config,
    )
    assert True in taken, f"kw-fold not taken: {taken}"

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    pcc = 0.997 if dtype == weight_dtype == ttnn.bfloat16 else 0.99
    assert_with_pcc(ref, ttnn.to_torch(tt_output).float().reshape(ref.shape), pcc)


# Halo exchange (kw_fold.hpp): with one M-block per core, a core boundary inside an image row takes the boundary rows'
# missing kw partial from the neighbour core instead of recomputing a halo tile-row. It sums the same partials in the
# same order, so it must match the recompute bit for bit. Every shape puts core boundaries inside image rows: K-blocked
# with and without weight servers, whole-K, LoFi / HiFi2 / HiFi4, dilation 2 (2-row strips), N = 1 (4-row subblock
# groups), two ACC column planes, a ragged last core of one tile-row, bf16 activations and bias.
# (id, H, W, C_in, C_out, dilation, activation dtype, weight dtype, fidelity, relu, cores)
XCHG_CASES = [
    ("kblocked_lofi", 64, 128, 128, 128, 1, "bf8", "bf8", "LoFi", True, 86),
    ("kblocked_hifi4", 64, 128, 128, 128, 1, "bf8", "bf8", "HiFi4", True, 86),
    ("kblocked_two_planes", 64, 128, 128, 256, 1, "bf8", "bf8", "LoFi", True, 86),
    ("kblocked_few_cores", 64, 128, 128, 128, 1, "bf8", "bf8", "LoFi", False, 30),
    ("kblocked_no_server", 88, 160, 128, 128, 1, "bf8", "bf8", "LoFi", True, 110),
    ("d2_bf8", 64, 128, 64, 64, 2, "bf8", "bf8", "LoFi", True, 86),
    ("n1_bf8", 64, 128, 64, 32, 1, "bf8", "bf8", "LoFi", True, 52),
    ("n3_hifi2", 64, 96, 64, 96, 1, "bf8", "bf8", "HiFi2", True, 40),
    ("ragged_one_row", 48, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 20),
    ("bf16", 32, 128, 64, 64, 1, "bf16", "bf16", "LoFi", True, 43),
    ("whole_k_n1", 128, 256, 32, 32, 1, "bf8", "bf8", "LoFi", True, 103),
    ("kblocked_n2_hifi4", 64, 128, 256, 64, 1, "bf8", "bf8", "HiFi4", False, 86),
    ("whole_k_n2_one_row", 32, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 64),
]


@pytest.mark.parametrize("case", XCHG_CASES, ids=[c[0] for c in XCHG_CASES])
def test_conv2d_3x3_kw_fold_halo_exchange(case, device, monkeypatch):
    import torch

    _, H, W, c_in, c_out, d, act, wdt, fidelity, relu, cores = case
    dtype, weight_dtype = DTYPES[act], DTYPES[wdt]
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    if cores > grid.x * grid.y:
        pytest.skip(f"{cores} cores do not fit {grid.x}x{grid.y}")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            [per_core * 32, c_in],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    taken = []
    kw_fold_mode = _pd.kw_fold_mode
    monkeypatch.setattr(_pd, "_kw_fold_pays", lambda *a: True)
    monkeypatch.setattr(_pd, "kw_fold_mode", lambda *a: taken.append(kw_fold_mode(*a)) or taken[-1])
    exchange = [True]
    monkeypatch.setattr(_pd, "kw_fold_halo_exchange", lambda b: exchange[0] and b.max_m_blocks_per_core == 1)

    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    torch_weight = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    kwargs = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True
    )

    def run():
        tt_input = ttnn.from_torch(
            torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        tt_output = conv2d_3x3(
            tt_input,
            prepare_conv2d_weights(torch_weight, **kwargs),
            input_height=H,
            input_width=W,
            bias=prepare_conv2d_bias(torch_bias, **kwargs),
            padding=d,
            stride=1,
            dilation=d,
            activation="relu" if relu else None,
            compute_kernel_config=compute_config,
        )
        return ttnn.to_torch(tt_output).float()

    exchanged = run()
    again = run()  # program-cache hit: the exchange's semaphores and tokens start from zero again
    exchange[0] = False
    recomputed = run()
    assert taken and all(taken), f"kw-fold not taken: {taken}"
    assert torch.equal(again, exchanged), "program-cache re-run differs"
    assert torch.equal(
        exchanged, recomputed
    ), f"{int((exchanged != recomputed).sum())} outputs differ from the recompute"

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    pcc = 0.997 if dtype == weight_dtype == ttnn.bfloat16 else 0.99
    assert_with_pcc(ref, exchanged.reshape(ref.shape), pcc)
