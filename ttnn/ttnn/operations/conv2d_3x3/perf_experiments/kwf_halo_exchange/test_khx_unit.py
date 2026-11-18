# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The kw-fold halo exchange against the recompute it replaces: kw-fold forced on in both the real op (HEAD
3ca4eb28f46, halo tile-rows recomputed) and this experiment's copy (KHX_MODE, default 2: boundary rows exchanged with
the neighbour cores), same inputs, one process. The exchange sums Q_1, Q_0, Q_2 in the recompute's order, so the
outputs must be bit-identical; both are also checked against a fp32 torch conv.

Shapes are picked so core boundaries fall inside image rows (W / 32 tile-rows per image row does not divide the
per-core tile-rows): K-blocked (with and without weight servers) and whole-K, LoFi and HiFi4, dilation 2 (2-row
spill), N = 1 (4-row subblock groups: the last group runs first), two ACC column planes, a ragged last core of one
tile-row, bf16 activations and bias, both boundaries on one core.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_halo_exchange/test_khx_unit.py
"""

import importlib
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from khx_copy import copy_modules  # noqa: E402

# (id, H, W, C_in, C_out, dilation, activation dtype, weight dtype, fidelity, relu, cores)
CASES = [
    ("kblocked_lofi_res3", 64, 128, 128, 128, 1, "bf8", "bf8", "LoFi", True, 86),
    ("kblocked_hifi4_res3", 64, 128, 128, 128, 1, "bf8", "bf8", "HiFi4", True, 86),
    ("kblocked_two_planes", 64, 128, 128, 256, 1, "bf8", "bf8", "LoFi", True, 86),
    ("kblocked_few_cores", 64, 128, 128, 128, 1, "bf8", "bf8", "LoFi", False, 30),
    ("d2_spill2", 64, 128, 64, 64, 2, "bf8", "bf8", "LoFi", True, 86),
    ("n1_sbh4", 64, 128, 64, 32, 1, "bf8", "bf8", "LoFi", True, 52),
    ("n3_one_plane", 64, 96, 64, 96, 1, "bf8", "bf8", "HiFi2", True, 40),
    ("ragged_valid1", 48, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 20),
    ("bf16_act_bias", 32, 128, 64, 64, 1, "bf16", "bf16", "LoFi", True, 43),
    ("whole_k_n1", 128, 256, 32, 32, 1, "bf8", "bf8", "LoFi", True, 103),
    ("kblocked_no_server", 88, 160, 128, 128, 1, "bf8", "bf8", "LoFi", True, 110),
]
DTYPES = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


def ttnn_khx_mode():
    import os

    return int(os.environ.get("KHX_MODE", "2"))


def _force_kw_fold(monkeypatch, pd, taken):
    real_mode = pd.kw_fold_mode
    monkeypatch.setattr(pd, "_kw_fold_pays", lambda *a: True)
    monkeypatch.setattr(pd, "kw_fold_mode", lambda *a: taken.append(real_mode(*a)) or taken[-1])


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_khx_vs_recompute(case, device, monkeypatch):
    import torch

    _, H, W, c_in, c_out, d, act, wdt, fidelity, relu, cores = case
    dtype, weight_dtype = DTYPES[act], DTYPES[wdt]
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    if cores > grid.x * grid.y:
        pytest.skip("grid too small")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            [per_core * 32, c_in],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    real_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    real_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    copy_op, copy_pd = copy_modules()
    taken_base, taken_cand = [], []
    _force_kw_fold(monkeypatch, real_pd, taken_base)
    _force_kw_fold(monkeypatch, copy_pd, taken_cand)
    # Which boundaries the copy exchanged (its writer args: [top, bottom, ...] per core).
    xchg = []
    real_args = copy_pd._kwf_xchg_writer_args
    monkeypatch.setattr(copy_pd, "_kwf_xchg_writer_args", lambda *a: xchg.append(real_args(*a)) or xchg[-1])

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

    def run(mod):
        tt_input = ttnn.from_torch(
            torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        out = mod.conv2d_3x3(
            tt_input,
            mod.prepare_conv2d_weights(torch_weight, **kwargs),
            input_height=H,
            input_width=W,
            bias=mod.prepare_conv2d_bias(torch_bias, **kwargs),
            padding=d,
            stride=1,
            dilation=d,
            activation="relu" if relu else None,
            compute_kernel_config=compute_config,
        )
        return ttnn.to_torch(out).float(), str(out.memory_config())

    base, base_mc = run(real_op)
    cand, cand_mc = run(copy_op)
    # Program-cache hit: the exchange's semaphores and tokens start from zero again.
    again, _ = run(copy_op)
    assert torch.equal(again, cand), "program-cache re-run differs"

    assert True in taken_base and True in taken_cand, f"kw-fold not taken: {taken_base} {taken_cand}"
    assert base_mc == cand_mc

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    cand = cand.reshape(ref.shape)
    base = base.reshape(ref.shape)
    pcc = float(torch.corrcoef(torch.stack([ref.flatten().double(), cand.flatten().double()]))[0, 1])
    differ = int((base != cand).sum())
    mae = float((cand - ref).abs().mean())
    tops, bots = sum(a[0] for a in xchg), sum(a[1] for a in xchg)
    print(f"differing outputs {differ} / {base.numel()}, pcc {pcc:.8f}, mae {mae:.5f}, exchanged top {tops} bot {bots}")
    if ttnn_khx_mode() == 2:
        assert tops > 0 and tops == bots, "the shape must exercise the exchange"

    assert pcc >= (0.997 if dtype == weight_dtype == ttnn.bfloat16 else 0.99)
    assert differ == 0, f"{differ} outputs differ from the recompute"
