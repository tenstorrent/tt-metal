# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""matmul_kblock_cost compute-only bench (single core, exchange removed).

One K-block = bm x bk bf16 in0 tiles x bk x bn bf8b in1 tiles, NKB K-blocks accumulated into a fp32
partials block (packer L1 acc), LoFi + fp32 DEST (the conv2d_nhwc focus-case precision contract).
The producer only cycles the CBs over D resident slots, so out = sum_j A[j % D] @ W[j % D].
Env MK_VARIANTS = comma list of variant names (default: all), MK_SHAPE = bm,bk,bn,nkb (default 13,3,2,60).
Run: scripts/run_safe_pytest.sh --profile <this file>
"""
import os
from pathlib import Path

import importlib

import pytest
import ttnn

torch = importlib.import_module("torch")  # lazy alias: no global torch import under ttnn/

HERE = Path(__file__).parent
D = int(os.environ.get("MK_D", "4"))  # resident slots (ring depth); KACC=2 needs >= 2*KACC to keep producer slack
CB_IN0, CB_IN1, CB_OUT = 0, 1, 16

VARIANTS = {
    "helper": ["HELPER"],
    "raw_1x2_helperlike": ["INIT_KB", "PCFG_SB"],  # same LLK sequence as the helper
    "raw_1x2_hoist": ["INIT_KB"],  # pack reconfig / l1_acc once per K-block
    "raw_1x2_noinit": [],  # + no per-K-block reconfig/init
    "raw_2x2_tailinit": ["SBH=2", "INIT_KB"],
    "raw_2x2_tailover": ["SBH=2", "INIT_KB", "TAIL_OVER"],
    "raw_1x2_kacc2": ["KACC=2", "INIT_KB"],
    "raw_2x2_kacc2": ["SBH=2", "KACC=2", "INIT_KB"],
    "raw_2x2_kacc2_over": ["SBH=2", "KACC=2", "INIT_KB", "TAIL_OVER"],
    "raw_4x1": ["SBH=4", "SBW=1", "INIT_KB"],
    "raw_4x1_kacc2": ["SBH=4", "SBW=1", "KACC=2", "INIT_KB"],
    "raw_1x2_kacc2_nounroll": ["KACC=2", "INIT_KB", "NOUNROLL"],
    "raw_1x2_hoist_nounroll": ["INIT_KB", "NOUNROLL"],
    "diag_4x1_nopack": ["SBH=4", "SBW=1", "INIT_KB", "NO_PACK"],
    "raw_4x1_kacc2_nounroll": ["SBH=4", "SBW=1", "KACC=2", "INIT_KB", "NOUNROLL"],
    "raw_2x2_kacc2_nounroll": ["SBH=2", "KACC=2", "INIT_KB", "NOUNROLL"],
    "raw_1x2_kacc2_nounroll_noinit": ["KACC=2", "NOUNROLL"],
    "diag_1x2_kacc2_nounroll_nopack": ["KACC=2", "INIT_KB", "NOUNROLL", "NO_PACK"],
    "diag_1x2_kacc2_nounroll_nomath": ["KACC=2", "INIT_KB", "NOUNROLL", "NO_MATH"],
    "diag_h1x2_nopack": ["INIT_KB", "NO_PACK"],
    "diag_h1x2_nomath": ["INIT_KB", "NO_MATH"],
    "diag_h2x2_nopack": ["SBH=2", "INIT_KB", "TAIL_OVER", "NO_PACK"],
    "diag_hk2_1x2_nopack": ["KACC=2", "INIT_KB", "NO_PACK"],
    "diag_hk2_1x2_nomath": ["KACC=2", "INIT_KB", "NO_MATH"],
    "diag_1x2_nopack": ["INIT_KB", "PCFG_SB", "NO_PACK"],
    "diag_1x2_nomath": ["INIT_KB", "PCFG_SB", "NO_MATH"],
}
SEL = [v for v in os.environ.get("MK_VARIANTS", ",".join(VARIANTS)).split(",") if v]
BM, BK, BN, NKB = (int(x) for x in os.environ.get("MK_SHAPE", "13,3,2,60").split(","))


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _mc(rows):
    return ttnn.create_sharded_memory_config(
        shape=(rows, 32),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _to_stack(mats, rt, ct):
    """list of (rt*32, ct*32) matrices -> (n*rt*ct*32, 32) tile stack, tile order [slot][r][c]."""
    tiles = []
    for m in mats:
        for r in range(rt):
            for c in range(ct):
                tiles.append(m[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32])
    return torch.cat(tiles, 0)


def _from_stack(st, n, rt, ct):
    t = st.reshape(n, rt, ct, 32, 32).permute(0, 1, 3, 2, 4).reshape(n, rt * 32, ct * 32)
    return list(t)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("variant", SEL)
def test_kblock(device, variant):
    torch.manual_seed(0)
    A = [torch.randn(BM * 32, BK * 32).bfloat16().float() for _ in range(D)]
    W = [torch.randn(BK * 32, BN * 32) for _ in range(D)]
    a_st = _to_stack(A, BM, BK)
    w_st = _to_stack(W, BK, BN)
    # TAIL_OVER: the extra tail row reads up to bk tiles past the last slot (unused DEST rows, never packed)
    a_dev = ttnn.from_torch(
        a_st, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mc(a_st.shape[0])
    )
    w_dev = ttnn.from_torch(
        w_st, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mc(w_st.shape[0])
    )
    Wq = _from_stack(ttnn.to_torch(w_dev).float(), D, BK, BN)
    ref = sum(A[j % D] @ Wq[j % D] for j in range(NKB))
    out_rows = BM * BN * 32
    out_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape([out_rows, 32]), ttnn.float32, ttnn.TILE_LAYOUT, device, _mc(out_rows)
    )
    defines = [(d.split("=")[0], d.split("=")[1] if "=" in d else "1") for d in VARIANTS[variant]]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "bench_compute.cpp"),
        core_ranges=_core(),
        compile_time_args=[CB_IN0, CB_IN1, CB_OUT, BM, BK, BN, NKB],
        defines=defines,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
    )
    producer = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "bench_producer.cpp"),
        core_ranges=_core(),
        compile_time_args=[CB_IN0, CB_IN1, BM * BK, BK * BN, NKB],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN0, a_dev),
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN1, w_dev),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out_dev),
    ]
    desc = ttnn.ProgramDescriptor(kernels=[compute, producer], semaphores=[], cbs=cbs)
    ttnn.generic_op([a_dev, w_dev, out_dev], desc)
    got = _from_stack(ttnn.to_torch(out_dev).float(), 1, BM, BN)[0]
    p = pcc(got, ref)
    rel = ((got - ref).pow(2).mean().sqrt() / ref.std()).item()
    print(f"MK variant={variant} shape={BM},{BK},{BN},{NKB} PRECISION pcc={p:.6f} rel_rms={rel:.5f}")
    if not variant.startswith("diag"):
        assert p >= 0.995 and rel <= 0.05, f"pcc={p} rel={rel}"
