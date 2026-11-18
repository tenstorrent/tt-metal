# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""kw-fold + output-shift: isolated per-core bench of the new conv2d_3x3 compute chain (bench/kwf_compute.cpp).

One core runs the per-core work of a real HEIGHT config: its slab of input tile-rows (own tile-rows + halo, zero tiles
outside the image) and the [3 C_in, 3 C_out] weights resident in L1 (sharded tensors), bias + ReLU, bf8_b output.
Correctness: PCC vs the fp32 torch conv of the same (bf8_b-quantized) values at the case's tolerance (0.996).

    scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_output_shift/test_kwf_bench.py::test_kwf_bench[res3]"
"""

import os
from pathlib import Path

import pytest

import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "bench"

# Per-core work of the focus configs (HEIGHT, one M-block per core). t0 = the core's first output tile-row: a core
# in the image interior whose range starts and ends mid image row, so both halo spills and an image-row gap occur.
CASES = {
    "res3": dict(H=64, W=128, ci=128, co=128, d=1, bm=3, t0=63),
    "center": dict(H=128, W=256, ci=128, co=32, d=1, bm=10, t0=50),
    "stem2": dict(H=256, W=512, ci=64, co=64, d=1, bm=38, t0=38),
}
# Correctness sweep: image edges, image-row-aligned ranges, dilation, wider C_out (sbw < N).
SWEEP = {
    "res3_top": dict(H=64, W=128, ci=128, co=128, d=1, bm=3, t0=0),
    "res3_bottom": dict(H=64, W=128, ci=128, co=128, d=1, bm=3, t0=253),
    "center_aligned": dict(H=128, W=256, ci=128, co=32, d=1, bm=10, t0=40),
    "stem2_last": dict(H=256, W=512, ci=64, co=64, d=1, bm=38, t0=4058),
    "d2": dict(H=32, W=64, ci=64, co=64, d=2, bm=7, t0=21),
    "d8": dict(H=32, W=64, ci=64, co=64, d=8, bm=7, t0=29),
    "d18": dict(H=32, W=64, ci=64, co=64, d=18, bm=7, t0=30),
    "co256": dict(H=64, W=128, ci=128, co=256, d=1, bm=3, t0=63),
}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded(torch_t, dtype, layout, device):
    mc = ttnn.create_sharded_memory_config(
        shape=tuple(torch_t.shape),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(torch_t, dtype=dtype, layout=layout, device=device, memory_config=mc)


def _quant_bf8(t):
    """Host bf8_b round trip (no device): the values the device will see."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def _rowbase(c, i):
    tpr = c["W"] // 32
    img0 = (c["t0"] - 1 + tpr) // tpr
    img = (c["t0"] - 1 + i + tpr) // tpr
    return c["d"] + 32 * i + c["d"] * (img - img0)


def _subblock(n_tiles, sbw=None):
    if sbw is None:
        sbw = max(w for w in range(1, 5) if n_tiles % w == 0)
    return sbw, 4 // sbw


def make_inputs(c, seed=0):
    import torch

    torch.manual_seed(seed)
    H, W, ci, co = c["H"], c["W"], c["ci"], c["co"]
    x = torch.randn(H * W, ci, dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn(co, ci, 3, 3, dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(co, dtype=torch.float32).to(torch.bfloat16)
    return x, w, b


def reference(c, x, w, b, relu=True):
    """fp32 torch conv of the given values; output rows of the core's range [32 t0, 32 (t0 + bm))."""
    import torch

    H, W, ci, co, d = c["H"], c["W"], c["ci"], c["co"], c["d"]
    xi = x.float().reshape(1, H, W, ci).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(xi, w.float(), b.float(), padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(H * W, co)
    return ref[32 * c["t0"] : 32 * (c["t0"] + c["bm"])]


def run_bench(device, c, x, w, b, *, relu=True, sbw=None, final_batch=None):
    """-> (device output [bm*32, co] fp32, quantized (x, w, b) the device used)."""
    import torch

    H, W, ci, co, d, bm, t0 = c["H"], c["W"], c["ci"], c["co"], c["d"], c["bm"], c["t0"]
    assert W % 32 == 0 and ci % 32 == 0 and co % 32 == 0 and 1 <= d <= 32
    tpr = W // 32
    R = d * tpr
    m_tiles = H * W // 32
    N, Ct = co // 32, ci // 32
    sbw, sbh = _subblock(N, sbw)

    xq = _quant_bf8(x.float())
    # Slab: tile-rows [t0 - 1 - R, t0 + bm + 1 + R), zero rows outside the image.
    r0, r1 = t0 - 1 - R, t0 + bm + 1 + R
    slab = torch.zeros((r1 - r0) * 32, ci)
    lo, hi = max(r0, 0), min(r1, m_tiles)
    slab[(lo - r0) * 32 : (hi - r0) * 32] = xq[lo * 32 : hi * 32]
    # W'[kh * ci + c, kw * co + n] = w[n, c, kh, kw]
    wp = _quant_bf8(w.float().permute(2, 1, 3, 0).reshape(3 * ci, 3 * co))
    wq = wp.reshape(3, ci, 3, co).permute(3, 1, 0, 2).contiguous()
    bias_tile = torch.zeros(32, co)
    bias_tile[0] = b.float()
    bq = _quant_bf8(bias_tile)[0]

    t_in = _sharded(slab, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device)
    t_w = _sharded(wp, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device)
    t_b = _sharded(bias_tile, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device)
    t_out = _sharded(torch.zeros(bm * 32, co), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device)

    pitch = N * 32 * 4
    acc_rows = _rowbase(c, bm + 1) + 32 + d
    acc_bytes = -(-acc_rows * pitch // 4096) * 4096
    halo_top = int(t0 % tpr != 0)
    halo_bot = int((t0 + bm) % tpr != 0)

    def cb(index, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=_core(),
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, t_in),
        ttnn.cb_descriptor_from_sharded_tensor(1, t_w),
        ttnn.cb_descriptor_from_sharded_tensor(2, t_b),
        ttnn.cb_descriptor_from_sharded_tensor(3, t_out),
        cb(4, ttnn.float32, 4096, acc_bytes // 4096),
        cb(5, ttnn.float32, N * 4096, 1),
        cb(6, ttnn.float32, 16, bm),
        cb(7, ttnn.float32, 16, 1),
    ]
    if final_batch is None:
        final_batch = int(os.environ.get("KWF_FB", "4"))
    stream = int(sbw == N and os.environ.get("KWF_STREAM", "1") == "1")
    u2d = int(os.environ.get("KWF_U2D", "1") == "1")
    # u2d is also a CT arg so the two final-read modes never share a program-cache entry.
    ct_args = [N, Ct, bm, d, tpr, sbw, sbh, int(relu), final_batch, stream, u2d]
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    if u2d:
        modes[4] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t0, halo_top, halo_bot]
    rt_r = ttnn.RuntimeArgs()
    rt_r[0][0] = [t0]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "kwf_reader.cpp"),
            core_ranges=_core(),
            compile_time_args=ct_args,
            runtime_args=rt_r,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "kwf_compute.cpp"),
            core_ranges=_core(),
            compile_time_args=ct_args,
            runtime_args=rt,
            config=cfg,
        ),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    out = ttnn.generic_op([t_in, t_w, t_b, t_out], desc)
    got = ttnn.to_torch(out).float()
    for t in (t_in, t_w, t_b, t_out):
        t.deallocate()
    return got, (xq, wq, bq)


def err_stats(got, ref):
    import torch

    diff = (got - ref).abs()
    g, r = got.flatten().double(), ref.flatten().double()
    pcc = float(torch.corrcoef(torch.stack([g, r]))[0, 1])
    return dict(
        pcc=pcc, mean_abs=float(diff.mean()), max_abs=float(diff.max()), ref_rms=float(ref.pow(2).mean().sqrt())
    )


def _check(c, got, quant, x, w, b):
    ref_q = reference(c, *quant)  # same quantized values, fp32 math: isolates accumulation / shift error
    ref = reference(c, x, w, b)  # harness-style: bf16 inputs, fp32 math
    sq, s = err_stats(got, ref_q), err_stats(got, ref)
    print(f"KWF_ERR vs_quantized_ref {sq}")
    print(f"KWF_ERR vs_bf16_ref {s}")
    assert s["pcc"] >= 0.996, s
    return sq, s


@pytest.mark.parametrize("name", list(CASES))
def test_kwf_bench(name, device):
    c = CASES[name]
    x, w, b = make_inputs(c)
    sbw = int(os.environ["KWF_SBW"]) if os.environ.get("KWF_SBW") else None
    got, quant = run_bench(device, c, x, w, b, sbw=sbw)
    _check(c, got, quant, x, w, b)


@pytest.mark.parametrize("name", list(SWEEP))
def test_kwf_sweep(name, device):
    c = SWEEP[name]
    x, w, b = make_inputs(c, seed=1)
    got, quant = run_bench(device, c, x, w, b)
    _check(c, got, quant, x, w, b)


# Focus-case ids of the real op's perf harness for the bench cases.
_FOCUS_ID = {"res3": "res3_{1,2,3}_conv2", "center": "center_head_1", "stem2": "stem_conv2"}


@pytest.mark.parametrize("name", list(CASES))
def test_kwf_precision(name, device):
    """Error vs the fp32 torch conv (bf16 inputs) on the bench core's output rows: the real conv2d_3x3 (full focus
    config, same inputs) next to the bench chain (srcA tf32 final read, and lossless unpack-to-dest final read)."""
    import importlib

    import torch

    from eval.golden_tests.conv2d_3x3 import helpers

    perf = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf")
    case = perf._FOCUS_CASES[perf._FOCUS_IDS.index(_FOCUS_ID[name])]
    c = CASES[name]
    (H, W, ci), co, p, s, d = case["inputs"]
    assert (H, W, ci, co, d) == (c["H"], c["W"], c["ci"], c["co"], c["d"])
    x, w, b = make_inputs(c, seed=2)
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, ci),
        case["memory_layout"],
        layout=case["layout"],
        dtype=case["dtype"],
        device=device,
        extras=case["extras"],
    )
    from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

    kw = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=mc,
        device=device,
    )
    tt_w = prepare_conv2d_weights(w, **kw)
    tt_b = prepare_conv2d_bias(b, **kw)
    tt_x = ttnn.from_torch(
        x.reshape(1, 1, H * W, ci), dtype=case["dtype"], layout=case["layout"], device=device, memory_config=mc
    )
    out = conv2d_3x3(
        tt_x,
        tt_w,
        input_height=H,
        input_width=W,
        bias=tt_b,
        padding=d,
        dilation=d,
        activation="relu",
        compute_kernel_config=helpers.make_compute_kernel_config(True, ttnn.MathFidelity.LoFi),
    )
    op_rows = ttnn.to_torch(out).float().reshape(-1, co)[32 * c["t0"] : 32 * (c["t0"] + c["bm"])]
    ref = reference(c, x, w, b)
    print(f"KWF_PREC {name} current_op {err_stats(op_rows, ref)}")
    for u2d in ("0", "1"):
        os.environ["KWF_U2D"] = u2d
        got, _ = run_bench(device, c, x, w, b)
        print(f"KWF_PREC {name} kwfold_{'u2d_fp32' if u2d == '1' else 'srca_tf32'} {err_stats(got, ref)}")
    os.environ.pop("KWF_U2D")
