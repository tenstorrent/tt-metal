# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf_experiments/fold_finalize: isolated micro-bench of the owner's cross-device fold + row-0 finalize.

One core. cb_stats = G*Q resident fp32 record tiles (data in row 0, garbage elsewhere), cb_rms_root = Q finalized
tiles. Per rep the NCRISC driver pushes cb_stats ("records landed"), compute folds + finalizes + packs + pushes
cb_rms_root, the driver waits / pops (what the op's all-gather waits on). FF_LOOP (NCRISC) / REPS = latency.

    FF_VARIANTS=0,1,2,3,4,5 FF_G=4 FF_Q=1,2,3,5 FF_FP32=1 scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/fold_finalize/test_fold_bench.py -s

Correctness: row 0 of every finalized tile bit-identical to variant 0 (the op's code) and close to torch; a +inf
record lane must finalize to -0 (the sentinel contract).
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pathlib
import statistics
import struct

import ttnn
from loguru import logger

HERE = pathlib.Path(__file__).resolve().parent
KDIR = HERE / "kernels"
CB_X, CB_RMS_ROOT, CB_STATS = 1, 11, 15
NAMES = {
    0: "base",
    1: "hoist",
    2: "slots",
    3: "slots_bcast",
    4: "slots_row",
    5: "radd",
    6: "slots_row_pk",
    7: "multi",
    8: "multi_pk",
    9: "compact",
    10: "op_compact",
}


def _layout(v):
    return 2 if v in (9, 10) else (1 if v >= 7 else 0)


_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")


def _ints(name, default):
    return [int(v) for v in os.environ.get(name, default).split(",") if v]


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded(rows):
    return ttnn.create_sharded_memory_config(
        shape=(rows, 32),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _run(device, stats, out, *, G, Q, variant, reps, fp32, inv_w, eps, zones=False):
    ct = [G, Q, variant, reps, Q]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32, math_approx_mode=False
    )
    if fp32:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_STATS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "fold_compute.cpp"),
        core_ranges=_core(),
        compile_time_args=ct,
        defines=[("FF_ZONES", "1")] if zones else [],
        runtime_args=[(ttnn.CoreCoord(0, 0), [_bits(inv_w), _bits(eps)])],
        config=compute_cfg,
    )
    dm = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "fold_dm.cpp"),
        core_ranges=_core(),
        compile_time_args=[G, Q, reps, Q],
        runtime_args=[(ttnn.CoreCoord(0, 0), [])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    x_cb = ttnn.CBDescriptor(
        total_size=2048,
        core_ranges=_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_X, data_format=ttnn.bfloat16, page_size=2048)],
    )
    cbs = [
        x_cb,
        ttnn.cb_descriptor_from_sharded_tensor(CB_STATS, stats),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RMS_ROOT, out),
    ]
    prog = ttnn.ProgramDescriptor(kernels=[compute, dm], semaphores=[], cbs=cbs)
    ttnn.generic_op([stats, out], prog)


def _records(G, Q, seed):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    """G*Q tiles: row 0 = a chip partial Σx² (positive, a few inf lanes), rows 1..31 garbage."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(G * Q, 32, 32, generator=g) * 100.0
    row0 = torch.rand(G * Q, 32, generator=g) * 1e4 + torch.rand(G * Q, 32, generator=g)
    row0[0, 5] = float("inf")  # pos 0, row 0: an overflowed partial -> inv must be -0
    row0[0, 21] = 0.0
    if G * Q > 1:
        row0[1, 21] = 0.0
    t[:, 0, :] = row0
    return t.reshape(G * Q * 32, 32)


def _row0(out, Q, fp32, joined=False):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    o = ttnn.to_torch(out).reshape(Q, 32, 32)
    # joined record (row-only pack): row 0 of face 1 sits behind row 0 of face 0, i.e. in row 1 of face 0
    o = torch.cat([o[:, 0, :16], o[:, 1, :16]], dim=1) if joined else o[:, 0, :]
    return o.float().contiguous() if fp32 else o.to(torch.bfloat16).contiguous()


def _golden(rec, G, Q, inv_w, eps):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    r = rec.reshape(G, Q, 32, 32)[:, :, 0, :]  # [G, Q, 32]
    acc = r[0].clone()
    for p in range(1, G):
        acc = acc + r[p]  # fp32 position order
    return torch.rsqrt(acc * inv_w + eps)


def _all_ids():
    if not os.path.exists(_CSV):
        return set()
    with open(_CSV) as f:
        lines = f.read().splitlines()[2:]
    return {ln.split(",")[7].strip() for ln in lines if ln.strip()}


def _zone_ns(seen, zone, risc):
    with open(_CSV) as f:
        lines = f.read().splitlines()
    mhz = float(lines[0].split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    st, en = [], []
    for ln in lines[2:]:
        r = [x.strip() for x in ln.split(",")]
        if len(r) < 12 or r[7] in seen or r[10] != zone or r[3] != risc:
            continue
        (st if r[11] == "ZONE_START" else en).append(int(r[5]))
    st.sort()
    en.sort()
    return [(e - s) * 1000.0 / mhz for s, e in zip(st, en)]


def _setup(device, G, Q, fp32, seed):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    rec = _records(G, Q, seed)
    stats = ttnn.from_torch(
        rec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_sharded(G * Q * 32)
    )
    out = ttnn.from_torch(
        torch.zeros(Q * 32, 32),
        dtype=ttnn.float32 if fp32 else ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(Q * 32),
    )
    # [row][position] landing (variants 7 / 8): page r*G + pos
    rm = rec.reshape(G, Q, 32, 32).transpose(0, 1).reshape(G * Q * 32, 32).contiguous()
    stats_rm = ttnn.from_torch(
        rm, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_sharded(G * Q * 32)
    )
    # compact [row][position] landing (variant 9): the 128 B record contiguous at the page start (rows 0-1 of face 0)
    cp = torch.randn(G * Q, 32, 32) * 100.0
    r0 = rec.reshape(G, Q, 32, 32)[:, :, 0, :].transpose(0, 1).reshape(G * Q, 32)
    cp[:, 0, :16] = r0[:, :16]
    cp[:, 1, :16] = r0[:, 16:]
    stats_cp = ttnn.from_torch(
        cp.reshape(G * Q * 32, 32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(G * Q * 32),
    )
    return rec, (stats, stats_rm, stats_cp), out


def test_fold_bench(device):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    variants = _ints("FF_VARIANTS", "0,1,2,3,4,5")
    Gs = _ints("FF_G", "4")
    Qs = _ints("FF_Q", "1,2,3,5")
    fp32s = [bool(v) for v in _ints("FF_FP32", "1")]
    reps = int(os.environ.get("FF_REPS", "64"))
    zones = os.environ.get("FF_CMP_ZONES", "0") == "1"
    inv_w, eps = 1.0 / (4 * 1792), 1e-6
    table = []
    for fp32 in fp32s:
        for G in Gs:
            for Q in Qs:
                # ---- correctness (REPS=1): bit-identical to base, close to torch, -0 sentinel ----
                rec, stats2, out = _setup(device, G, Q, fp32, seed=G * 100 + Q)
                gold = _golden(rec, G, Q, inv_w, eps)
                ref = None
                for v in variants:
                    _run(device, stats2[_layout(v)], out, G=G, Q=Q, variant=v, reps=1, fp32=fp32, inv_w=inv_w, eps=eps)
                    got = _row0(out, Q, fp32, joined=(v in (6, 8, 9, 10)))
                    bits = got.view(torch.int32) if fp32 else got.view(torch.int16)
                    if ref is None:
                        ref = bits.clone()
                    ndiff = int((bits != ref).sum())
                    assert ndiff == 0, f"G={G} Q={Q} fp32={fp32} {NAMES[v]}: {ndiff} row-0 lanes differ from base"
                    g = got.float()
                    # fp32 DEST: an inf record finalizes to -0 (the sentinel). bf16 DEST: the base op itself returns +0
                    # there (measured; pre-existing) — the candidates are pinned bit-identical to base instead.
                    if fp32:
                        assert g[0, 5].item() == 0.0 and torch.signbit(g[0, 5]).item(), "inf record must give -0"
                    else:
                        g[0, 5] = -0.0
                    fin = torch.isfinite(gold) & (gold != 0)
                    rel = ((g[fin] - gold[fin]).abs() / gold[fin].abs()).max().item()
                    assert rel < (1e-3 if fp32 else 3e-2), f"{NAMES[v]} rel {rel}"
                    nz = (g == 0) & ~torch.signbit(g)
                    assert not nz.any(), "inv never +0"
                # ---- perf: one REPS run per variant ----
                for v in variants:
                    ttnn.ReadDeviceProfiler(device)
                    seen = _all_ids()
                    _run(
                        device,
                        stats2[_layout(v)],
                        out,
                        G=G,
                        Q=Q,
                        variant=v,
                        reps=reps,
                        fp32=fp32,
                        inv_w=inv_w,
                        eps=eps,
                        zones=zones,
                    )
                    ttnn.synchronize_device(device)
                    ttnn.ReadDeviceProfiler(device)
                    loop = _zone_ns(seen, "FF_LOOP", "NCRISC")
                    per = loop[0] / reps if loop else float("nan")
                    extra = ""
                    if zones:
                        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
                            f = _zone_ns(seen, "cmp_fold", risc)
                            extra += f" {risc}:fold={statistics.median(f):.0f}" if f else ""
                    table.append((fp32, G, Q, NAMES[v], per))
                    logger.info(f"FF fp32={int(fp32)} G={G} Q={Q} {NAMES[v]:12s} {per:8.1f} ns/rep{extra}")
    logger.info("FF summary (ns per fold, records-landed -> cb_rms_root visible):")
    for fp32, G, Q, name, per in table:
        logger.info(f"FF   fp32={int(fp32)} G={G} Q={Q} {name:12s} {per:8.1f}")
