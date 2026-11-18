# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t2_raw_kwalk_hxw isolated K-walk bench: genify_matmul's own compute kernel (the real file for `base`, this dir's
copy for the candidates) fed by a push-only reader over L1 shards holding two K-block slots (no NoC traffic) and a
pop-only writer. Epilogue none, one block per core. Default = the focus block: 16 x 5 tiles, 4 x 1 subblocks,
K = 160 as 18 K-blocks of 9 (last 7); A bf16 -> SrcB, B bfp8_b -> SrcA; HiFi2, fp32 DEST.

Env: HXWB_VARIANTS (see VARIANTS), HXWB_BLOCK="bm,bn,sbh,sbw,bk,K", HXWB_FID, HXWB_FP32, HXWB_BDTYPE, HXWB_GRID="gx,gy",
HXWB_CHECK=1 (write the output to DRAM and compare: torch.equal across variants + PCC vs torch).
"""

import os

from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).resolve().parent
REAL_COMPUTE = HERE.parents[1] / "kernels" / "genify_matmul_compute.cpp"
OWN_COMPUTE = HERE / "kernels" / "genify_matmul_compute.cpp"
TILE = 32
CB_ACT, CB_WEIGHT, CB_OUT, CB_PARTIALS = 0, 1, 16, 24
CLOCK_MHZ = 1350.0

VARIANTS = {
    "base": (False, []),  # helper matmul_block<Out> (the op today for sb_h > 1)
    "kw": (False, ["GM_NONE_RAW_WALK"]),  # the op's Perf 1 raw walk + copy (sb_h == 1 blocks only)
    "ref": (True, ["GM_HXW_REF_INTERM"]),  # helper Interm walk + identity copy
    "a": (True, ["GM_HXW", "GM_HXW_NONE_FORCE"]),  # raw h x w walk + identity copy
    "b": (True, ["GM_HXW", "GM_HXW_NONE_FORCE", "GM_HXW_STREAM"]),  # raw h x w walk, streamed copy
    "a_skip": (True, ["GM_HXW", "GM_HXW_NONE_FORCE", "SKIP_COMPUTE"]),
    "base_skip": (False, ["SKIP_COMPUTE"]),
}


def _env(name, default):
    return os.environ.get(name, default)


def _block():
    bm, bn, sbh, sbw, bk, k = (int(x) for x in _env("HXWB_BLOCK", "16,5,4,1,9,160").split(","))
    nkb = -(-k // bk)
    last = k - (nkb - 1) * bk
    return bm, bn, sbh, sbw, bk, k, nkb, last


def _grid(device):
    g = os.environ.get("HXWB_GRID")
    if g:
        return tuple(int(x) for x in g.split(","))
    cg = device.compute_with_storage_grid_size()
    return cg.x, cg.y


def _cfg():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, _env("HXWB_FID", "HiFi2")),
        fp32_dest_acc_en=_env("HXWB_FP32", "1") == "1",
        math_approx_mode=False,
    )


def _inputs(device, gx, gy):
    import torch  # local: the ttnn tree forbids a global torch import

    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    ncores = gx * gy
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    torch.manual_seed(0)
    a = torch.randn(2 * bm * TILE, bk * TILE)  # slot s = tile rows [s * bm, (s + 1) * bm)
    b = torch.randn(2 * bk * TILE, bn * TILE) / (k * TILE) ** 0.5

    def shard(t, dtype):
        h, w = t.shape
        mc = ttnn.create_sharded_memory_config(
            shape=(h, w),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.from_torch(
            t.repeat(ncores, 1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )

    return shard(a, ttnn.bfloat16), shard(b, getattr(ttnn, _env("HXWB_BDTYPE", "bfloat8_b"))), grid, ncores


def _reference(ta, tb):
    import torch  # local: the ttnn tree forbids a global torch import

    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    aq = ttnn.to_torch(ta).float()[: 2 * bm * TILE].double()
    bq = ttnn.to_torch(tb).float()[: 2 * bk * TILE].double()
    acc = torch.zeros(bm * TILE, bn * TILE, dtype=torch.float64)
    for kb in range(nkb):
        s = kb % 2
        kk = (last if kb == nkb - 1 else bk) * TILE
        acc += aq[s * bm * TILE : (s + 1) * bm * TILE, :kk] @ bq[s * bk * TILE : s * bk * TILE + kk, :]
    return acc.float()


def _descriptor(ta, tb, out, grid, gx, gy, variant, check):
    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    own, defs = VARIANTS[variant]
    p_dt, p_bytes = (ttnn.float32, 4096) if _env("HXWB_FP32", "1") == "1" else (ttnn.bfloat16, 2048)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_ACT, ta),
        ttnn.cb_descriptor_from_sharded_tensor(CB_WEIGHT, tb),
        ttnn.CBDescriptor(  # the op's p_dtype: fp32 on 32-bit DEST, else bf16
            total_size=bm * bn * p_bytes,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_PARTIALS, data_format=p_dt, page_size=p_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * sbh * bn * 2048,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.bfloat16, page_size=2048)
            ],
        ),
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "kernels" / "bench_reader.cpp"),
        core_ranges=grid,
        compile_time_args=[CB_ACT, CB_WEIGHT, bm * bk, bk * bn, nkb],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ReaderConfigDescriptor(),
    )
    wrt = ttnn.RuntimeArgs()
    for y in range(gy):
        for x in range(gx):
            wrt[x][y] = [out.buffer_address(), y * gx + x]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "kernels" / "bench_writer.cpp"),
        core_ranges=grid,
        compile_time_args=[CB_OUT, bm * bn, 2048, *ttnn.TensorAccessorArgs(out).get_compile_time_args()],
        runtime_args=wrt,
        defines=[("CHECK_OUT", "1")] if check else [],
        config=ttnn.WriterConfigDescriptor(),
    )
    crt = ttnn.RuntimeArgs()
    for y in range(gy):
        for x in range(gx):
            crt[x][y] = [1, 0xFFFFFFFF]
    compute_ct = [
        CB_ACT,
        CB_WEIGHT,
        CB_OUT,
        CB_PARTIALS,
        1,
        bm // sbh,
        bn // sbw,
        sbh,
        sbw,
        bk,
        nkb,
        last,
        bn,
        0,
        2,
        4,
        bm,
        4,
        0,
        0,
        5,
        0,
        0,
        0,
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(OWN_COMPUTE if own else REAL_COMPUTE),
        core_ranges=grid,
        compile_time_args=compute_ct,
        runtime_args=crt,
        defines=[(d, "1") for d in defs],
        config=_cfg(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    vals = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            e = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                vals.append(float(e.duration))
    return max(vals) if vals else None


def test_hxw_bench(device):
    import torch  # local: the ttnn tree forbids a global torch import

    gx, gy = _grid(device)
    ta, tb, grid, ncores = _inputs(device, gx, gy)
    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    check = _env("HXWB_CHECK", "0") == "1"
    ref = _reference(ta, tb) if check else None
    fid = _env("HXWB_FID", "HiFi2")
    phases = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}[fid]
    model_ns = bm * bn * k * 17.5 * phases / CLOCK_MHZ * 1000.0
    rows, prev = [], []
    for v in [x for x in _env("HXWB_VARIANTS", "base,ref,a,b").split(",") if x]:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([ncores * bm * TILE, bn * TILE]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.generic_op([ta, tb, out], _descriptor(ta, tb, out, grid, gx, gy, v, check))
        ttnn.synchronize_device(device)
        line = f"HXWB {v:<10}"
        if check:
            got = ttnn.to_torch(out).float().reshape(ncores, bm * TILE, bn * TILE)
            spread = (got - got[0:1]).abs().max().item()
            pcc = torch.corrcoef(torch.stack([got[0].flatten(), ref.flatten()]))[0, 1].item()
            rms = ((got[0] - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            eq = [pv for pv, po in prev if torch.equal(po, got[0])]
            line += f" pcc {pcc:.6f} rel_rms {rms:.3e} core_spread {spread:.1e} equal={eq}"
            prev.append((v, got[0]))
            if "skip" not in v and pcc <= 0.999:  # per-tile relative error map (debug)
                e = (got[0] - ref).reshape(bm, TILE, bn, TILE).abs().amax(dim=(1, 3))
                r = ref.reshape(bm, TILE, bn, TILE).abs().amax(dim=(1, 3))
                print(f"HXWB_ERRMAP {v}\n" + "\n".join(" ".join(f"{x:5.2f}" for x in row) for row in (e / r).tolist()))
            if "skip" not in v:
                assert pcc > 0.999 and spread == 0.0, line
        else:
            line += f" (kernel ns: ops CSV row {len(rows)}; model {model_ns:.0f} ns = 17.5 x {phases} cyc/tile-K)"
        rows.append(line)
        out.deallocate()
    print(
        f"\nHXWB block={bm}x{bn} sb={sbh}x{sbw} bk={bk} K={k} nkb={nkb} last={last} grid={gx}x{gy} fid={fid} "
        f"fp32={_env('HXWB_FP32', '1')} B={_env('HXWB_BDTYPE', 'bfloat8_b')}\n" + "\n".join(rows)
    )
