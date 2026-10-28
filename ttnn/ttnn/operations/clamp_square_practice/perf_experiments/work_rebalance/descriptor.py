# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""work_rebalance experiment: descriptors for the work-distribution variants of clamp_square_practice.

  baseline        - the op's own descriptor (contiguous equal ranges, split_work_to_cores row_wise)
  cores_<GXxGY>   - baseline kernels on a smaller core grid (fewer active cores)
  strided         - static block-cyclic ownership: core c owns blocks c, c+N, ...
  dyn_k<K>        - dynamic claiming: reader fetch-and-adds a shared counter, K blocks per claim
"""
import os
from pathlib import Path

import ttnn

from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

KDIR = Path(__file__).parent / "kernels"
CB_IN, CB_OUT, CB_IDX = 0, 1, 2
IDX_PAGE = 16
IDX_DEPTH = 8
_NONCE = 0
NUM_SEMS = 4  # counter, published, done, atomic-return


def _block_tiles(total_tiles, grid_cores):
    lim = pd.dest_limit(pd.FP32_DEST_ACC_EN, pd.DST_FULL_SYNC_EN)
    per_core = -(-total_tiles // grid_cores)
    return max(1, min(pd.BLOCK_TILES_DEFAULT, lim, per_core))


def _common(x, out, core_grid, bt_override=None):
    device = x.device()
    if core_grid is None:
        core_grid = device.compute_with_storage_grid_size()
    T = x.buffer_num_pages()
    bt = _block_tiles(T, core_grid.x * core_grid.y)
    if bt_override:
        bt = min(bt, bt_override)
    nb = -(-T // bt)
    (ncores, all_cores, *_rest) = ttnn.split_work_to_cores(core_grid, nb, row_wise=True)
    cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
    cbs = [
        ttnn.CBDescriptor(
            total_size=pd.INPUT_DEPTH_BLOCKS * bt * x.buffer_page_size(),
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=x.dtype, page_size=x.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=pd.OUTPUT_DEPTH_BLOCKS * bt * out.buffer_page_size(),
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=out.dtype, page_size=out.buffer_page_size())
            ],
        ),
    ]
    return device, T, bt, nb, all_cores, cores, cbs


def _compute_cfg():
    return ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=pd.FP32_DEST_ACC_EN, dst_full_sync_en=pd.DST_FULL_SYNC_EN)


def strided(x, out, min_val, max_val, core_grid=None):
    device, T, bt, nb, all_cores, cores, cbs = _common(x, out, core_grid)
    N = len(cores)
    mn, mx = pd._fp32_bits(min_val), pd._fp32_bits(max_val)
    r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        owned = len(range(i, nb, N))
        r_rt[c.x][c.y] = [x.buffer_address(), i, N, nb, T]
        w_rt[c.x][c.y] = [out.buffer_address(), i, N, nb, T]
        c_rt[c.x][c.y] = [owned * bt, mn, mx]
    r_ct = [CB_IN, bt, x.buffer_page_size()] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    w_ct = [CB_OUT, bt, out.buffer_page_size()] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "reader_strided.cpp"),
            core_ranges=all_cores,
            compile_time_args=r_ct,
            runtime_args=r_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "writer_strided.cpp"),
            core_ranges=all_cores,
            compile_time_args=w_ct,
            runtime_args=w_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(pd.KERNEL_DIR / "clamp_square_practice_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=[CB_IN, CB_OUT, bt],
            runtime_args=c_rt,
            config=_compute_cfg(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs), {"cores": N, "bt": bt, "nb": nb}


def dynamic(
    x,
    out,
    min_val,
    max_val,
    claim_blocks=1,
    core_grid=None,
    stats=None,
    counter="center",
    jit=False,
    bt_override=None,
    guided=0,
):
    device, T, bt, nb, all_cores, cores, cbs = _common(x, out, core_grid, bt_override)
    N = len(cores)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=IDX_DEPTH * IDX_PAGE,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_IDX, data_format=ttnn.uint32, page_size=IDX_PAGE)
            ],
        )
    )
    ccore = cores[N // 2] if counter == "center" else cores[0]
    cphys = device.worker_core_from_logical_core(ccore)
    mn, mx = pd._fp32_bits(min_val), pd._fp32_bits(max_val)
    r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    stats_addr = stats.buffer_address() if stats is not None else 0
    G = device.compute_with_storage_grid_size()
    G = G.x * G.y
    if stats is not None:
        for cb_index in (4, 5):
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=1024,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=ttnn.uint32, page_size=1024)
                    ],
                )
            )
    global _NONCE
    _NONCE = (_NONCE + 0x10000) & 0xFFFFFFFF  # per-launch tag base; consecutive launches never share tags
    for i, c in enumerate(cores):
        r_rt[c.x][c.y] = [x.buffer_address(), cphys.x, cphys.y, nb, T, stats_addr, i, max(1, guided * N), _NONCE]
        w_rt[c.x][c.y] = [out.buffer_address(), T, stats_addr, G + i, _NONCE]
        c_rt[c.x][c.y] = [mn, mx]
    r_ct = [CB_IN, CB_IDX, bt, x.buffer_page_size(), claim_blocks] + list(
        ttnn.TensorAccessorArgs(x).get_compile_time_args()
    )
    r_defines = [("CLAIM_JIT", "1")] if jit else []
    if os.environ.get("WR_UNIQUE_RET") == "1":
        r_defines.append(("UNIQUE_RET", "1"))
        cbs.append(
            ttnn.CBDescriptor(
                total_size=16 * G,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=6, data_format=ttnn.uint32, page_size=16 * G)],
            )
        )
    if os.environ.get("WR_ATOMIC_VC"):  # experiment knob: VC for the claim atomic
        r_defines.append(("ATOMIC_VC", os.environ["WR_ATOMIC_VC"]))
    if guided:
        r_defines.append(("CLAIM_GUIDED", "1"))
    if stats is not None:
        r_ct += list(ttnn.TensorAccessorArgs(stats).get_compile_time_args())
        r_defines.append(("WR_STATS", "1"))
    w_ct = [CB_OUT, CB_IDX, bt, out.buffer_page_size()] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    if stats is not None:
        w_ct += list(ttnn.TensorAccessorArgs(stats).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "reader_dyn.cpp"),
            core_ranges=all_cores,
            compile_time_args=r_ct,
            defines=r_defines,
            runtime_args=r_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "writer_dyn.cpp"),
            core_ranges=all_cores,
            compile_time_args=w_ct,
            runtime_args=w_rt,
            defines=[("WR_STATS", "1")] if stats is not None else [],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute_dyn.cpp"),
            core_ranges=all_cores,
            compile_time_args=[CB_IN, CB_OUT, bt],
            runtime_args=c_rt,
            config=_compute_cfg(),
        ),
    ]
    # SEM_RET (3) starts at 0xFFFFFFFF: no claim can return it, so the reader detects the first return.
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=all_cores, initial_value=(0xFFFFFFFF if i == 3 else 0))
        for i in range(NUM_SEMS)
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), {"cores": N, "bt": bt, "nb": nb}


def build(variant, x, out, min_val, max_val, stats=None):
    """Return (ProgramDescriptor, info dict) for a variant name."""
    if variant == "baseline":
        return pd.create_program_descriptor(x, out, min_val, max_val), {}
    if variant.startswith("cores_"):
        gx, gy = (int(v) for v in variant[len("cores_") :].split("x"))
        return pd.create_program_descriptor(x, out, min_val, max_val, core_grid=ttnn.CoreCoord(gx, gy)), {
            "grid": (gx, gy)
        }
    if variant == "strided":
        return strided(x, out, min_val, max_val)
    if variant == "skew":
        return skew(x, out, min_val, max_val)
    if variant.startswith("guided_f"):
        # guided_f<F>: tile-granular guided self-scheduling, divisor = F * num_cores
        return dynamic(x, out, min_val, max_val, stats=stats, guided=int(variant[len("guided_f") :]))
    if variant.startswith("dyn_k"):
        rest = variant[len("dyn_k") :]
        grid = None
        jit = rest.endswith("_jit")
        rest = rest[: -len("_jit")] if jit else rest
        bt_override = None
        if "_g" in rest:
            rest, g = rest.split("_g")
            gx, gy = (int(v) for v in g.split("x"))
            grid = ttnn.CoreCoord(gx, gy)
        if "_b" in rest:
            rest, b = rest.split("_b")
            bt_override = int(b)
        return dynamic(
            x,
            out,
            min_val,
            max_val,
            claim_blocks=int(rest),
            core_grid=grid,
            stats=stats,
            jit=jit,
            bt_override=bt_override,
        )
    raise ValueError(variant)


def skew(x, out, min_val, max_val, mode="id"):
    """Baseline descriptor (same split, same CBs, same compute) with reader/writer swapped for the
    rotated in-block order. mode: 'id' rot = core linear id; 'half' rot = id*(bt/2)... kept simple."""
    desc = pd.create_program_descriptor(x, out, min_val, max_val)
    rk, wk, ck = desc.kernels
    rk.kernel_source = str(KDIR / "reader_skew.cpp")
    wk.kernel_source = str(KDIR / "writer_skew.cpp")
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    T = x.buffer_num_pages()
    (_n, all_cores, *_r) = ttnn.split_work_to_cores(grid, T, row_wise=True)
    rr, wr_ = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(ttnn.corerange_to_cores(all_cores, row_wise=True)):
        ra = list(rk.runtime_args[c.x][c.y])
        wa = list(wk.runtime_args[c.x][c.y])
        rr[c.x][c.y] = ra + [i]
        wr_[c.x][c.y] = wa + [i]
    rk.runtime_args = rr
    wk.runtime_args = wr_
    desc.kernels = [rk, wk, ck]
    return desc, {"cores": _n}
