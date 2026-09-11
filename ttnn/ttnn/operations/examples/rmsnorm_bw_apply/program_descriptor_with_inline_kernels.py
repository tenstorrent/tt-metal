# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Distributed-RMSNorm backward `apply` step as ONE all-SFPU eltwise chain (generic op).

For y = gamma * x / rms the input gradient is

    dx = gamma * dy / rms  -  x * E[x * g] / rms^2        with g = gamma * dy / rms

PR #54829 computes this on the host as a composite of ~6 full-size eltwise ops
(multiply, divide, square, divide, multiply, subtract), each a separate program launch
that round-trips an fp32 intermediate through DRAM. This example computes the same value
in a single reader -> compute -> writer program where the compute kernel is one
`eltwise_chain` per tile-row:

    D0 = dy                               CopyTile          (unpack-to-dest, lossless fp32)
    D1 = gamma  (row 0 broadcast down)    UnaryBcast<Row>   (unpack-to-dest, lossless fp32)
    D0 = D0 * D1                          MulBinary         (SFPU, fp32)
    D1 = 1/rms  (col 0 broadcast across)  UnaryBcast<Col>
    D0 = D0 * D1                          MulBinary         -> g
    D1 = E[x*g]/rms^2 (col bcast)         UnaryBcast<Col>
    D2 = x                                CopyTile
    D0 = D0 + (-1) * D2 * D1              Addcmul<Float32>  (SFPU, fp32) -> dx
    pack D0

No FPU element is used, so with fp32_dest_acc_en every intermediate stays fp32 in DEST.

Inputs (all fp32, TILE, interleaved DRAM, rank 4):
    x, dy       [N, C, H, W]
    gamma       [1, 1, 1, W]
    inv_rms, d  [N, C, H, 1]   column stats, value in column 0 (host-side tiny ops:
                               inv_rms = rsqrt(E[x^2] + eps), d = E[x*g] * inv_rms^2)
Output: dx [N, C, H, W] fp32.

Work split: tile-rows (N*C*Ht of them) are assigned contiguously to cores.
"""

import struct

import ttnn

TILE = 32

CB_DY = 0
CB_X = 1
CB_GAMMA = 2
CB_INV_RMS = 3
CB_D = 4
CB_OUT = 16
CB_ACC = 17  # dgamma partial accumulator (compute-private until pushed to the writer)

NEG_ONE_BITS = struct.unpack("<I", struct.pack("<f", -1.0))[0]

# ---------------------------------------------------------------------------------------------
# Reader (NCRISC / NoC0): gamma once, then per tile-row: the two column stats + Wt dy + Wt x.
# ---------------------------------------------------------------------------------------------
_READER_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_dy = 0, cb_x = 1, cb_gamma = 2, cb_inv = 3, cb_d = 4;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr auto dy_args = TensorAccessorArgs<2>();
    constexpr auto x_args = TensorAccessorArgs<dy_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto inv_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<inv_args.next_compile_time_args_offset()>();

    const uint32_t dy_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t inv_addr = get_arg_val<uint32_t>(3);
    const uint32_t d_addr = get_arg_val<uint32_t>(4);
    const uint32_t row_start = get_arg_val<uint32_t>(5);
    const uint32_t row_count = get_arg_val<uint32_t>(6);

    const auto dy_acc = TensorAccessor(dy_args, dy_addr, page);
    const auto x_acc = TensorAccessor(x_args, x_addr, page);
    const auto g_acc = TensorAccessor(g_args, g_addr, page);
    const auto inv_acc = TensorAccessor(inv_args, inv_addr, page);
    const auto d_acc = TensorAccessor(d_args, d_addr, page);

    // gamma: Wt tiles, read once, never popped by compute.
    cb_reserve_back(cb_gamma, Wt);
    {
        const uint32_t l1 = get_write_ptr(cb_gamma);
#ifndef RB_STUB_DM
        for (uint32_t c = 0; c < Wt; ++c) {
            noc_async_read(g_acc.get_noc_addr(c), l1 + c * page, page);
        }
#endif
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, Wt);

    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        cb_reserve_back(cb_inv, 1);
        cb_reserve_back(cb_d, 1);
#ifndef RB_STUB_DM
        noc_async_read(inv_acc.get_noc_addr(r), get_write_ptr(cb_inv), page);
        noc_async_read(d_acc.get_noc_addr(r), get_write_ptr(cb_d), page);
#endif

        cb_reserve_back(cb_dy, Wt);
        cb_reserve_back(cb_x, Wt);
        const uint32_t dy_l1 = get_write_ptr(cb_dy);
        const uint32_t x_l1 = get_write_ptr(cb_x);
        const uint32_t base = r * Wt;
#ifndef RB_STUB_DM
        for (uint32_t c = 0; c < Wt; ++c) {
            noc_async_read(dy_acc.get_noc_addr(base + c), dy_l1 + c * page, page);
            noc_async_read(x_acc.get_noc_addr(base + c), x_l1 + c * page, page);
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_inv, 1);
        cb_push_back(cb_d, 1);
        cb_push_back(cb_dy, Wt);
        cb_push_back(cb_x, Wt);
    }
}
"""

# ---------------------------------------------------------------------------------------------
# Compute: one eltwise_chain per tile-row, all arithmetic on the SFPU.
# ---------------------------------------------------------------------------------------------
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"

void kernel_main() {
    constexpr uint32_t cb_dy = 0, cb_x = 1, cb_gamma = 2, cb_inv = 3, cb_d = 4, cb_out = 16, cb_acc = 17;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t neg_one_bits = get_compile_time_arg_val(1);
    constexpr uint32_t with_dgamma = get_compile_time_arg_val(4);
    // DEBUG (compile-time arg 5): dgamma chain mode. 0 = real (L1-accumulating strided pack),
    // 1 = second chain skipped entirely (just release the CBs), 2 = second chain runs but packs
    // plainly (walk, no accumulation) so the cost of the accumulating pack can be isolated.
    constexpr uint32_t dg_mode = get_compile_time_arg_val(5);
    const uint32_t row_count = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup(cb_dy, cb_gamma, cb_out);

    // dgamma partial: a persistent [1 x Wt] tile-row in L1 that every row's dy*x/rms tiles are
    // pack-accumulated into (column c -> tile c). Reserved once, zeroed once, pushed once at the end.
    if constexpr (with_dgamma) {
        cb_reserve_back(cb_acc, Wt);
        eltwise_chain(
            IterationShape::row(Wt),
            FillScalar<Dst::D0>{0.0f},
            PackTile<output(cb_acc, ReservePolicy::None, PushPolicy::None), Dst::D0>{});
    }
    // With dgamma, the dx chain must leave dy / x / inv_rms in their CBs for the second chain.
    constexpr PopPolicy pop_shared = with_dgamma ? PopPolicy::None : PopPolicy::AtEnd;

    constexpr auto stream = [](uint32_t cb) {
        return input(cb, WaitPolicy::Upfront, pop_shared, OperandKind::Block);
    };
    constexpr auto stream_last = [](uint32_t cb) {
        return input(cb, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Block);
    };
    constexpr auto held_col_last = [](uint32_t cb) {
        return input(cb, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col);
    };
    // Held for the whole row: gamma is never popped (read once per launch); the two column
    // stats are one tile per row, popped when the row is done.
    constexpr auto held_row = input(cb_gamma, WaitPolicy::Upfront, PopPolicy::None, OperandKind::Row);
    constexpr auto held_col = [](uint32_t cb) {
        return input(cb, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col);
    };

    // DEBUG bisection knob (compile-time arg 2): 0 = dy only, 1 = dy*gamma, 2 = dy*gamma/rms,
    // 3 = full dx (default). Knob 3 (compile-time arg 3): 1 = distinct DEST slots (no D1 reuse).
    constexpr uint32_t stage = get_compile_time_arg_val(2);
    constexpr uint32_t distinct = get_compile_time_arg_val(3);
    constexpr Dst S_INV = distinct ? Dst::D3 : Dst::D1;
    constexpr Dst S_D = distinct ? Dst::D4 : Dst::D1;

    for (uint32_t r = 0; r < row_count; ++r) {
        if constexpr (stage == 0) {
            eltwise_chain(
                IterationShape::row(Wt),
                CopyTile<stream(cb_dy), Dst::D0>{},
                PackTile<output(cb_out, ReservePolicy::PerTile, PushPolicy::PerTile), Dst::D0>{});
            cb_wait_front(cb_inv, 1); cb_pop_front(cb_inv, 1);
            cb_wait_front(cb_d, 1); cb_pop_front(cb_d, 1);
            cb_wait_front(cb_x, Wt); cb_pop_front(cb_x, Wt);
        } else if constexpr (stage == 1) {
            eltwise_chain(
                IterationShape::row(Wt),
                CopyTile<stream(cb_dy), Dst::D0>{},
                UnaryBcast<BroadcastDim::Row, held_row, Dst::D1>{},
                MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<output(cb_out, ReservePolicy::PerTile, PushPolicy::PerTile), Dst::D0>{});
            cb_wait_front(cb_inv, 1); cb_pop_front(cb_inv, 1);
            cb_wait_front(cb_d, 1); cb_pop_front(cb_d, 1);
            cb_wait_front(cb_x, Wt); cb_pop_front(cb_x, Wt);
        } else if constexpr (stage == 2) {
            eltwise_chain(
                IterationShape::row(Wt),
                CopyTile<stream(cb_dy), Dst::D0>{},
                UnaryBcast<BroadcastDim::Row, held_row, Dst::D1>{},
                MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                UnaryBcast<BroadcastDim::Col, held_col(cb_inv), S_INV>{},
                MulBinary<Dst::D0, S_INV, Dst::D0>{},
                PackTile<output(cb_out, ReservePolicy::PerTile, PushPolicy::PerTile), Dst::D0>{});
            cb_wait_front(cb_d, 1); cb_pop_front(cb_d, 1);
            cb_wait_front(cb_x, Wt); cb_pop_front(cb_x, Wt);
        } else {
            eltwise_chain(
                IterationShape::row(Wt),
                CopyTile<stream(cb_dy), Dst::D0>{},                                     // D0 = dy
                UnaryBcast<BroadcastDim::Row, held_row, Dst::D1>{},                     // D1 = gamma
                MulBinary<Dst::D0, Dst::D1, Dst::D0>{},                                 // D0 = dy*gamma
                UnaryBcast<BroadcastDim::Col, input(cb_inv, WaitPolicy::Upfront, pop_shared, OperandKind::Col), S_INV>{},  // = 1/rms
                MulBinary<Dst::D0, S_INV, Dst::D0>{},                                   // D0 = g
                UnaryBcast<BroadcastDim::Col, held_col(cb_d), S_D>{},                   // = E[x*g]/rms^2
                CopyTile<stream(cb_x), Dst::D2>{},                                      // D2 = x
                Addcmul<DataFormat::Float32, Dst::D0, Dst::D2, S_D, Dst::D0>{neg_one_bits},  // D0 = g - x*d
                PackTile<output(cb_out, ReservePolicy::PerTile, PushPolicy::PerTile), Dst::D0>{});

            if constexpr (with_dgamma && dg_mode == 1) {
                cb_wait_front(cb_dy, Wt); cb_pop_front(cb_dy, Wt);
                cb_wait_front(cb_x, Wt); cb_pop_front(cb_x, Wt);
                cb_wait_front(cb_inv, 1); cb_pop_front(cb_inv, 1);
            } else if constexpr (with_dgamma && dg_mode == 2) {
                eltwise_chain(
                    IterationShape::row(Wt),
                    CopyTile<stream_last(cb_dy), Dst::D0>{},
                    UnaryBcast<BroadcastDim::Col, held_col_last(cb_inv), Dst::D1>{},
                    MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    CopyTile<stream_last(cb_x), Dst::D2>{},
                    MulBinary<Dst::D0, Dst::D2, Dst::D0>{},
                    PackTile<output(cb_acc, ReservePolicy::None, PushPolicy::None), Dst::D0>{});
            } else if constexpr (with_dgamma) {
                // dgamma integrand dy * x / rms, accumulated per column into cb_acc tile c.
                // L1 accumulation pins the pack, so TileOffset::Strided supplies the column index.
                eltwise_chain(
                    IterationShape::row(Wt),
                    CopyTile<stream_last(cb_dy), Dst::D0>{},                                // D0 = dy
                    UnaryBcast<BroadcastDim::Col, held_col_last(cb_inv), Dst::D1>{},        // D1 = 1/rms
                    MulBinary<Dst::D0, Dst::D1, Dst::D0>{},                                 // D0 = dy/rms
                    CopyTile<stream_last(cb_x), Dst::D2>{},                                 // D2 = x
                    MulBinary<Dst::D0, Dst::D2, Dst::D0>{},                                 // D0 = dy*x/rms
                    PackTile<
                        output(cb_acc, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled,
                               PackRelu::Disabled, L1Accumulation::AddToExisting, DestAccumulation::Disabled,
                               TileOffset::Strided),
                        Dst::D0>{StridedTileRange{0, Wt}});
            }
        }
    }
    if constexpr (with_dgamma) {
        cb_push_back(cb_acc, Wt);
    }
}
"""

# ---------------------------------------------------------------------------------------------
# Writer (BRISC / NoC1): per tile-row, drain Wt output tiles to DRAM.
# ---------------------------------------------------------------------------------------------
_WRITER_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16, cb_acc = 17;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr uint32_t with_dgamma = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    constexpr auto part_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);
    const uint32_t part_addr = get_arg_val<uint32_t>(3);
    const uint32_t core_idx = get_arg_val<uint32_t>(4);

    const auto out_acc = TensorAccessor(out_args, out_addr, page);
    const auto part_acc = TensorAccessor(part_args, part_addr, page);

    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        cb_wait_front(cb_out, Wt);
        const uint32_t l1 = get_read_ptr(cb_out);
        const uint32_t base = r * Wt;
#ifndef RB_STUB_DM
        for (uint32_t c = 0; c < Wt; ++c) {
            noc_async_write(l1 + c * page, out_acc.get_noc_addr(base + c), page);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
    }
    if constexpr (with_dgamma) {
        // this core's dgamma partial: one [32 x W] tile-row at tile-row core_idx of the partial tensor
        cb_wait_front(cb_acc, Wt);
        const uint32_t l1 = get_read_ptr(cb_acc);
#ifndef RB_STUB_DM
        for (uint32_t c = 0; c < Wt; ++c) {
            noc_async_write(l1 + c * page, part_acc.get_noc_addr(core_idx * Wt + c), page);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_acc, Wt);
    }
}
"""


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_rows(num_rows, n):
    base, rem = divmod(num_rows, n)
    out, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        out.append((start, count))
        start += count
    return out


def _cb(cb_id, tiles, page, core_ranges):
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.float32, page_size=page)],
    )


def validate(x, dy, gamma, inv_rms, d):
    for name, t in (("x", x), ("dy", dy), ("gamma", gamma), ("inv_rms", inv_rms), ("d", d)):
        if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"rmsnorm_bw_apply: {name} must be float32 TILE_LAYOUT")
        if len(t.shape) != 4:
            raise ValueError(f"rmsnorm_bw_apply: {name} must be rank 4")
        if t.is_sharded():
            raise ValueError(f"rmsnorm_bw_apply: {name} must be interleaved")
    n, c, h, w = x.shape
    if list(dy.shape) != [n, c, h, w]:
        raise ValueError("rmsnorm_bw_apply: x and dy shapes must match")
    if h % TILE or w % TILE:
        raise ValueError("rmsnorm_bw_apply: H and W must be tile aligned")
    if list(gamma.shape) != [1, 1, 1, w]:
        raise ValueError("rmsnorm_bw_apply: gamma must be [1,1,1,W]")
    for name, t in (("inv_rms", inv_rms), ("d", d)):
        if list(t.shape)[:3] != [n, c, h] or t.shape[3] > TILE:
            raise ValueError(f"rmsnorm_bw_apply: {name} must be [N,C,H,<=32] (one tile column)")


def create_program_descriptor(
    x,
    dy,
    gamma,
    inv_rms,
    d,
    out,
    partial=None,
    *,
    num_cores=None,
    stage=3,
    distinct_slots=False,
    unpack_to_dest=True,
    full_sync=False,
    skip_compute=False,
    stub_dm=False,
    dg_mode=0,
):
    device = x.device()
    n, c, h, w = x.shape
    Wt = w // TILE
    num_rows = n * c * (h // TILE)
    page = x.buffer_aligned_page_size()
    assert page == 4096, f"expected 4 KiB fp32 tile pages, got {page}"

    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = min(num_cores or max_cores, max_cores, num_rows)
    cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(num_cores)]
    core_ranges = _core_range_set(cores)
    assignment = _assign_rows(num_rows, num_cores)

    cbs = [
        _cb(CB_DY, 2 * Wt, page, core_ranges),  # double-buffered row
        _cb(CB_X, 2 * Wt, page, core_ranges),
        _cb(CB_GAMMA, Wt, page, core_ranges),
        _cb(CB_INV_RMS, 2, page, core_ranges),
        _cb(CB_D, 2, page, core_ranges),
        _cb(CB_OUT, 2 * Wt, page, core_ranges),
    ]
    with_dgamma = partial is not None
    if with_dgamma:
        cbs.append(_cb(CB_ACC, Wt, page, core_ranges))

    reader_ct = [Wt, page]
    for t in (dy, x, gamma, inv_rms, d):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [Wt, page, 1 if with_dgamma else 0]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(partial if with_dgamma else out).get_compile_time_args())
    compute_ct = [Wt, NEG_ONE_BITS, stage, 1 if distinct_slots else 0, 1 if with_dgamma else 0, dg_mode]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in (dy, x, gamma, inv_rms, d)]
    part_addr = partial.buffer_address() if with_dgamma else 0
    for k, (core, (row_start, row_count)) in enumerate(zip(cores, assignment)):
        reader_rt[core.x][core.y] = [*addrs, row_start, row_count]
        writer_rt[core.x][core.y] = [out.buffer_address(), row_start, row_count, part_addr, k]
        compute_rt[core.x][core.y] = [row_count]

    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=full_sync or distinct_slots
    )
    # Every fp32 operand that is copied or broadcast into DEST must be routed unpack-to-dest,
    # otherwise it passes through the 19-bit srcA register (TF32 truncation). This tag also
    # drives get_operand_dst_format(), which is what unary_bcast's fp32 auto-detect keys on.
    modes = [ttnn.UnpackToDestMode.Default] * 64
    if unpack_to_dest:
        for cb in (CB_DY, CB_X, CB_GAMMA, CB_INV_RMS, CB_D):
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if with_dgamma:
        modes[CB_ACC] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = modes

    dm_defines = [("RB_STUB_DM", "1")] if stub_dm else []
    compute_defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")] if skip_compute else []
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER_KERNEL,
            source_type=src,
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=dm_defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER_KERNEL,
            source_type=src,
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=dm_defines,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE_KERNEL,
            source_type=src,
            core_ranges=core_ranges,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=compute_defines,
            config=compute_cfg,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def rmsnorm_bw_apply(x, dy, gamma, inv_rms, d, *, num_cores=None, with_dgamma=False, **debug):
    """dx = gamma*dy*inv_rms - x*d, one all-SFPU chain per tile-row. Returns dx (fp32).

    With with_dgamma=True also returns dgamma = sum over N,C,H of dy*x/rms as a [1,1,1,W] fp32
    tensor: each core pack-accumulates its rows' dy*x/rms into one [32, W] partial in L1, the
    writer emits the per-core partials as a [1,1,32*num_cores,W] tensor, and a single small
    ttnn.sum finishes the cross-core / cross-row reduction.
    """
    validate(x, dy, gamma, inv_rms, d)
    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    if not with_dgamma:
        desc = create_program_descriptor(x, dy, gamma, inv_rms, d, out, num_cores=num_cores, **debug)
        return ttnn.generic_op([x, dy, gamma, inv_rms, d, out], desc)

    n, c, h, w = x.shape
    grid = device.compute_with_storage_grid_size()
    ncores = min(num_cores or grid.x * grid.y, grid.x * grid.y, n * c * (h // TILE))
    partial = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, TILE * ncores, w]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = create_program_descriptor(x, dy, gamma, inv_rms, d, out, partial, num_cores=ncores, **debug)
    # generic_op returns io_tensors.back(); both `out` and `partial` are written by the program.
    ttnn.generic_op([x, dy, gamma, inv_rms, d, out, partial], desc)
    dgamma = ttnn.sum(partial, dim=[0, 1, 2], keepdim=True)
    return out, dgamma
