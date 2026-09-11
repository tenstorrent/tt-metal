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
CB_ACC = 17  # dgamma per-row accumulator, compute-private (L1 pack-accumulate across rows)
CB_PART = 18  # dgamma collapsed column tiles (row 0 valid), compute -> writer
CB_ZERO_DONE = 19  # compute -> writer: gather zero-filled
CB_GO = 20  # writer -> compute: all contributions landed
SEM_READY1, SEM_ARRIVE1, SEM_READY2, SEM_ARRIVE2 = 0, 1, 2, 3

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
#include "api/compute/compute_kernel_api.h"  // sfpu_reduce (within-tile SFPU column collapse)
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_dy = 0, cb_x = 1, cb_gamma = 2, cb_inv = 3, cb_d = 4, cb_out = 16, cb_acc = 17, cb_part = 18;
    constexpr uint32_t cb_zero_done = 19, cb_go = 20;
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
        // ---- dgamma: on-chip two-stage tree reduction (tensix_all_reduce "tree" pattern) ----
        // Every stage is the same primitive: a gather buffer (cb_acc, Wt tiles, zero-filled) whose ROW i
        // holds contributor i's 32-float partial per column tile, collapsed 32 rows -> 1 on the SFPU.
        //   stage 0 (every core): collapse the L1-accumulated per-row partials of this core's tile-rows.
        //   stage 1 (row leaders, x == 0): contributors of the grid row scatter into row x; collapse.
        //   stage 2 (root, (0,0)): row leaders scatter into row y; collapse -> final dgamma row.
        // The dataflow (writer) kernel owns the NoC scatter + semaphores; the handshakes with compute
        // are two 1-page CBs: cb_zero_done (compute -> writer: gather is zero-filled, open for writes)
        // and cb_go (writer -> compute: every contribution has landed, collapse now).
        const uint32_t role = get_arg_val<uint32_t>(1);  // 0 member, 1 row leader, 2 root

        // Collapse cb_acc (Wt resident tiles, 32 per-row partials each) into cb_part (row 0 = column sums).
        // copy_tile reads cb_acc through unpack-to-dest so the fp32 partials reach DEST unrounded;
        // sfpu_reduce is the same SFPU collapse the reduce helper's AccumulateViaAdd finalize uses.
        auto collapse_acc_to_part = [&]() {
            cb_wait_front(cb_acc, Wt);
            cb_reserve_back(cb_part, Wt);
            reconfig_data_format_srca(cb_acc);
            pack_reconfig_data_format(cb_part);
            copy_tile_to_dst_init_short(cb_acc);
            sfpu_reduce_init<ckernel::PoolType::SUM, DataFormat::Float32>();
            for (uint32_t c = 0; c < Wt; ++c) {
                tile_regs_acquire();
                copy_tile(cb_acc, c, 0);
                sfpu_reduce<ckernel::PoolType::SUM, DataFormat::Float32, ckernel::ReduceDim::REDUCE_COL>(0, 1, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_part);
                tile_regs_release();
            }
            cb_push_back(cb_part, Wt);
            cb_pop_front(cb_acc, Wt);
        };
        // Zero-fill cb_acc as a gather buffer and tell the writer it is open for remote writes.
        auto open_gather = [&]() {
            cb_reserve_back(cb_acc, Wt);
            eltwise_chain(
                IterationShape::row(Wt),
                FillScalar<Dst::D0>{0.0f},
                PackTile<output(cb_acc, ReservePolicy::None, PushPolicy::None), Dst::D0>{});
            cb_push_back(cb_acc, Wt);
            cb_reserve_back(cb_zero_done, 1);
            cb_push_back(cb_zero_done, 1);
        };
        auto wait_go = [&]() {
            cb_wait_front(cb_go, 1);
            cb_pop_front(cb_go, 1);
        };

        cb_push_back(cb_acc, Wt);
        collapse_acc_to_part();  // stage 0: this core's partial row
        for (uint32_t stage = 0; stage < role; ++stage) {  // leader: 1 gather; root: 2 gathers
            open_gather();
            wait_go();
            collapse_acc_to_part();
        }
    }
}
"""

# ---------------------------------------------------------------------------------------------
# Writer (BRISC / NoC1): per tile-row, drain Wt output tiles to DRAM.
# ---------------------------------------------------------------------------------------------
_WRITER_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// Scatter ROW 0 of each of the Wt collapsed column tiles in `src_l1` (32 fp32 = 64 B in face 0 + 64 B in
// face 1) into in-tile row `row` of the destination core's gather buffer (same CB => same L1 address on
// every core). Row r of a 32x32 fp32 tile: face (r<16 ? 0 : 2) [+1 for cols 16..31], byte (r%16)*64.
FORCE_INLINE void scatter_row(uint32_t src_l1, uint32_t dst_l1, uint32_t dst_x, uint32_t dst_y, uint32_t row,
                              uint32_t Wt, uint32_t page) {
    const uint32_t face_r = (row < 16u) ? 0u : 2u;
    const uint32_t in_face = (row & 15u) * 64u;
    for (uint32_t c = 0; c < Wt; ++c) {
        const uint32_t s = src_l1 + c * page;
        const uint32_t d = dst_l1 + c * page + face_r * 1024u + in_face;
        noc_async_write(s, get_noc_addr(dst_x, dst_y, d), 64u);
        noc_async_write(s + 1024u, get_noc_addr(dst_x, dst_y, d + 1024u), 64u);
    }
}

void kernel_main() {
    constexpr uint32_t cb_out = 16, cb_acc = 17, cb_part = 18, cb_zero_done = 19, cb_go = 20;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    constexpr uint32_t with_dgamma = get_compile_time_arg_val(2);
    constexpr uint32_t sem_ready1 = get_compile_time_arg_val(3);
    constexpr uint32_t sem_arrive1 = get_compile_time_arg_val(4);
    constexpr uint32_t sem_ready2 = get_compile_time_arg_val(5);
    constexpr uint32_t sem_arrive2 = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto dg_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);
    const uint32_t dg_addr = get_arg_val<uint32_t>(3);
    const uint32_t role = get_arg_val<uint32_t>(4);      // 0 member, 1 row leader, 2 root
    const uint32_t my_col = get_arg_val<uint32_t>(5);    // logical grid x (row index inside the leader's gather)
    const uint32_t my_row = get_arg_val<uint32_t>(6);    // logical grid y (row index inside the root's gather)
    const uint32_t my_vx = get_arg_val<uint32_t>(7);
    const uint32_t my_vy = get_arg_val<uint32_t>(8);
    const uint32_t leader_vx = get_arg_val<uint32_t>(9);
    const uint32_t leader_vy = get_arg_val<uint32_t>(10);
    const uint32_t root_vx = get_arg_val<uint32_t>(11);
    const uint32_t root_vy = get_arg_val<uint32_t>(12);
    const uint32_t row_cols = get_arg_val<uint32_t>(13);   // contributors in my grid row (leaders only)
    const uint32_t n_leaders = get_arg_val<uint32_t>(14);  // row leaders (root only)
    const uint32_t ntab = get_arg_val<uint32_t>(15);       // table entries that follow
    // table: virtual x of grid column i (i < row_cols), then virtual y of grid row j (j < n_leaders)
    constexpr uint32_t TAB = 16;

    const auto out_acc = TensorAccessor(out_args, out_addr, page);
    const auto dg_acc = TensorAccessor(dg_args, dg_addr, page);

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
        Semaphore<> ready1(sem_ready1), arrive1(sem_arrive1), ready2(sem_ready2), arrive2(sem_arrive2);
        Noc noc;
        // The gather buffer is cb_acc. This kernel never reserves/pushes it, so its write pointer is the CB
        // base — identical on every core (same CBDescriptor), which is what makes the remote address below
        // resolvable: the leader's compute zero-fills exactly that window before opening it.
        const uint32_t gather_l1 = get_write_ptr(cb_acc);
        auto open_and_signal = [&](Semaphore<>& ready, uint32_t n, uint32_t tab_off, bool along_x) {
            cb_wait_front(cb_zero_done, 1);
            cb_pop_front(cb_zero_done, 1);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t v = get_arg_val<uint32_t>(TAB + tab_off + i);
                if (along_x) ready.up(noc, v, my_vy, 1); else ready.up(noc, my_vx, v, 1);
            }
        };
        auto go = [&]() {
            cb_reserve_back(cb_go, 1);
            cb_push_back(cb_go, 1);
        };

        // ---- stage 1: my partial row -> my row leader's gather, row my_col ----
        cb_wait_front(cb_part, Wt);  // stage-0 partial (compute collapsed my accumulated rows)
        if (role >= 1) open_and_signal(ready1, row_cols, /*tab_off=*/0, /*along_x=*/true);
        ready1.wait_min(1);
        scatter_row(get_read_ptr(cb_part), gather_l1, leader_vx, leader_vy, my_col, Wt, page);
        noc_async_write_barrier();
        arrive1.up(noc, leader_vx, leader_vy, 1);
        cb_pop_front(cb_part, Wt);

        if (role >= 1) {
            arrive1.wait_min(row_cols);
            go();  // compute collapses the row gather -> cb_part (row partial)
            // ---- stage 2: row partial -> root's gather, row my_row ----
            cb_wait_front(cb_part, Wt);
            if (role == 2) open_and_signal(ready2, n_leaders, /*tab_off=*/row_cols, /*along_x=*/false);
            ready2.wait_min(1);
            scatter_row(get_read_ptr(cb_part), gather_l1, root_vx, root_vy, my_row, Wt, page);
            noc_async_write_barrier();
            arrive2.up(noc, root_vx, root_vy, 1);
            cb_pop_front(cb_part, Wt);
        }
        if (role == 2) {
            arrive2.wait_min(n_leaders);
            go();  // compute collapses the column gather -> cb_part (final dgamma, row 0 valid)
            cb_wait_front(cb_part, Wt);
            const uint32_t l1 = get_read_ptr(cb_part);
#ifndef RB_STUB_DM
            for (uint32_t c = 0; c < Wt; ++c) {  // dgamma is [1,1,1,W]: one padded tile-row, row 0 logical
                noc_async_write(l1 + c * page, dg_acc.get_noc_addr(c), page);
            }
#endif
            noc_async_write_barrier();
            cb_pop_front(cb_part, Wt);
        }
        (void)ntab;
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
    dgamma_out=None,
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
    with_dgamma = dgamma_out is not None
    semaphores = []
    if with_dgamma:
        cbs.append(_cb(CB_ACC, Wt, page, core_ranges))
        cbs.append(_cb(CB_PART, Wt, page, core_ranges))
        for cb_id in (CB_ZERO_DONE, CB_GO):
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=64,
                    core_ranges=core_ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.float32, page_size=64)
                    ],
                )
            )
        semaphores = [
            ttnn.SemaphoreDescriptor(id=sid, core_ranges=core_ranges, initial_value=0)
            for sid in (SEM_READY1, SEM_ARRIVE1, SEM_READY2, SEM_ARRIVE2)
        ]

    reader_ct = [Wt, page]
    for t in (dy, x, gamma, inv_rms, d):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [Wt, page, 1 if with_dgamma else 0, SEM_READY1, SEM_ARRIVE1, SEM_READY2, SEM_ARRIVE2]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(dgamma_out if with_dgamma else out).get_compile_time_args())
    compute_ct = [Wt, NEG_ONE_BITS, stage, 1 if distinct_slots else 0, 1 if with_dgamma else 0, dg_mode]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in (dy, x, gamma, inv_rms, d)]
    # Tree topology over the logical grid: row leaders at x == 0 gather their grid row (stage 1), the
    # root (0, 0) gathers the leaders down column 0 (stage 2). Ragged last row is handled per leader.
    dg_addr = dgamma_out.buffer_address() if with_dgamma else 0
    vcoord = lambda cx, cy: device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
    n_rows_used = (num_cores + grid.x - 1) // grid.x
    root_v = vcoord(0, 0)
    for k, (core, (row_start, row_count)) in enumerate(zip(cores, assignment)):
        reader_rt[core.x][core.y] = [*addrs, row_start, row_count]
        role = 2 if (core.x == 0 and core.y == 0) else (1 if core.x == 0 else 0)
        row_cols = min(grid.x, num_cores - core.y * grid.x)
        me, leader = vcoord(core.x, core.y), vcoord(0, core.y)
        table = [vcoord(cx, core.y).x for cx in range(row_cols)] + [vcoord(0, cy).y for cy in range(n_rows_used)]
        writer_rt[core.x][core.y] = [
            out.buffer_address(),
            row_start,
            row_count,
            dg_addr,
            role,
            core.x,
            core.y,
            me.x,
            me.y,
            leader.x,
            leader.y,
            root_v.x,
            root_v.y,
            row_cols,
            n_rows_used,
            len(table),
            *table,
        ]
        compute_rt[core.x][core.y] = [row_count, role]

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
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def rmsnorm_bw_apply(x, dy, gamma, inv_rms, d, *, num_cores=None, with_dgamma=False, **debug):
    """dx = gamma*dy*inv_rms - x*d, one all-SFPU chain per tile-row. Returns dx (fp32).

    With with_dgamma=True also returns dgamma = sum over N,C,H of dy*x/rms as a [1,1,1,W] fp32
    tensor, fully on-chip (no follow-up ops): each core pack-accumulates its rows' dy*x/rms into a
    [32, W] fp32 L1 window and collapses it on the SFPU; the per-core rows are then tree-reduced over
    the core grid (row members -> row leader, row leaders -> root; each hop scatters 32-float rows
    into the receiver's zero-filled gather tiles, which the receiver collapses on the SFPU again).
    The root writes the final row into `dgamma`.
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
    # dgamma [1,1,1,W]: the root core writes its final collapsed tiles straight into this tensor's single
    # (padded) tile-row; row 0 is the logical row, rows 1..31 are padding and hold stale lanes.
    dgamma = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, w]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = create_program_descriptor(x, dy, gamma, inv_rms, d, out, dgamma, num_cores=num_cores, **debug)
    # generic_op returns io_tensors.back(); both `out` and `dgamma` are written by the program.
    ttnn.generic_op([x, dy, gamma, inv_rms, d, out, dgamma], desc)
    return out, dgamma
