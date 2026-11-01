// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// output_chunk_interleave bake-off: out = c * inv_std (col bcast) * gamma (row bcast) + beta (row bcast).
// Pre: c (rows x Wt) pushed at the front of cb_wb, inv_std (rows) pushed at the front of cb_inv, gamma /
// beta (Wt row-0 tiles) pushed. Post: c and inv_std popped, rows x Wt output tiles pushed.
//
//   0 BASE          the op's three whole-row passes: wb -> wa (normalize), wa -> wb (scale), wb -> out (shift)
//   1 HELPER_CHUNK  chunk-major: per row-aligned chunk of G tiles, normalize wb -> wa, scale wa -> wc,
//                   shift wc -> out. One eltwise_chain call per op per chunk (helper owns init/reconfig).
//   2 RAW_CHUNK     variant 1 in raw LLK: one init per op per chunk, srcB / pack reconfig only on a format
//                   change (srcA is the work format for all three ops).
//   3 RAW_SKEW      variant 2 software-pipelined by one chunk per op (N0 | N1 S0 | N2 S1 H0 | ...): no op
//                   waits on the pack of the op issued right before it.
//   4 HELPER_SKEW   variant 3's schedule with the variant-1 helper calls.
//
// Why a third work CB (wc, 2 chunks): wa / wb alias one allocation and wb still holds the unread tail of
// c while chunk k is scaled, so scale cannot push into wb (its FIFO front is c, not the scaled chunk).
// Raw LLK (variants 2/3) bypasses compute_kernel_lib::eltwise_chain to keep the srcB / pack format
// state across the per-chunk op switches; the helper re-emits its fold-local reconfig on every call.

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace oci {

constexpr uint32_t BASE = 0;
constexpr uint32_t HELPER_CHUNK = 1;
constexpr uint32_t RAW_CHUNK = 2;
constexpr uint32_t RAW_SKEW = 3;
constexpr uint32_t HELPER_SKEW = 4;

using compute_kernel_lib::DataFormatReconfig;

constexpr compute_kernel_lib::InputSpec lane_in(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block, reconfig);
}

constexpr compute_kernel_lib::OutputSpec lane_out(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return output(cb, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize, reconfig);
}

template <uint32_t Wt, uint32_t G>
struct ChunkGrid {
    static constexpr uint32_t per_row = (Wt + G - 1) / G;
    static ALWI uint32_t row(uint32_t k) { return k / per_row; }
    static ALWI uint32_t col(uint32_t k) { return (k % per_row) * G; }
    static ALWI uint32_t tiles(uint32_t k) {
        const uint32_t c0 = col(k);
        return (Wt - c0) < G ? (Wt - c0) : G;
    }
};

// Issue order: op `o` (0 N, 1 S, 2 H) on chunk `step - o * skew`.
template <uint32_t Skew, class F>
ALWI void schedule(uint32_t num_chunks, F&& run_op) {
    const uint32_t steps = num_chunks + 2 * Skew;
    for (uint32_t step = 0; step < steps; ++step) {
        for (uint32_t op = 0; op < 3; ++op) {
            const uint32_t lag = op * Skew;
            if (step >= lag && step - lag < num_chunks) {
                run_op(op, step - lag);
            }
        }
    }
}

// One op on one chunk of n tiles: in (front, popped) x b_cb[b_base (+ lane for row bcast)] -> out.
template <uint32_t Op, uint32_t cb_in, uint32_t cb_b, uint32_t cb_o, uint32_t lanes>
ALWI void raw_chunk(
    DataflowBuffer& in, DataflowBuffer& out, uint32_t b_base, uint32_t n, uint32_t& srcb_cb, uint32_t& pack_cb) {
    if (srcb_cb != cb_b) {
        reconfig_data_format_srcb(srcb_cb, cb_b);
        srcb_cb = cb_b;
    }
    if (pack_cb != cb_o) {
        pack_reconfig_data_format(pack_cb, cb_o);
        pack_cb = cb_o;
    }
    if constexpr (Op == 0) {
        mul_bcast_cols_init(cb_in, cb_b);
    } else if constexpr (Op == 1) {
        mul_bcast_rows_init(cb_in, cb_b);
    } else {
        add_bcast_rows_init(cb_in, cb_b);
    }
    for (uint32_t t = 0; t < n; t += lanes) {
        const uint32_t m = (n - t) < lanes ? (n - t) : lanes;
        in.wait_front(m);
        tile_regs_acquire();
        for (uint32_t j = 0; j < m; ++j) {
            if constexpr (Op == 0) {
                mul_tiles_bcast_cols(cb_in, cb_b, j, b_base, j);
            } else if constexpr (Op == 1) {
                mul_tiles_bcast_rows(cb_in, cb_b, j, b_base + t + j, j);
            } else {
                add_tiles_bcast_rows(cb_in, cb_b, j, b_base + t + j, j);
            }
        }
        tile_regs_commit();
        out.reserve_back(m);
        tile_regs_wait();
        for (uint32_t j = 0; j < m; ++j) {
            pack_tile(j, cb_o);
        }
        tile_regs_release();
        out.push_back(m);
        in.pop_front(m);
    }
}

template <
    uint32_t Variant,
    uint32_t cb_wa,
    uint32_t cb_wb,
    uint32_t cb_wc,
    uint32_t cb_inv,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_out,
    uint32_t Wt,
    uint32_t dest_lanes,
    uint32_t G>
ALWI void output_stage(
    uint32_t rows,
    DataflowBuffer& work_a,
    DataflowBuffer& work_b,
    DataflowBuffer& work_c,
    DataflowBuffer& inv_std,
    DataflowBuffer& gamma,
    DataflowBuffer& beta,
    DataflowBuffer& output) {
    using namespace compute_kernel_lib;
    constexpr auto same_fmt = DataFormatReconfig::Disabled;

    if constexpr (Variant == BASE) {
        const auto lane_shape = IterationShape::grid(rows, Wt).block_size(dest_lanes);
        mul<lane_in(cb_wb),
            input(cb_inv, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_out(cb_wa)>(lane_shape, work_b, inv_std, work_a);
        gamma.wait_front(Wt);
        beta.wait_front(Wt);
        mul<lane_in(cb_wa, same_fmt),
            input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            lane_out(cb_wb, same_fmt)>(lane_shape, work_a, gamma, work_b);
        add<lane_in(cb_wb, same_fmt),
            input(cb_beta, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row, same_fmt),
            lane_out(cb_out)>(lane_shape, work_b, beta, output);
        return;
    }

    using Grid = ChunkGrid<Wt, G>;
    const uint32_t num_chunks = rows * Grid::per_row;
    constexpr uint32_t skew = (Variant == RAW_SKEW || Variant == HELPER_SKEW) ? 1 : 0;
    inv_std.wait_front(rows);
    gamma.wait_front(Wt);
    beta.wait_front(Wt);

    if constexpr (Variant == HELPER_CHUNK || Variant == HELPER_SKEW) {
        constexpr auto inv_spec = input(
            cb_inv,
            BroadcastDim::Col,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Scalar,
            DataFormatReconfig::Enabled,
            TileAddressing::Offset);
        constexpr auto gamma_spec = input(
            cb_gamma,
            BroadcastDim::Row,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Block,
            DataFormatReconfig::Enabled,
            TileAddressing::Offset);
        constexpr auto beta_spec = input(
            cb_beta,
            BroadcastDim::Row,
            WaitPolicy::None,
            PopPolicy::None,
            InputTileMapping::Block,
            DataFormatReconfig::Enabled,
            TileAddressing::Offset);
        schedule<skew>(num_chunks, [&](uint32_t op, uint32_t k) {
            const auto shape = IterationShape::tiles(Grid::tiles(k)).block_size(dest_lanes);
            if (op == 0) {
                eltwise_chain(
                    shape,
                    BinaryFpu<BinaryFpuOp::Mul, lane_in(cb_wb), inv_spec>{work_b, inv_std, 0, Grid::row(k)},
                    PackTile<lane_out(cb_wa)>{work_a});
            } else if (op == 1) {
                eltwise_chain(
                    shape,
                    BinaryFpu<BinaryFpuOp::Mul, lane_in(cb_wa), gamma_spec>{work_a, gamma, 0, Grid::col(k)},
                    PackTile<lane_out(cb_wc)>{work_c});
            } else {
                eltwise_chain(
                    shape,
                    BinaryFpu<BinaryFpuOp::Add, lane_in(cb_wc), beta_spec>{work_c, beta, 0, Grid::col(k)},
                    PackTile<lane_out(cb_out)>{output});
            }
        });
    } else {
        constexpr uint32_t lanes = dest_lanes < DEST_AUTO_LIMIT ? dest_lanes : DEST_AUTO_LIMIT;
        // srcA is the work format for every op here; only srcB (inv_std vs gamma/beta) and the pack
        // target (work vs output) change format between ops.
        reconfig_data_format(cb_wb, cb_inv);
        pack_reconfig_data_format(cb_wa);
        uint32_t srcb_cb = cb_inv;
        uint32_t pack_cb = cb_wa;
        schedule<skew>(num_chunks, [&](uint32_t op, uint32_t k) {
            const uint32_t r = Grid::row(k);
            const uint32_t c0 = Grid::col(k);
            const uint32_t n = Grid::tiles(k);
            if (op == 0) {
                raw_chunk<0, cb_wb, cb_inv, cb_wa, lanes>(work_b, work_a, r, n, srcb_cb, pack_cb);
            } else if (op == 1) {
                raw_chunk<1, cb_wa, cb_gamma, cb_wc, lanes>(work_a, work_c, c0, n, srcb_cb, pack_cb);
            } else {
                raw_chunk<2, cb_wc, cb_beta, cb_out, lanes>(work_c, output, c0, n, srcb_cb, pack_cb);
            }
        });
    }
    inv_std.pop_front(rows);
}

}  // namespace oci
