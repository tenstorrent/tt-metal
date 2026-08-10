// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The RMS normalisation of a candidate's dot against its own residual:
//
//   scores[c] = dots[c] * rsqrt(sum_squares[c] * inv_hidden_size + eps)
//
// Both statistics arrive from the same collective, so the whole chain is one
// pass over the reduced tensor. It runs entirely in dst — the SFPU scalar
// binops and the SFPU binary multiply are all dst-to-dst — so the scale, the
// epsilon, the reciprocal square root and the final multiply never round to
// the output dtype between steps.
//
// Above one partial the statistics arrive per rank rather than summed, and the
// cross-rank sum opens the chain. Folding it in here is what lets the collective
// be a gather: a reducing collective costs a second program to produce data this
// pass already has in dst.

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

namespace {

constexpr auto cb_stats = tt::CBIndex::c_0;  // [sum_squares, dots] per output tile
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t kOperands = 2;
constexpr uint32_t onetile = 1;

constexpr uint32_t dst_rms = 0;
constexpr uint32_t dst_dots = 1;
constexpr uint32_t dst_partial = 2;

}  // namespace

void kernel_main() {
    // compile-time args — fp32 values as their bit patterns, which is what the
    // SFPU scalar binops take.
    constexpr uint32_t inv_hidden_size = get_compile_time_arg_val(0);
    constexpr uint32_t eps = get_compile_time_arg_val(1);
    constexpr uint32_t num_partials = get_compile_time_arg_val(2);

    constexpr uint32_t tiles_per_candidate = kOperands * num_partials;

    // runtime args
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);

    CircularBuffer cb_stats_obj(cb_stats);
    CircularBuffer cb_out_obj(cb_out);

    unary_op_init_common(cb_stats, cb_out);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        cb_stats_obj.wait_front(tiles_per_candidate);
        tile_regs_acquire();
        copy_tile(cb_stats, 0, dst_rms);
        copy_tile(cb_stats, 1, dst_dots);

        if constexpr (num_partials > 1) {
            add_binary_tile_init();
            for (uint32_t p = 1; p < num_partials; ++p) {
                copy_tile(cb_stats, kOperands * p, dst_partial);
                add_binary_tile(dst_rms, dst_partial, dst_rms);
                copy_tile(cb_stats, kOperands * p + 1, dst_partial);
                add_binary_tile(dst_dots, dst_partial, dst_dots);
            }
        }

        binop_with_scalar_tile_init();
        mul_unary_tile(dst_rms, inv_hidden_size);
        add_unary_tile(dst_rms, eps);

        rsqrt_tile_init();
        rsqrt_tile(dst_rms);

        mul_binary_tile_init();
        mul_binary_tile(dst_dots, dst_rms, dst_dots);
        tile_regs_commit();
        cb_stats_obj.pop_front(tiles_per_candidate);

        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst_dots, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);
    }
}
