// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One token row-tile's scores for every site, in the statistics tile's own layout: a
// tile row is a token, column 0 the candidate's sum of squares and column 1 + r its dot
// with query r. Every step below is elementwise across tiles of that layout, so the
// sites ride along in the columns and nothing is transposed:
//
//   stats_c  = sum_p partial_{c,p}                           across tensor-parallel ranks
//   rr_c     = rsqrt(stats_c * inv_hidden_size + eps)       column 0 meaningful
//   scores_c = stats_c * rr_c[:, 0]                          column broadcast
//   shift    = max_c scores_c
//   exp_c    = exp(scores_c - shift)
//   mass     = sum_c exp_c
//
// Columns outside 1..R carry whatever the arithmetic makes of them; the writer reads
// only the site columns.

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/device/kernels/attn_res_scores_items.hpp"

using namespace ckernel;

namespace {

constexpr auto cb_stats = tt::CBIndex::c_0;
constexpr auto cb_rr = tt::CBIndex::c_1;
constexpr auto cb_scores = tt::CBIndex::c_2;
constexpr auto cb_shift = tt::CBIndex::c_3;
constexpr auto cb_summed = tt::CBIndex::c_4;
constexpr auto cb_shift_out = tt::CBIndex::c_16;
constexpr auto cb_exp_out = tt::CBIndex::c_17;
constexpr auto cb_mass_out = tt::CBIndex::c_18;

constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
constexpr uint32_t num_partials = get_compile_time_arg_val(1);
// fp32 bit patterns, which is what the SFPU scalar binops take.
constexpr uint32_t inv_hidden_size = get_compile_time_arg_val(2);
constexpr uint32_t eps = get_compile_time_arg_val(3);
constexpr uint32_t kSplitArgs = 4;

constexpr uint32_t dst_a = 0;
constexpr uint32_t dst_b = 1;
constexpr uint32_t dst_c = 2;

// `scores_c - shift`, exponentiated, in `dst`. `dst_b` holds the shift.
ALWI void centred_exp(uint32_t c, uint32_t dst) {
    copy_tile(cb_scores, c, dst);
    sub_binary_tile_init();
    sub_binary_tile(dst, dst_b, dst);
    exp_tile_init();
    exp_tile(dst);
}

}  // namespace

void kernel_main() {
    CircularBuffer cb_stats_obj(cb_stats);
    CircularBuffer cb_rr_obj(cb_rr);
    CircularBuffer cb_scores_obj(cb_scores);
    CircularBuffer cb_shift_obj(cb_shift);
    CircularBuffer cb_summed_obj(cb_summed);
    CircularBuffer cb_shift_out_obj(cb_shift_out);
    CircularBuffer cb_exp_out_obj(cb_exp_out);
    CircularBuffer cb_mass_out_obj(cb_mass_out);

    compute_kernel_hw_startup(cb_stats, cb_summed, cb_scores);

    // Every item of this core repeats the same arithmetic on its row; which sites the
    // writer keeps is none of compute's business.
    for_each_attn_res_scores_item<kSplitArgs, 1>([&](const AttnResScoresItem&) {
        // The statistics, summed across ranks in fp32 dest.
        cb_stats_obj.wait_front(num_candidates * num_partials);
        cb_summed_obj.reserve_back(num_candidates);
        copy_init(cb_stats);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            tile_regs_acquire();
            copy_tile(cb_stats, c * num_partials, dst_a);
            for (uint32_t p = 1; p < num_partials; ++p) {
                copy_tile(cb_stats, c * num_partials + p, dst_b);
                add_binary_tile_init();
                add_binary_tile(dst_a, dst_b, dst_a);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_a, cb_summed);
            tile_regs_release();
        }
        cb_summed_obj.push_back(num_candidates);
        cb_stats_obj.pop_front(num_candidates * num_partials);

        // Reciprocal RMS per candidate. Only column 0 means anything.
        cb_summed_obj.wait_front(num_candidates);
        cb_rr_obj.reserve_back(num_candidates);
        copy_init(cb_summed);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            tile_regs_acquire();
            copy_tile(cb_summed, c, dst_a);
            binop_with_scalar_tile_init();
            mul_unary_tile(dst_a, inv_hidden_size);
            add_unary_tile(dst_a, eps);
            rsqrt_tile_init();
            rsqrt_tile(dst_a);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_a, cb_rr);
            tile_regs_release();
        }
        cb_rr_obj.push_back(num_candidates);

        // Scores: every column of a candidate's statistics scaled by its column 0.
        cb_rr_obj.wait_front(num_candidates);
        cb_scores_obj.reserve_back(num_candidates);
        mul_bcast_cols_init(cb_summed, cb_rr);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_summed, cb_rr, c, c, dst_a);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_a, cb_scores);
            tile_regs_release();
        }
        cb_scores_obj.push_back(num_candidates);
        cb_rr_obj.pop_front(num_candidates);
        cb_summed_obj.pop_front(num_candidates);

        // The shift, packed twice: compute reads it back below, and a circular buffer
        // has a single consumer.
        cb_scores_obj.wait_front(num_candidates);
        cb_shift_obj.reserve_back(1);
        cb_shift_out_obj.reserve_back(1);
        copy_init(cb_scores);
        tile_regs_acquire();
        copy_tile(cb_scores, 0, dst_a);
        for (uint32_t c = 1; c < num_candidates; ++c) {
            copy_tile(cb_scores, c, dst_b);
            binary_max_tile_init();
            binary_max_tile(dst_a, dst_b, dst_a);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_a, cb_shift);
        pack_tile(dst_a, cb_shift_out);
        tile_regs_release();
        cb_shift_obj.push_back(1);
        cb_shift_out_obj.push_back(1);

        // The exponentials, one candidate at a time.
        cb_shift_obj.wait_front(1);
        cb_exp_out_obj.reserve_back(num_candidates);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            tile_regs_acquire();
            copy_tile(cb_shift, 0, dst_b);
            centred_exp(c, dst_a);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_a, cb_exp_out);
            tile_regs_release();
        }
        cb_exp_out_obj.push_back(num_candidates);

        // The mass, summed in dest from the same exponentials: recomputing them is a
        // few SFPU passes, where reading the packed ones back would need a second
        // consumer of a buffer the writer drains.
        cb_mass_out_obj.reserve_back(1);
        tile_regs_acquire();
        copy_tile(cb_shift, 0, dst_b);
        centred_exp(0, dst_a);
        for (uint32_t c = 1; c < num_candidates; ++c) {
            centred_exp(c, dst_c);
            add_binary_tile_init();
            add_binary_tile(dst_a, dst_c, dst_a);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_a, cb_mass_out);
        tile_regs_release();
        cb_mass_out_obj.push_back(1);

        cb_shift_obj.pop_front(1);
        cb_scores_obj.pop_front(num_candidates);
    });
}
