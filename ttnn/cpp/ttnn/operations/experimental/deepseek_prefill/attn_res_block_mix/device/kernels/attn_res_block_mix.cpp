// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One run of a token row-tile: score the row, then mix every site's partial over the run.
//
// Scoring stays in the statistics tile's own layout — a tile row is a token, column 0 the
// candidate's sum of squares, column 1 + r its dot with query r — so every step is
// elementwise across tiles and the sites ride along in the columns:
//
//   stats_c  = sum_p partial_{c,p}                       across tensor-parallel ranks
//   rr_c     = rsqrt(stats_c * inv_hidden_size + eps)   column 0 meaningful
//   scores_c = stats_c * rr_c[:, 0]                      column broadcast
//   shift    = max_c scores_c
//   e_c      = exp(scores_c - shift)
//   mass     = sum_c e_c
//
// The mixture MACs each candidate tile into one dest accumulator per site, the weight a
// column broadcast of the site's own weight tile. The run's candidate tiles are read from
// DRAM once and stay in their buffer until every site group has used them.

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_items.hpp"

using namespace ckernel;

namespace {

constexpr auto cb_stats = tt::CBIndex::c_0;
constexpr auto cb_rr = tt::CBIndex::c_1;
constexpr auto cb_scores = tt::CBIndex::c_2;
constexpr auto cb_shift = tt::CBIndex::c_3;
constexpr auto cb_summed = tt::CBIndex::c_4;
constexpr auto cb_exp = tt::CBIndex::c_5;
constexpr auto cb_values = tt::CBIndex::c_6;
constexpr auto cb_weights = tt::CBIndex::c_7;
constexpr auto cb_shift_out = tt::CBIndex::c_16;
constexpr auto cb_mass_out = tt::CBIndex::c_17;
constexpr auto cb_partials = tt::CBIndex::c_18;      // even site groups, for the writer
constexpr auto cb_partials_odd = tt::CBIndex::c_19;  // odd site groups, for the reader

constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
constexpr uint32_t num_partials = get_compile_time_arg_val(1);
constexpr uint32_t num_sites = get_compile_time_arg_val(2);
// fp32 bit patterns, which is what the SFPU scalar binops take.
constexpr uint32_t inv_hidden_size = get_compile_time_arg_val(3);
constexpr uint32_t eps = get_compile_time_arg_val(4);
constexpr uint32_t sites_per_group = get_compile_time_arg_val(5);
constexpr uint32_t num_groups = get_compile_time_arg_val(6);
constexpr uint32_t kSplitArgs = 7;

constexpr uint32_t dst_a = 0;
constexpr uint32_t dst_b = 1;
constexpr uint32_t dst_c = 2;

// One dest session per candidate: `body(c)` computes into dst_a, which is packed to `out`.
template <typename Body>
ALWI void per_candidate(uint32_t out, Body body) {
    for (uint32_t c = 0; c < num_candidates; ++c) {
        tile_regs_acquire();
        body(c);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_a, out);
        tile_regs_release();
    }
}

// `scores_c - shift`, exponentiated, in `dst`. `dst_b` holds the shift.
ALWI void centred_exp(uint32_t c, uint32_t dst) {
    copy_tile(cb_scores, c, dst);
    sub_binary_tile_init();
    sub_binary_tile(dst, dst_b, dst);
    exp_tile_init();
    exp_tile(dst);
}

// The row's exponentials (packed, one tile per candidate) for the writer to spread into
// per-site weights, and its shift and mass for the writer to scatter.
void score_row() {
    CircularBuffer cb_stats_obj(cb_stats);
    CircularBuffer cb_rr_obj(cb_rr);
    CircularBuffer cb_scores_obj(cb_scores);
    CircularBuffer cb_shift_obj(cb_shift);
    CircularBuffer cb_summed_obj(cb_summed);
    CircularBuffer cb_exp_obj(cb_exp);
    CircularBuffer cb_shift_out_obj(cb_shift_out);
    CircularBuffer cb_mass_out_obj(cb_mass_out);

    pack_reconfig_data_format(cb_summed);

    // The statistics, summed across ranks in fp32 dest.
    cb_stats_obj.wait_front(num_candidates * num_partials);
    cb_summed_obj.reserve_back(num_candidates);
    copy_init(cb_stats);
    per_candidate(cb_summed, [](uint32_t c) {
        copy_tile(cb_stats, c * num_partials, dst_a);
        for (uint32_t p = 1; p < num_partials; ++p) {
            copy_tile(cb_stats, c * num_partials + p, dst_b);
            add_binary_tile_init();
            add_binary_tile(dst_a, dst_b, dst_a);
        }
    });
    cb_summed_obj.push_back(num_candidates);
    cb_stats_obj.pop_front(num_candidates * num_partials);

    // Reciprocal RMS per candidate. Only column 0 means anything.
    cb_summed_obj.wait_front(num_candidates);
    cb_rr_obj.reserve_back(num_candidates);
    copy_init(cb_summed);
    per_candidate(cb_rr, [](uint32_t c) {
        copy_tile(cb_summed, c, dst_a);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_a, inv_hidden_size);
        add_unary_tile(dst_a, eps);
        rsqrt_tile_init();
        rsqrt_tile(dst_a);
    });
    cb_rr_obj.push_back(num_candidates);

    // Scores: every column of a candidate's statistics scaled by its column 0.
    cb_rr_obj.wait_front(num_candidates);
    cb_scores_obj.reserve_back(num_candidates);
    mul_bcast_cols_init(cb_summed, cb_rr);
    per_candidate(cb_scores, [](uint32_t c) { mul_tiles_bcast_cols(cb_summed, cb_rr, c, c, dst_a); });
    cb_scores_obj.push_back(num_candidates);
    cb_rr_obj.pop_front(num_candidates);
    cb_summed_obj.pop_front(num_candidates);

    // The shift, packed twice: compute reads it back below, and a circular buffer has a
    // single consumer.
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

    // The exponentials, in the weights' dtype.
    cb_shift_obj.wait_front(1);
    cb_exp_obj.reserve_back(num_candidates);
    pack_reconfig_data_format(cb_exp);
    per_candidate(cb_exp, [](uint32_t c) {
        copy_tile(cb_shift, 0, dst_b);
        centred_exp(c, dst_a);
    });
    cb_exp_obj.push_back(num_candidates);

    // The mass, summed in fp32 dest from the same exponentials: recomputing them is a few
    // SFPU passes, where the packed ones are bf16 and drained by the writer.
    cb_mass_out_obj.reserve_back(1);
    pack_reconfig_data_format(cb_mass_out);
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
}

// Every site's partial over the run, a site group at a time: one accumulator per site of the
// group, the group's weights for site s on candidate c at s * C + c, the run's candidate
// tiles held for every group at w * C + c.
void mix_run(uint32_t num_tiles) {
    CircularBuffer cb_values_obj(cb_values);
    CircularBuffer cb_weights_obj(cb_weights);
    CircularBuffer cb_partials_obj(cb_partials);
    CircularBuffer cb_partials_odd_obj(cb_partials_odd);

    // acc_to_dest turns each broadcast multiply into a MAC against its accumulator. The
    // accumulators start at zero: the packer clears a dest section on every release.
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_values, cb_weights);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_values, cb_weights, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_values, cb_weights);
    pack_reconfig_data_format(cb_partials);

    cb_values_obj.wait_front(num_tiles * num_candidates);
    for (uint32_t group = 0; group < num_groups; ++group) {
        const uint32_t first_site = group * sites_per_group;
        const uint32_t sites = first_site + sites_per_group <= num_sites ? sites_per_group : num_sites - first_site;
        cb_weights_obj.wait_front(sites * num_candidates);
        for (uint32_t w = 0; w < num_tiles; ++w) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < num_candidates; ++c) {
                for (uint32_t s = 0; s < sites; ++s) {
                    mul_tiles_bcast_cols(cb_values, cb_weights, w * num_candidates + c, s * num_candidates + c, s);
                }
            }
            tile_regs_commit();
            // Odd groups go out on the reader's NoC; see attn_res_block_mix_partials.hpp.
            const bool odd = group % 2 == 1;
            CircularBuffer& out_obj = odd ? cb_partials_odd_obj : cb_partials_obj;
            const auto out = odd ? cb_partials_odd : cb_partials;
            out_obj.reserve_back(sites);
            tile_regs_wait();
            for (uint32_t s = 0; s < sites; ++s) {
                pack_tile(s, out);
            }
            tile_regs_release();
            out_obj.push_back(sites);
        }
        cb_weights_obj.pop_front(sites * num_candidates);
    }
    cb_values_obj.pop_front(num_tiles * num_candidates);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_stats, cb_summed, cb_scores);
    for_each_attn_res_block_mix_item<kSplitArgs>([](const AttnResBlockMixItem& item) {
        score_row();
        mix_run(item.num_tiles());
    });
}
