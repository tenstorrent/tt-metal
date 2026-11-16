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
// Each step is an eltwise chain. The folds across partials and candidates (the sum, max and
// mass) are one dest session apiece, in fp32 dest on the SFPU; see attn_res_score_chain.hpp.
//
// A closed site has no live stream to fold in, so its weights are e_c / mass and its partial
// is the read's value.
//
// The mixture MACs each candidate tile into one dest accumulator per site, the weight a
// column broadcast of the site's own weight tile. The run's candidate tiles are read from
// DRAM once and stay in their buffer until every site group has used them.

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/attn_res_score_chain.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_items.hpp"

using namespace ckernel;
using namespace compute_kernel_lib;
using attn_res::eltwise_chain_of;
using attn_res::owning_tile_at;
using attn_res::tile_at;
using attn_res::unroll;

namespace {

constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
constexpr uint32_t cb_rr = get_named_compile_time_arg_val("cb_rr");
constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
constexpr uint32_t cb_shift = get_named_compile_time_arg_val("cb_shift");
constexpr uint32_t cb_summed = get_named_compile_time_arg_val("cb_summed");
constexpr uint32_t cb_exp = get_named_compile_time_arg_val("cb_exp");
constexpr uint32_t cb_values = get_named_compile_time_arg_val("cb_values");
constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
constexpr uint32_t cb_mass = get_named_compile_time_arg_val("cb_mass");
constexpr uint32_t cb_closed_mask = get_named_compile_time_arg_val("cb_closed_mask");
constexpr uint32_t cb_factor = get_named_compile_time_arg_val("cb_factor");
constexpr uint32_t cb_shift_out = get_named_compile_time_arg_val("cb_shift_out");
constexpr uint32_t cb_mass_out = get_named_compile_time_arg_val("cb_mass_out");
constexpr uint32_t cb_partials = get_named_compile_time_arg_val("cb_partials");  // even site groups, for the writer
constexpr uint32_t cb_partials_odd =
    get_named_compile_time_arg_val("cb_partials_odd");  // odd site groups, for the reader

constexpr uint32_t num_candidates = get_named_compile_time_arg_val("num_candidates");
constexpr uint32_t num_partials = get_named_compile_time_arg_val("num_partials");
constexpr uint32_t num_sites = get_named_compile_time_arg_val("num_sites");
// fp32 bit patterns, which is what the SFPU scalar binops take.
constexpr uint32_t inv_hidden_size = get_named_compile_time_arg_val("inv_hidden_size");
constexpr uint32_t eps = get_named_compile_time_arg_val("eps");
constexpr uint32_t sites_per_group = get_named_compile_time_arg_val("sites_per_group");
constexpr uint32_t num_groups = get_named_compile_time_arg_val("num_groups");
constexpr uint32_t num_closed_sites = get_named_compile_time_arg_val("num_closed_sites");
constexpr bool has_closed_sites = num_closed_sites > 0;
constexpr uint32_t kOne = 0x3F800000;  // 1.0f

using Buffer = DataflowBufferFor<dfb::AccessPattern::UNKNOWN, dfb::AccessPattern::UNKNOWN>;

// The scoring's buffers, bound once for every chain that reads or writes them.
struct ScoreBuffers {
    Buffer stats{cb_stats};
    Buffer summed{cb_summed};
    Buffer rr{cb_rr};
    Buffer scores{cb_scores};
    Buffer shift{cb_shift};
    Buffer shift_out{cb_shift_out};
    Buffer mass{cb_mass};
    Buffer mass_out{cb_mass_out};
    Buffer closed_mask{cb_closed_mask};
    Buffer factor{cb_factor};
    Buffer exp{cb_exp};
};

// `scores_c - shift`, exponentiated, into `Slot`. D1 holds the shift.
template <Dst Slot>
ALWI auto centred_exp(Buffer& scores, uint32_t c) {
    return std::make_tuple(
        CopyTile<tile_at(cb_scores), Slot>{scores, c}, SubBinary<Slot, Dst::D1, Slot>{}, Exp<Approx::Exact, Slot>{});
}

// Per site, what its exponentials are scaled by: 1 / mass in a closed site's column, 1 in
// the rest. The mask is 1 in the closed sites' columns and 0 elsewhere.
void closed_site_factor(ScoreBuffers& b) {
    eltwise_chain(
        IterationShape::one_tile(),
        CopyTile<input(cb_mass, WaitPolicy::Upfront, PopPolicy::AtEnd)>{b.mass},
        Recip<>{},
        SubUnary<>{kOne},
        CopyTile<input(cb_closed_mask, WaitPolicy::None, PopPolicy::None), Dst::D1>{b.closed_mask},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        AddUnary<>{kOne},
        PackTile<output(cb_factor)>{b.factor});
}

// The row's exponentials (packed, one tile per candidate) for the writer to spread into
// per-site weights, and its shift and mass for the writer to scatter.
void score_row(ScoreBuffers& b) {
    // The statistics, summed across ranks in fp32 dest. The reader lays them out
    // partial-major, so partial p of every candidate is the window at p * C; the last
    // partial's read waits for the whole set and pops it.
    eltwise_chain_of(
        IterationShape::tiles(num_candidates),
        unroll<num_partials>([&](auto p) {
            constexpr bool last = p == num_partials - 1;
            constexpr auto spec = input(
                cb_stats,
                last ? WaitPolicy::Upfront : WaitPolicy::None,
                last ? PopPolicy::AtEnd : PopPolicy::None,
                InputTileMapping::Block,
                DataFormatReconfig::Enabled,
                TileAddressing::Offset);
            if constexpr (p == 0) {
                return std::make_tuple(CopyTile<spec, Dst::D0>{b.stats, 0});
            } else {
                return std::make_tuple(
                    CopyTile<spec, Dst::D1>{b.stats, p * num_candidates}, AddBinary<Dst::D0, Dst::D1, Dst::D0>{});
            }
        }),
        std::make_tuple(PackTile<output(cb_summed)>{b.summed}));

    // Reciprocal RMS per candidate. Only column 0 means anything.
    eltwise_chain(
        IterationShape::tiles(num_candidates),
        CopyTile<input(cb_summed, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block)>{b.summed},
        MulUnary<>{inv_hidden_size},
        AddUnary<>{eps},
        Rsqrt<>{},
        PackTile<output(cb_rr)>{b.rr});

    // Scores: every column of a candidate's statistics scaled by its column 0. An SFPU
    // product in fp32, so the mixture's math fidelity never reaches the scores.
    eltwise_chain(
        IterationShape::tiles(num_candidates),
        CopyTile<input(cb_summed, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)>{b.summed},
        UnaryBcast<
            BroadcastDim::Col,
            input(cb_rr, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            Dst::D1>{b.rr},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        PackTile<output(cb_scores)>{b.scores});

    // The shift, packed twice: compute reads it back below, and a circular buffer has a
    // single consumer. The last candidate's read waits for every score; they stay for the
    // exponentials.
    eltwise_chain_of(
        IterationShape::one_tile(),
        unroll<num_candidates>([&](auto c) {
            constexpr auto spec =
                c == num_candidates - 1 ? owning_tile_at(cb_scores, PopPolicy::None) : tile_at(cb_scores);
            if constexpr (c == 0) {
                return std::make_tuple(CopyTile<spec, Dst::D0>{b.scores, 0});
            } else {
                return std::make_tuple(CopyTile<spec, Dst::D1>{b.scores, c}, BinaryMax<Dst::D0, Dst::D1, Dst::D0>{});
            }
        }),
        std::make_tuple(PackTile<output(cb_shift)>{b.shift}, PackTile<output(cb_shift_out)>{b.shift_out}));

    // The mass, summed in fp32 dest from the exponentials: recomputing them below is a few
    // SFPU passes, where the packed ones are bf16 and drained by the reader.
    eltwise_chain_of(
        IterationShape::one_tile(),
        std::make_tuple(CopyTile<input(cb_shift, WaitPolicy::Upfront, PopPolicy::None), Dst::D1>{b.shift}),
        centred_exp<Dst::D0>(b.scores, 0),
        unroll<num_candidates - 1>([&](auto c) {
            return std::tuple_cat(
                centred_exp<Dst::D2>(b.scores, c + 1), std::make_tuple(AddBinary<Dst::D0, Dst::D2, Dst::D0>{}));
        }),
        std::make_tuple(
            PackTile<output(cb_mass_out)>{b.mass_out}, Optional<has_closed_sites, PackTile<output(cb_mass)>>{b.mass}));

    if constexpr (has_closed_sites) {
        closed_site_factor(b);
    }

    // The exponentials, in the weights' dtype; a closed site's over its mass.
    eltwise_chain(
        IterationShape::tiles(num_candidates),
        CopyTile<input(cb_shift, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{b.shift},
        CopyTile<input(cb_scores, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)>{b.scores},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Exp<>{},
        Optional<has_closed_sites, CopyTile<input(cb_factor, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D2>>{
            b.factor},
        Optional<has_closed_sites, MulBinary<Dst::D0, Dst::D2, Dst::D0>>{},
        PackTile<output(cb_exp)>{b.exp});
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
    mul_bcast_cols_init(cb_values, cb_weights, true /*acc_to_dest*/);
    reconfig_data_format(cb_values, cb_weights);
    pack_reconfig_data_format(cb_partials);

    cb_values_obj.wait_front(num_tiles * num_candidates);
    for (uint32_t group = 0; group < num_groups; ++group) {
        const uint32_t sites = sites_in_group(group, sites_per_group, num_sites);
        cb_weights_obj.wait_front(sites_per_group * num_candidates);
        for (uint32_t w = 0; w < num_tiles; ++w) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < num_candidates; ++c) {
                for (uint32_t s = 0; s < sites; ++s) {
                    mul_tiles_bcast_cols(cb_values, cb_weights, w * num_candidates + c, s * num_candidates + c, s);
                }
            }
            tile_regs_commit();
            // Odd groups go out on the reader's NoC; see attn_res_block_mix_partials.hpp.
            const bool odd = reader_writes_group(group);
            CircularBuffer& out_obj = odd ? cb_partials_odd_obj : cb_partials_obj;
            const auto out = odd ? cb_partials_odd : cb_partials;
            out_obj.reserve_back(sites_per_group);
            tile_regs_wait();
            for (uint32_t s = 0; s < sites; ++s) {
                pack_tile(s, out);
            }
            tile_regs_release();
            out_obj.push_back(sites_per_group);
        }
        cb_weights_obj.pop_front(sites_per_group * num_candidates);
    }
    cb_values_obj.pop_front(num_tiles * num_candidates);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_stats, cb_summed, cb_scores);
    ScoreBuffers buffers;
    if constexpr (has_closed_sites) {
        // Built once by the reader and held for every row.
        buffers.closed_mask.wait_front(1);
    }
    for_each_attn_res_block_mix_item([&](const AttnResBlockMixItem& item) {
        score_row(buffers);
        mix_run(item.num_tiles());
    });
}
