// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weighted reduction over the candidate axis for a group of sites at once, one
// pass over the input per group:
//   dst[s] += input_tile[c] * weight_col[s][c]   for c in [0, num_candidates)
//
// The multiply and the add are the same instruction: `mul_bcast_cols_init` with
// acc_to_dest turns every `mul_tiles_bcast_cols` into a MAC against its destination.
// That is what buys the fusion: no intermediate DFB, no second read of the input.
//
// The accumulators start at zero without being cleared here. The whole of DEST
// is zeroed once before `kernel_main`, and the packer zeroes the section it just
// drained on every `tile_regs_release()` — all of it, not the tiles that were
// packed — so a section is clean whichever of the G tiles the previous position
// wrote (`_llk_pack_dest_section_done_`).
//
// One DEST tile per site is what makes the input read amortize: a candidate tile
// is unpacked once and MAC'd into every accumulator in the group, so the group
// costs the same DRAM traffic as a single site used to.
//
// The weight tile holds its scalar in column 0, which is what BroadcastType::COL
// reads, so a [R, C, H, 1] tile needs no pre-pass to get into this form.

#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

using namespace ckernel;

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_candidates = get_arg(args::num_candidates);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t num_sites = get_arg(args::num_sites);
    constexpr uint32_t sites_per_group = get_arg(args::sites_per_group);
    constexpr uint32_t num_groups = get_arg(args::num_groups);

    // runtime args
    const uint32_t num_positions = get_arg(args::num_positions);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out0(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    mul_bcast_cols_init(dfb::in0, dfb::in1, true /*acc_to_dest*/);
    reconfig_data_format(dfb::in0, dfb::in1);

    for (uint32_t group = 0; group < num_groups; ++group) {
        const uint32_t first_site = group * sites_per_group;
        const uint32_t sites_in_group =
            (first_site + sites_per_group <= num_sites) ? sites_per_group : num_sites - first_site;
        const uint32_t weights_per_set = sites_in_group * num_candidates;

        // One weight set per token row, shared by the Wt positions in that row.
        // The reader turns the set over on the same test — `i % Wt == 0` over the
        // global position index — so the two stay in step without a semaphore.
        uint32_t width_index = start_id % Wt;
        dfb_in1.wait_front(weights_per_set);

        for (uint32_t i = 0; i < num_positions; ++i) {
            if (i != 0 && width_index == 0) {
                dfb_in1.pop_front(weights_per_set);
                dfb_in1.wait_front(weights_per_set);
            }

            dfb_in0.wait_front(num_candidates);
            tile_regs_acquire();
            for (uint32_t c = 0; c < num_candidates; ++c) {
                // Candidate outermost: the unpacker holds one input tile while it
                // is MAC'd across the group, which is the reuse the grouping is
                // for. The weight set is laid out site-major, so a site's
                // candidate c sits at s * num_candidates + c.
                for (uint32_t s = 0; s < sites_in_group; ++s) {
                    mul_tiles_bcast_cols(dfb::in0, dfb::in1, c, s * num_candidates + c, s);
                }
            }
            tile_regs_commit();
            dfb_in0.pop_front(num_candidates);

            dfb_out0.reserve_back(sites_in_group);
            pack_reconfig_data_format(dfb::out);
            tile_regs_wait();
            for (uint32_t s = 0; s < sites_in_group; ++s) {
                pack_tile(s, dfb::out);
            }
            tile_regs_release();
            dfb_out0.push_back(sites_in_group);

            ++width_index;
            if (width_index == Wt) {
                width_index = 0;
            }
        }

        // Leave the DFB balanced: the last row's set was waited on but never turned over.
        dfb_in1.pop_front(weights_per_set);
    }
}
