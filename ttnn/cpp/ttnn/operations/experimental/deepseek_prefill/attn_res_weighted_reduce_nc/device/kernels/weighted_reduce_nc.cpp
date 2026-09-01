// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weighted reduction over the candidate axis for a group of sites at once, one
// pass over the input per group:
//   dst[s] += input_tile[c] * weight_col[s][c]   for c in [0, num_candidates)
//
// The multiply and the add are the same instruction. `compute_kernel_hw_startup`
// and `bcast_init` set up PACK + UNPACK + hw_configure for ELWMUL with a column
// broadcast; overriding the MATH init with acc_to_dest=1 turns every
// `mul_tiles_bcast_cols` into a MAC against its destination. That is what buys the
// fusion: no intermediate CB, no second read of the input.
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
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_sites = get_compile_time_arg_val(2);
    constexpr uint32_t sites_per_group = get_compile_time_arg_val(3);
    constexpr uint32_t num_groups = get_compile_time_arg_val(4);

    // runtime args
    const uint32_t num_positions = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    CircularBuffer cb_in0_obj(cb_in0);
    CircularBuffer cb_in1_obj(cb_in1);
    CircularBuffer cb_out0_obj(cb_out0);

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_in0, cb_in1);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_in0, cb_in1, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_in0, cb_in1);

    for (uint32_t group = 0; group < num_groups; ++group) {
        const uint32_t first_site = group * sites_per_group;
        const uint32_t sites_in_group =
            (first_site + sites_per_group <= num_sites) ? sites_per_group : num_sites - first_site;
        const uint32_t weights_per_set = sites_in_group * num_candidates;

        // One weight set per token row, shared by the Wt positions in that row.
        // The reader turns the set over on the same test — `i % Wt == 0` over the
        // global position index — so the two stay in step without a semaphore.
        uint32_t width_index = start_id % Wt;
        cb_in1_obj.wait_front(weights_per_set);

        for (uint32_t i = 0; i < num_positions; ++i) {
            if (i != 0 && width_index == 0) {
                cb_in1_obj.pop_front(weights_per_set);
                cb_in1_obj.wait_front(weights_per_set);
            }

            cb_in0_obj.wait_front(num_candidates);
            tile_regs_acquire();
            for (uint32_t c = 0; c < num_candidates; ++c) {
                // Candidate outermost: the unpacker holds one input tile while it
                // is MAC'd across the group, which is the reuse the grouping is
                // for. The weight set is laid out site-major, so a site's
                // candidate c sits at s * num_candidates + c.
                for (uint32_t s = 0; s < sites_in_group; ++s) {
                    mul_tiles_bcast_cols(cb_in0, cb_in1, c, s * num_candidates + c, s);
                }
            }
            tile_regs_commit();
            cb_in0_obj.pop_front(num_candidates);

            cb_out0_obj.reserve_back(sites_in_group);
            pack_reconfig_data_format(cb_out0);
            tile_regs_wait();
            for (uint32_t s = 0; s < sites_in_group; ++s) {
                pack_tile(s, cb_out0);
            }
            tile_regs_release();
            cb_out0_obj.push_back(sites_in_group);

            ++width_index;
            if (width_index == Wt) {
                width_index = 0;
            }
        }

        // Leave the CB balanced: the last row's set was waited on but never turned over.
        cb_in1_obj.pop_front(weights_per_set);
    }
}
