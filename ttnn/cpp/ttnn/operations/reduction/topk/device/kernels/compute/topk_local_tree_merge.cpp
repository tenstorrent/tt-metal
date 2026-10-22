// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "topk_local_common.hpp"

// topk_local.cpp plus a tree merge: in round r core i (i % 2^(r+1) == 0) merges core i + 2^r's top Kt tiles.

void kernel_main() {
    using namespace topk_local;

    constexpr std::uint32_t merge_tiles = 2 * Kt;  // own Kt tiles, then the partner's
    // The writer lands [own Kt | partner Kt] tiles in the landing CBs; the merge CBs are the in-place workspace.
    constexpr std::uint32_t landing_values_dfb_index = get_compile_time_arg_val(16);
    constexpr std::uint32_t landing_indices_dfb_index = get_compile_time_arg_val(17);
    constexpr std::uint32_t merge_values_dfb_index = get_compile_time_arg_val(18);
    constexpr std::uint32_t merge_indices_dfb_index = get_compile_time_arg_val(19);

    std::uint32_t direction_init = get_arg_val<std::uint32_t>(0);
    const std::uint32_t core_id = get_arg_val<std::uint32_t>(1);  // Index among the local cores
    const std::uint32_t num_recv_rounds = get_arg_val<std::uint32_t>(2);

    init();

    uint32_t seq_per_2tiles = initial_seq_per_2tiles;

    for (std::uint32_t ht = 0; ht < Ht; ++ht) {
        sort_row(direction_init, seq_per_2tiles);
        pack_row_top_k();

        for (std::uint32_t r = 0; r < num_recv_rounds; ++r) {
            copy_front_tiles(
                landing_values_dfb_index, merge_values_dfb_index, /*num_tiles*/ merge_tiles, /*src_pop*/ merge_tiles);
            if constexpr (!fused_keys) {
                copy_front_tiles(
                    landing_indices_dfb_index,
                    merge_indices_dfb_index,
                    /*num_tiles*/ merge_tiles,
                    /*src_pop*/ merge_tiles);
            }

            // Partners of the next round must be sorted in opposite directions, like direction_init does locally.
            const bool merge_ascending = (largest == 0) != (((core_id >> (r + 1)) & 1) == 1);
            std::uint32_t merge_num_k_sequences = (merge_tiles * 32) / K;
            uint32_t merge_seq_per_2tiles = initial_seq_per_2tiles;
            process_iteration<network_stable, fused_keys, tie_order>(
                0,  // Single merge step: at m_iter 0 tile t is paired with tile t + Kt
                K,
                merge_tiles,
                merge_num_k_sequences,
                tiles_per_seq,
                merge_values_dfb_index,
                merge_indices_dfb_index,
                input_dest_start,
                input_dest_end,
                index_dest_start,
                index_dest_end,
                !merge_ascending,  // process_iteration rebuilds the kept sequence with ascending = !largest
                switch_dir,
                logk,
                merge_seq_per_2tiles,
                largest);

            copy_front_tiles(merge_values_dfb_index, values_dfb_index, /*num_tiles*/ Kt, /*src_pop*/ merge_tiles);
            if constexpr (!fused_keys) {
                copy_front_tiles(
                    merge_indices_dfb_index, output_ind_dfb_index, /*num_tiles*/ Kt, /*src_pop*/ merge_tiles);
            }
        }
    }
}
