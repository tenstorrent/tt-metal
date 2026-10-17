// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

#include "topk_common_funcs.hpp"

// topk_local.cpp plus a tree merge: in round r core i (i % 2^(r+1) == 0) merges core i + 2^r's top Kt tiles.

// Packs in the destination format (the transposed CBs may be a wider intermediate), then pops src_pop source tiles.
void copy_front_tiles(
    std::uint32_t src_dfb_index, std::uint32_t dst_dfb_index, std::uint32_t num_tiles, std::uint32_t src_pop) {
    DataflowBuffer src_dfb(static_cast<uint16_t>(src_dfb_index));
    DataflowBuffer dst_dfb(static_cast<uint16_t>(dst_dfb_index));

    reconfig_data_format_srca(src_dfb_index);
    copy_init(src_dfb_index);
    pack_reconfig_data_format(dst_dfb_index);

    src_dfb.wait_front(static_cast<uint16_t>(num_tiles));
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(src_dfb_index, i, 0);
        tile_regs_commit();

        dst_dfb.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, dst_dfb_index);
        tile_regs_release();

        dst_dfb.push_back(1);
    }
    src_dfb.wait_front(static_cast<uint16_t>(src_pop));
    src_dfb.pop_front(static_cast<uint16_t>(src_pop));
}

void kernel_main() {
    constexpr std::uint32_t input_dfb_index = get_compile_time_arg_val(0);
    constexpr std::uint32_t index_dfb_index = get_compile_time_arg_val(1);
    constexpr std::uint32_t input_transposed_dfb_index = get_compile_time_arg_val(2);
    constexpr std::uint32_t index_transposed_dfb_index = get_compile_time_arg_val(3);
    constexpr std::uint32_t values_dfb_index = get_compile_time_arg_val(4);
    constexpr std::uint32_t output_ind_dfb_index = get_compile_time_arg_val(5);
    constexpr std::uint32_t Ht = get_compile_time_arg_val(6);
    constexpr std::uint32_t Wt = get_compile_time_arg_val(7);
    constexpr std::uint32_t K = get_compile_time_arg_val(8);
    constexpr std::uint32_t Kt = get_compile_time_arg_val(9);
    constexpr std::uint32_t logk = get_compile_time_arg_val(10);
    constexpr std::uint32_t logWt = get_compile_time_arg_val(11);
    constexpr std::uint32_t largest = get_compile_time_arg_val(12);
    constexpr bool stable_sort = get_compile_time_arg_val(14) == 1;  // Ties keep the lowest index

    // Fused-key stable mode: packed [bf16|u16] keys instead of separate value and index tiles.
    constexpr bool fused_keys = get_compile_time_arg_val(15) == 1;
    // The packed key IS the stable tie-break; the network itself runs unstable in fused mode.
    constexpr bool network_stable = stable_sort && !fused_keys;
    // The writer lands [own Kt | partner Kt] tiles in the landing CBs; the merge CBs are the in-place workspace.
    constexpr std::uint32_t landing_values_dfb_index = get_compile_time_arg_val(16);
    constexpr std::uint32_t landing_indices_dfb_index = get_compile_time_arg_val(17);
    constexpr std::uint32_t merge_values_dfb_index = get_compile_time_arg_val(18);
    constexpr std::uint32_t merge_indices_dfb_index = get_compile_time_arg_val(19);

    std::uint32_t direction_init = get_arg_val<std::uint32_t>(0);
    const std::uint32_t core_id = get_arg_val<std::uint32_t>(1);  // Index among the local cores
    const std::uint32_t num_recv_rounds = get_arg_val<std::uint32_t>(2);

    constexpr std::uint32_t input_dest_start = 0;
    constexpr std::uint32_t index_dest_start = 2;
    constexpr std::uint32_t input_dest_end = 1;
    constexpr std::uint32_t index_dest_end = 3;
    constexpr std::uint32_t tiles_per_seq = (K + 31) / 32;

    // Supports K only up to 64
    const int end_phase = (K <= 64) ? logk - 1 : 5;

    compute_kernel_hw_startup(input_dfb_index, index_dfb_index, input_transposed_dfb_index);
    ckernel::topk_tile_init<fused_keys>();
    constexpr auto tie_order = ckernel::topk_tie_order_from_global_direction(largest != 0);

    const bool switch_dir = (K == 64);
    // process_iteration halves the seq_per_2tiles it gets, so each tree merge starts again from this value.
    constexpr uint32_t initial_seq_per_2tiles = std::max<uint32_t>((2 * 32) / K, 2);
    uint32_t seq_per_2tiles = initial_seq_per_2tiles;

    for (std::uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = !largest;

        process_and_sort_tiles<network_stable, fused_keys, largest != 0, tie_order>(
            input_dfb_index,
            index_dfb_index,
            input_transposed_dfb_index,
            index_transposed_dfb_index,
            Wt,
            switch_dir,
            ascending,
            end_phase);

        std::uint32_t num_k_sequences = (Wt * 32) / K;

        // log(Wt) merge iterations; iteration n pairs tiles at distance 2^n.
        for (std::uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            process_iteration<network_stable, fused_keys, tie_order>(
                m_iter,
                K,
                Wt,
                num_k_sequences,
                tiles_per_seq,
                input_transposed_dfb_index,
                index_transposed_dfb_index,
                input_dest_start,
                input_dest_end,
                index_dest_start,
                index_dest_end,
                !direction_init,
                switch_dir,
                logk,
                seq_per_2tiles,
                largest);
        }

        // The first Kt tiles of the transposed buffers hold the local top k; repack them in the output formats.
        copy_front_tiles(input_transposed_dfb_index, values_dfb_index, Kt, Wt);
        if constexpr (!fused_keys) {
            // In fused mode the indices ride inside the packed value tiles; there is no index stream.
            copy_front_tiles(index_transposed_dfb_index, output_ind_dfb_index, Kt, Wt);
        }

        for (std::uint32_t r = 0; r < num_recv_rounds; ++r) {
            copy_front_tiles(landing_values_dfb_index, merge_values_dfb_index, 2 * Kt, 2 * Kt);
            if constexpr (!fused_keys) {
                copy_front_tiles(landing_indices_dfb_index, merge_indices_dfb_index, 2 * Kt, 2 * Kt);
            }

            // Partners of the next round must be sorted in opposite directions, like direction_init does locally.
            const bool merge_ascending = (largest == 0) != (((core_id >> (r + 1)) & 1) == 1);
            std::uint32_t merge_num_k_sequences = (2 * Kt * 32) / K;
            uint32_t merge_seq_per_2tiles = initial_seq_per_2tiles;
            process_iteration<network_stable, fused_keys, tie_order>(
                0,  // Single merge step: at m_iter 0 tile t is paired with tile t + Kt
                K,
                2 * Kt,  // Own Kt tiles followed by the partner's Kt tiles
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

            copy_front_tiles(merge_values_dfb_index, values_dfb_index, Kt, 2 * Kt);
            if constexpr (!fused_keys) {
                copy_front_tiles(merge_indices_dfb_index, output_ind_dfb_index, Kt, 2 * Kt);
            }
        }
    }
}
