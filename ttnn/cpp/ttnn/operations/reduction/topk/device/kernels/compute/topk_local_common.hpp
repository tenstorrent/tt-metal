// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

#include "topk_common_funcs.hpp"

// Compile time args 0 to 15 and the per row local sort of topk_local.cpp and topk_local_tree_merge.cpp.
namespace topk_local {

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

// Fused-key stable mode: sort packed [bf16|u16] keys with the unstable network instead of
// running the comparator-stable network on separate value/index tiles.
constexpr bool fused_keys = get_compile_time_arg_val(15) == 1;
// The packed key IS the stable tie-break; the network itself runs unstable in fused mode.
constexpr bool network_stable = stable_sort && !fused_keys;
constexpr auto tie_order = ckernel::topk_tie_order_from_global_direction(largest != 0);

// Dest registers: values in 0 and 1, indices in 2 and 3.
constexpr std::uint32_t input_dest_start = 0;
constexpr std::uint32_t input_dest_end = 1;
constexpr std::uint32_t index_dest_start = 2;
constexpr std::uint32_t index_dest_end = 3;
constexpr std::uint32_t tiles_per_seq = (K + 31) / 32;

// Supports K only up to 64
constexpr int end_phase = (K <= 64) ? logk - 1 : 5;
constexpr bool switch_dir = (K == 64);
// process_iteration halves the seq_per_2tiles it gets.
constexpr std::uint32_t initial_seq_per_2tiles = std::max<std::uint32_t>((2 * 32) / K, 2);

ALWI void init() {
    compute_kernel_hw_startup(input_dfb_index, index_dfb_index, input_transposed_dfb_index);
    ckernel::topk_tile_init<fused_keys>();
}

// Sorts one row of Wt tiles into the transposed CBs; their first Kt tiles then hold the row's top k.
ALWI void sort_row(std::uint32_t direction_init, std::uint32_t& seq_per_2tiles) {
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
}

// Packs in the destination format (the source CB may be a wider intermediate), then pops src_pop source tiles.
ALWI void copy_front_tiles(
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

// Hands the row's top Kt tiles to the writer in the output formats and frees the transposed CBs.
ALWI void pack_row_top_k() {
    copy_front_tiles(input_transposed_dfb_index, values_dfb_index, /*num_tiles*/ Kt, /*src_pop*/ Wt);
    if constexpr (!fused_keys) {
        // In fused mode the indices ride inside the packed value tiles; there is no index stream.
        copy_front_tiles(index_transposed_dfb_index, output_ind_dfb_index, /*num_tiles*/ Kt, /*src_pop*/ Wt);
    }
}

}  // namespace topk_local
