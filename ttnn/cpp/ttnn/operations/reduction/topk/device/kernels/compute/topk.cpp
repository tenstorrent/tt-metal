// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"

// topk llk needs a global variable atm
// this can only be removed once that's fixed
int32_t topk_replay_init = 0;

namespace NAMESPACE {

void process_tile_pair(
    uint32_t left_ind,
    uint32_t right_ind,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end,
    bool ascending,
    uint32_t m_iter,
    uint32_t K,
    uint32_t logk,
    bool target_tiles_is_one) {
    acquire_dst();

    copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
    copy_tile(input_transposed_cb_index, left_ind, input_dest_start);
    if (!target_tiles_is_one) {
        copy_tile(input_transposed_cb_index, right_ind, input_dest_end);
    }

    // unpack indices into dest
    copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
    copy_tile(index_transposed_cb_index, left_ind, index_dest_start);
    if (!target_tiles_is_one) {
        copy_tile(index_transposed_cb_index, right_ind, index_dest_end);
    }

    // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
    // sort within the larger 32 values
    ckernel::topk_rebuild(0, (uint32_t)ascending, m_iter, K, logk, target_tiles_is_one);

    // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32
    // values for topk, which was in input_dest_start
    pack_reconfig_data_format(input_transposed_cb_index);
    pack_tile<true>(input_dest_start, input_transposed_cb_index, left_ind);
    if (!target_tiles_is_one) {
        pack_tile<true>(input_dest_end, input_transposed_cb_index, right_ind);
    }

    // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32
    // values for topk, which was in index_dest_start
    pack_reconfig_data_format(index_transposed_cb_index);
    pack_tile<true>(index_dest_start, index_transposed_cb_index, left_ind);
    if (!target_tiles_is_one) {
        pack_tile<true>(index_dest_end, index_transposed_cb_index, right_ind);
    }
    release_dst();
}

void process_sequence(
    uint32_t idx,
    uint32_t tiles_per_seq,
    uint32_t target_tiles,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end,
    bool& ascending,
    bool switch_dir,
    uint32_t m_iter,
    uint32_t K,
    uint32_t logk,
    uint32_t Wt) {
    uint32_t sel_tile_id[2];
    uint32_t sel_tile_id_ptr = 0;

    for (uint32_t t = 0; t < tiles_per_seq; t++) {
        uint32_t left_ind = ((idx * (1 << (m_iter + 1)) * K) >> 5) + t;
        if (left_ind >= Wt) {
            break;
        }
        sel_tile_id[sel_tile_id_ptr] = left_ind;
        sel_tile_id_ptr++;
        if (sel_tile_id_ptr == target_tiles) {
            process_tile_pair(
                sel_tile_id[0],
                sel_tile_id[1],
                input_transposed_cb_index,
                index_transposed_cb_index,
                input_dest_start,
                input_dest_end,
                index_dest_start,
                index_dest_end,
                ascending,
                m_iter,
                K,
                logk,
                target_tiles == 1);
            sel_tile_id_ptr = 0;
            ascending = switch_dir ? !ascending : ascending;
        }
    }
}

void process_tiles(
    uint32_t m_iter,
    uint32_t K,
    uint32_t Wt,
    uint32_t num_k_sequences,
    uint32_t tiles_per_seq,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end) {
    uint32_t dist = ((1 << m_iter) * K) >> 5;
    int seq_per_2tiles = std::max((2 * 32) / K, (uint32_t)2);
    for (uint32_t i = 0; i < num_k_sequences; i += seq_per_2tiles) {
        for (uint32_t t = 0; t < tiles_per_seq; t++) {
            uint32_t left_tile_id = ((i * (1 << m_iter) * K) >> 5) + t;
            uint32_t right_tile_id = left_tile_id + dist;
            if (left_tile_id == right_tile_id) {
                right_tile_id = left_tile_id + 1;
            }

            if (left_tile_id >= Wt || right_tile_id >= Wt) {
                break;
            }

            acquire_dst();

            copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
            copy_tile(input_transposed_cb_index, left_tile_id, input_dest_start);
            copy_tile(input_transposed_cb_index, right_tile_id, input_dest_end);

            // unpack indices into dest
            copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
            copy_tile(index_transposed_cb_index, left_tile_id, index_dest_start);
            copy_tile(index_transposed_cb_index, right_tile_id, index_dest_end);

            // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
            ckernel::topk_merge(0, m_iter, K);

            // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values
            // for topk, which was in input_dest_start
            pack_reconfig_data_format(input_transposed_cb_index);
            pack_tile<true>(input_dest_start, input_transposed_cb_index, left_tile_id);
            pack_tile<true>(input_dest_end, input_transposed_cb_index, right_tile_id);

            // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values
            // for topk, which was in index_dest_start
            pack_reconfig_data_format(index_transposed_cb_index);
            pack_tile<true>(index_dest_start, index_transposed_cb_index, left_tile_id);
            pack_tile<true>(index_dest_end, index_transposed_cb_index, right_tile_id);
            release_dst();
        }
    }
}

void process_iteration(
    uint32_t m_iter,
    uint32_t K,
    uint32_t Wt,
    uint32_t num_k_sequences,
    uint32_t tiles_per_seq,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end,
    bool largest,
    bool switch_dir,
    uint32_t logk) {
    cb_wait_front(input_transposed_cb_index, Wt);
    cb_wait_front(index_transposed_cb_index, Wt);

    process_tiles(
        m_iter,
        K,
        Wt,
        num_k_sequences,
        tiles_per_seq,
        input_transposed_cb_index,
        index_transposed_cb_index,
        input_dest_start,
        input_dest_end,
        index_dest_start,
        index_dest_end);

    cb_reserve_back(input_transposed_cb_index, Wt);
    cb_reserve_back(index_transposed_cb_index, Wt);

    cb_pop_front(input_transposed_cb_index, Wt);
    cb_pop_front(index_transposed_cb_index, Wt);

    cb_push_back(input_transposed_cb_index, Wt);
    cb_push_back(index_transposed_cb_index, Wt);

    // we have decreased our search space by half
    num_k_sequences = num_k_sequences >> 1;
    int target_tiles = (Wt == 1 || ((num_k_sequences == 1) && (tiles_per_seq == 1))) ? 1 : 2;
    bool ascending = !largest;

    cb_wait_front(input_transposed_cb_index, Wt);
    cb_wait_front(index_transposed_cb_index, Wt);

    int seq_per_2tiles = std::max((2 * 32) / K, (uint32_t)2);
    for (uint32_t idx = 0; idx < num_k_sequences; idx += (seq_per_2tiles >> 1)) {
        process_sequence(
            idx,
            tiles_per_seq,
            target_tiles,
            input_transposed_cb_index,
            index_transposed_cb_index,
            input_dest_start,
            input_dest_end,
            index_dest_start,
            index_dest_end,
            ascending,
            switch_dir,
            m_iter,
            K,
            logk,
            Wt);
    }

    cb_reserve_back(input_transposed_cb_index, Wt);
    cb_reserve_back(index_transposed_cb_index, Wt);

    cb_pop_front(input_transposed_cb_index, Wt);
    cb_pop_front(index_transposed_cb_index, Wt);

    cb_push_back(input_transposed_cb_index, Wt);
    cb_push_back(index_transposed_cb_index, Wt);
}

void transpose_and_pack(
    uint32_t cb_index, uint32_t transposed_cb_index, uint32_t dest_cb_index, uint32_t Kt, uint32_t Wt) {
    reconfig_data_format_srca(transposed_cb_index);
    transpose_wh_init_short(transposed_cb_index);
    pack_reconfig_data_format(transposed_cb_index);

    cb_wait_front(transposed_cb_index, Kt);
    for (uint32_t i = 0; i < Kt; ++i) {
        acquire_dst();
        cb_reserve_back(dest_cb_index, 1);
        transpose_wh_tile(transposed_cb_index, i, 0);
        pack_tile(0, dest_cb_index);
        cb_push_back(dest_cb_index, 1);
        release_dst();
    }
    cb_wait_front(transposed_cb_index, Wt);
    cb_pop_front(transposed_cb_index, Wt);
}

void MAIN {
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t logk = get_compile_time_arg_val(9);
    constexpr uint32_t logNk = get_compile_time_arg_val(10);
    constexpr uint32_t largest = get_compile_time_arg_val(11);
    constexpr uint32_t sorted = get_compile_time_arg_val(12);

    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    constexpr uint32_t tiles_per_seq = (K >> 5) + (K % 32);

    int end_phase = (K <= 64) ? logk - 1 : 5;
    // init pack, compute and unpack

    ckernel::topk_tile_init();
    transpose_wh_init(input_cb_index, input_transposed_cb_index);

    bool switch_dir = (K == 64);
    int seq_per_2tiles = std::max((2 * 32) / K, (uint32_t)2);

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = !largest;

        cb_reserve_back(input_transposed_cb_index, Wt);
        cb_reserve_back(index_transposed_cb_index, Wt);

        // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
        for (uint32_t wt = 0; wt < Wt; wt += 2) {
            acquire_dst();
            // local sort into k groups
            cb_wait_front(input_cb_index, 2);
            cb_wait_front(index_cb_index, 2);

            reconfig_data_format_srca(input_cb_index);
            transpose_wh_init_short(input_cb_index);
            transpose_wh_tile(input_cb_index, 0, 0);
            transpose_wh_tile(input_cb_index, 1, 1);

            reconfig_data_format_srca(index_cb_index);
            transpose_wh_init_short(index_cb_index);
            transpose_wh_tile(index_cb_index, 0, 2);
            transpose_wh_tile(index_cb_index, 1, 3);

            // llk_topk_sort -> inplace
            ckernel::topk_local_sort(0, (int)ascending, end_phase);

            // pack value tiles into cb_intermed0
            pack_reconfig_data_format(input_transposed_cb_index);
            pack_tile(0, input_transposed_cb_index);
            pack_tile(1, input_transposed_cb_index);

            // pack index tiles into cb_intermed1
            pack_reconfig_data_format(index_transposed_cb_index);
            pack_tile(2, index_transposed_cb_index);
            pack_tile(3, index_transposed_cb_index);

            cb_pop_front(input_cb_index, 2);
            cb_pop_front(index_cb_index, 2);
            release_dst();
            ascending = switch_dir ? !ascending : ascending;
        }

        cb_push_back(input_transposed_cb_index, Wt);
        cb_push_back(index_transposed_cb_index, Wt);

        uint32_t num_k_sequences = (Wt * 32) / K;

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each
        // pair. second iteration we compare 0th and 2nd tile, then 4th and 6th, etc. logNk iteration we compare 0th and
        // Wt/2 tile single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logNk; ++m_iter) {
            process_iteration(
                m_iter,
                K,
                Wt,
                num_k_sequences,
                tiles_per_seq,
                input_transposed_cb_index,
                index_transposed_cb_index,
                input_dest_start,
                input_dest_end,
                index_dest_start,
                index_dest_end,
                largest,
                switch_dir,
                logk);
        }

        constexpr uint32_t Kt = K % TILE_WIDTH == 0 ? K / TILE_WIDTH : K / TILE_WIDTH + 1;

        // transpose value tiles and pack into output buffer
        transpose_and_pack(values_cb_index, input_transposed_cb_index, values_cb_index, Kt, Wt);

        // transpose index tiles and pack into output buffer
        transpose_and_pack(output_ind_cb_index, index_transposed_cb_index, output_ind_cb_index, Kt, Wt);
    }
}

}  // namespace NAMESPACE
