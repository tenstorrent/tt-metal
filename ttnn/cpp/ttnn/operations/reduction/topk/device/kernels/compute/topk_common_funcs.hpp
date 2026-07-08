// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"

void process_and_sort_tiles(
    uint32_t input_dfb_index,
    uint32_t index_dfb_index,
    uint32_t input_transposed_dfb_index,
    uint32_t index_transposed_dfb_index,
    uint32_t Wt,
    bool switch_dir,
    bool& ascending,
    int end_phase) {
    DataflowBuffer input_dfb(input_dfb_index);
    DataflowBuffer index_dfb(index_dfb_index);
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);

    input_transposed_dfb.reserve_back(Wt);
    index_transposed_dfb.reserve_back(Wt);

    // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
    for (uint32_t wt = 0; wt < Wt; wt += 2) {
        // local sort into k groups
        // for the last iteration, we only need to wait for 1 tile if Wt is odd, otherwise we wait for 2 tiles
        uint32_t tiles_to_wait = ((Wt % 2 != 0) && (wt + 2 > Wt)) ? 1 : 2;
        input_dfb.wait_front(tiles_to_wait);
        index_dfb.wait_front(tiles_to_wait);

        tile_regs_acquire();
        reconfig_data_format_srca(input_dfb_index);
        transpose_init(input_dfb_index);
        transpose_tile(input_dfb_index, 0, 0);
        if (tiles_to_wait == 2) {
            transpose_tile(input_dfb_index, 1, 1);
        }
        reconfig_data_format_srca(index_dfb_index);
        transpose_init(index_dfb_index);
        transpose_tile(index_dfb_index, 0, 2);
        if (tiles_to_wait == 2) {
            transpose_tile(index_dfb_index, 1, 3);
        }
        // llk_topk_sort -> inplace
        ckernel::topk_local_sort(0, (int)ascending, end_phase);
        tile_regs_commit();

        input_dfb.pop_front(tiles_to_wait);
        index_dfb.pop_front(tiles_to_wait);

        tile_regs_wait();
        // pack value tiles into cb_intermed0
        pack_reconfig_data_format(input_transposed_dfb_index);
        pack_tile(0, input_transposed_dfb_index);
        if (tiles_to_wait == 2) {
            pack_tile(1, input_transposed_dfb_index);
        }
        // pack index tiles into cb_intermed1
        pack_reconfig_data_format(index_transposed_dfb_index);
        pack_tile(2, index_transposed_dfb_index);
        if (tiles_to_wait == 2) {
            pack_tile(3, index_transposed_dfb_index);
        }
        tile_regs_release();
        ascending = switch_dir ? !ascending : ascending;
    }

    input_transposed_dfb.push_back(Wt);
    index_transposed_dfb.push_back(Wt);
}

void process_tile_pair(
    uint32_t left_ind,
    uint32_t right_ind,
    uint32_t input_transposed_dfb_index,
    uint32_t index_transposed_dfb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end,
    bool ascending,
    uint32_t m_iter,
    uint32_t K,
    uint32_t logk,
    bool target_tiles_is_one) {
    tile_regs_acquire();

    copy_tile_to_dst_init_short_with_dt(index_transposed_dfb_index, input_transposed_dfb_index);
    copy_tile(input_transposed_dfb_index, left_ind, input_dest_start);
    if (!target_tiles_is_one) {
        copy_tile(input_transposed_dfb_index, right_ind, input_dest_end);
    }

    // unpack indices into dest
    copy_tile_to_dst_init_short_with_dt(input_transposed_dfb_index, index_transposed_dfb_index);
    copy_tile(index_transposed_dfb_index, left_ind, index_dest_start);
    if (!target_tiles_is_one) {
        copy_tile(index_transposed_dfb_index, right_ind, index_dest_end);
    }

    // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
    // sort within the larger 32 values
    ckernel::topk_rebuild(0, (uint32_t)ascending, m_iter, K, logk, target_tiles_is_one);

    tile_regs_commit();
    tile_regs_wait();
    // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32
    // values for topk, which was in input_dest_start
    pack_reconfig_data_format(input_transposed_dfb_index);
    pack_tile<true>(input_dest_start, input_transposed_dfb_index, left_ind);
    if (!target_tiles_is_one) {
        pack_tile<true>(input_dest_end, input_transposed_dfb_index, right_ind);
    }

    // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32
    // values for topk, which was in index_dest_start
    pack_reconfig_data_format(index_transposed_dfb_index);
    pack_tile<true>(index_dest_start, index_transposed_dfb_index, left_ind);
    if (!target_tiles_is_one) {
        pack_tile<true>(index_dest_end, index_transposed_dfb_index, right_ind);
    }
    tile_regs_release();
}

void process_tiles(
    uint32_t m_iter,
    uint32_t K,
    uint32_t Wt,
    uint32_t num_k_sequences,
    uint32_t tiles_per_seq,
    uint32_t input_transposed_dfb_index,
    uint32_t index_transposed_dfb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end,
    bool largest,
    int seq_per_2tiles) {
    uint32_t dist = ((1 << m_iter) * K) >> 5;
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

            tile_regs_acquire();

            copy_tile_to_dst_init_short_with_dt(index_transposed_dfb_index, input_transposed_dfb_index);
            copy_tile(input_transposed_dfb_index, left_tile_id, input_dest_start);
            copy_tile(input_transposed_dfb_index, right_tile_id, input_dest_end);

            // unpack indices into dest
            copy_tile_to_dst_init_short_with_dt(input_transposed_dfb_index, index_transposed_dfb_index);
            copy_tile(index_transposed_dfb_index, left_tile_id, index_dest_start);
            copy_tile(index_transposed_dfb_index, right_tile_id, index_dest_end);

            // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
            if (largest) {
                ckernel::topk_merge<false>(0, m_iter, K);
            } else {
                ckernel::topk_merge<true>(0, m_iter, K);
            }

            // ckernel::topk_merge(0, m_iter, K);

            tile_regs_commit();
            tile_regs_wait();
            // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values
            // for topk, which was in input_dest_start
            pack_reconfig_data_format(input_transposed_dfb_index);
            pack_tile<true>(input_dest_start, input_transposed_dfb_index, left_tile_id);
            pack_tile<true>(input_dest_end, input_transposed_dfb_index, right_tile_id);

            // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values
            // for topk, which was in index_dest_start
            pack_reconfig_data_format(index_transposed_dfb_index);
            pack_tile<true>(index_dest_start, index_transposed_dfb_index, left_tile_id);
            pack_tile<true>(index_dest_end, index_transposed_dfb_index, right_tile_id);
            tile_regs_release();
        }
    }
}

void process_iteration(
    uint32_t m_iter,
    uint32_t K,
    uint32_t Wt,
    uint32_t& num_k_sequences,
    uint32_t tiles_per_seq,
    uint32_t input_transposed_dfb_index,
    uint32_t index_transposed_dfb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end,
    bool largest,
    bool switch_dir,
    uint32_t logk,
    int& seq_per_2tiles,
    bool largest_param) {
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);

    input_transposed_dfb.wait_front(Wt);
    index_transposed_dfb.wait_front(Wt);

    process_tiles(
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
        largest_param,
        seq_per_2tiles);

    input_transposed_dfb.reserve_back(Wt);
    index_transposed_dfb.reserve_back(Wt);

    input_transposed_dfb.pop_front(Wt);
    index_transposed_dfb.pop_front(Wt);

    input_transposed_dfb.push_back(Wt);
    index_transposed_dfb.push_back(Wt);

    // we have decreased our search space by half
    num_k_sequences = num_k_sequences >> 1;
    int target_tiles = (Wt == 1 || ((num_k_sequences == 1) && (tiles_per_seq == 1))) ? 1 : 2;

    int sel_tile_id[2];
    int sel_tile_id_ptr = 0;
    seq_per_2tiles = (seq_per_2tiles == 2) ? 2 : seq_per_2tiles >> 1;
    bool ascending = !largest;

    input_transposed_dfb.wait_front(Wt);
    index_transposed_dfb.wait_front(Wt);

    for (uint32_t idx = 0; idx < num_k_sequences; idx += (seq_per_2tiles >> 1)) {
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
                    input_transposed_dfb_index,
                    index_transposed_dfb_index,
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

    input_transposed_dfb.reserve_back(Wt);
    index_transposed_dfb.reserve_back(Wt);

    input_transposed_dfb.pop_front(Wt);
    index_transposed_dfb.pop_front(Wt);

    input_transposed_dfb.push_back(Wt);
    index_transposed_dfb.push_back(Wt);
}

void transpose_and_pack(uint32_t transposed_dfb_index, uint32_t dest_dfb_index, uint32_t Kt, uint32_t Wt) {
    DataflowBuffer transposed_dfb(transposed_dfb_index);
    DataflowBuffer dest_dfb(dest_dfb_index);

    reconfig_data_format_srca(transposed_dfb_index);
    transpose_init(transposed_dfb_index);
    // Pack using the DESTINATION CB format: transposed_dfb may be bf16 (higher-precision
    // intermediate) while dest_dfb is the original bfp8/bfp4 output format.
    pack_reconfig_data_format(dest_dfb_index);

    transposed_dfb.wait_front(Kt);
    for (uint32_t i = 0; i < Kt; ++i) {
        tile_regs_acquire();
        transpose_tile(transposed_dfb_index, i, 0);
        tile_regs_commit();

        dest_dfb.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, dest_dfb_index);
        tile_regs_release();

        dest_dfb.push_back(1);
    }
    transposed_dfb.wait_front(Wt);
    transposed_dfb.pop_front(Wt);
}
