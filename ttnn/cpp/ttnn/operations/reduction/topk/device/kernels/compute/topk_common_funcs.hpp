// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

namespace NAMESPACE {
void process_and_sort_tiles(
    uint32_t input_cb_index,
    uint32_t index_cb_index,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t Wt,
    bool switch_dir,
    bool& ascending,
    int end_phase) {
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
}

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

            acquire_dst();

            copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
            copy_tile(input_transposed_cb_index, left_tile_id, input_dest_start);
            copy_tile(input_transposed_cb_index, right_tile_id, input_dest_end);

            // unpack indices into dest
            copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
            copy_tile(index_transposed_cb_index, left_tile_id, index_dest_start);
            copy_tile(index_transposed_cb_index, right_tile_id, index_dest_end);

            // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
            if (largest) {
                ckernel::topk_merge<false>(0, m_iter, K);
            } else {
                ckernel::topk_merge<true>(0, m_iter, K);
            }

            // ckernel::topk_merge(0, m_iter, K);

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
    uint32_t& num_k_sequences,
    uint32_t tiles_per_seq,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t input_dest_start,
    uint32_t input_dest_end,
    uint32_t index_dest_start,
    uint32_t index_dest_end,
    bool largest,
    bool switch_dir,
    uint32_t logk,
    int& seq_per_2tiles,
    bool largest_param) {
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
        index_dest_end,
        largest_param,
        seq_per_2tiles);

    cb_reserve_back(input_transposed_cb_index, Wt);
    cb_reserve_back(index_transposed_cb_index, Wt);

    cb_pop_front(input_transposed_cb_index, Wt);
    cb_pop_front(index_transposed_cb_index, Wt);

    cb_push_back(input_transposed_cb_index, Wt);
    cb_push_back(index_transposed_cb_index, Wt);

    // we have decreased our search space by half
    num_k_sequences = num_k_sequences >> 1;
    int target_tiles = (Wt == 1 || ((num_k_sequences == 1) && (tiles_per_seq == 1))) ? 1 : 2;

    int sel_tile_id[2];
    int sel_tile_id_ptr = 0;
    seq_per_2tiles = (seq_per_2tiles == 2) ? 2 : seq_per_2tiles >> 1;
    bool ascending = !largest;

    cb_wait_front(input_transposed_cb_index, Wt);
    cb_wait_front(index_transposed_cb_index, Wt);

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

    cb_reserve_back(input_transposed_cb_index, Wt);
    cb_reserve_back(index_transposed_cb_index, Wt);

    cb_pop_front(input_transposed_cb_index, Wt);
    cb_pop_front(index_transposed_cb_index, Wt);

    cb_push_back(input_transposed_cb_index, Wt);
    cb_push_back(index_transposed_cb_index, Wt);
}

void transpose_and_pack(uint32_t transposed_cb_index, uint32_t dest_cb_index, uint32_t Kt, uint32_t Wt) {
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
}  // namespace NAMESPACE
