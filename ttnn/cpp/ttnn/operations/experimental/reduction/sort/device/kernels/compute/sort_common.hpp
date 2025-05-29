// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

namespace NAMESPACE {

void sort_Wt_tiles_row_to_bitonic_sequence(
    const uint32_t input_cb_index,
    const uint32_t index_cb_index,
    const uint32_t input_transposed_cb_index,
    const uint32_t index_transposed_cb_index,
    const uint32_t Wt,
    const bool switch_dir,
    const bool ascending,
    const int end_phase) {
    cb_reserve_back(input_transposed_cb_index, Wt);
    cb_reserve_back(index_transposed_cb_index, Wt);

    bool ascending_local = ascending;
    for (uint32_t wt = 0; wt < Wt; wt += 2) {
        tile_regs_acquire();

        cb_wait_front(input_cb_index, 2);
        cb_wait_front(index_cb_index, 2);

        // topk_local_sort sorts by columns - transpose input tiles for sorting
        reconfig_data_format_srca(input_cb_index);
        transpose_wh_init_short(input_cb_index);
        transpose_wh_tile(input_cb_index, 0, 0);
        transpose_wh_tile(input_cb_index, 1, 1);

        reconfig_data_format_srca(index_cb_index);
        transpose_wh_init_short(index_cb_index);
        transpose_wh_tile(index_cb_index, 0, 2);
        transpose_wh_tile(index_cb_index, 1, 3);

        // llk_topk_sort -> inplace
        ckernel::topk_local_sort(0, (int)ascending_local, end_phase);

        tile_regs_commit();
        tile_regs_wait();

        // pack value tiles into transposed buffer
        pack_reconfig_data_format(input_transposed_cb_index);
        pack_tile(0, input_transposed_cb_index);
        pack_tile(1, input_transposed_cb_index);

        // pack index tiles into index transposed buffer
        pack_reconfig_data_format(index_transposed_cb_index);
        pack_tile(2, index_transposed_cb_index);
        pack_tile(3, index_transposed_cb_index);
        cb_pop_front(input_cb_index, 2);
        cb_pop_front(index_cb_index, 2);

        tile_regs_release();

        // Switch sorting direction for bitonic merge sort
        ascending_local = switch_dir ? !ascending_local : ascending_local;
    }

    cb_push_back(input_transposed_cb_index, Wt);
    cb_push_back(index_transposed_cb_index, Wt);
}

void transpose_and_pack(uint32_t transposed_cb_index, uint32_t dest_cb_index, uint32_t Wt) {
    constexpr uint32_t one_tile = 1;

    // Transpose from sorting by column to right structure
    reconfig_data_format_srca(transposed_cb_index);
    transpose_wh_init_short(transposed_cb_index);
    pack_reconfig_data_format(transposed_cb_index);

    cb_wait_front(transposed_cb_index, Wt);

    for (uint32_t i = 0; i < Wt; ++i) {
        tile_regs_acquire();

        cb_reserve_back(dest_cb_index, one_tile);
        transpose_wh_tile(transposed_cb_index, i, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dest_cb_index);
        cb_push_back(dest_cb_index, one_tile);

        tile_regs_release();
    }

    cb_wait_front(transposed_cb_index, Wt);
    cb_pop_front(transposed_cb_index, Wt);
}

}  // namespace NAMESPACE
