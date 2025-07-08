// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

namespace NAMESPACE {

/**
 * @brief Sorts Wt tiles from row-major order into a bitonic sequence using local sorting and transposition.
 *
 * This function processes tiles in pairs, transposes them for column-wise sorting,
 * performs an in-place local sort (bitonic sort phase), and then packs the sorted
 * and transposed tiles into output buffers. The sorting direction can be switched
 * between pairs to facilitate bitonic merge sort.
 *
 * @param input_cb_index Index of the input circular buffer containing value tiles.
 * @param index_cb_index Index of the input circular buffer containing index tiles.
 * @param input_transposed_cb_index Index of the output circular buffer for transposed value tiles.
 * @param index_transposed_cb_index Index of the output circular buffer for transposed index tiles.
 * @param Wt Number of tiles to process (should be even).
 * @param switch_dir If true, alternates the sorting direction for each tile pair to build a bitonic sequence.
 * @param ascending Initial sorting direction: true for ascending, false for descending.
 * @param end_phase Indicates the current phase of the bitonic sort (used by the local sort kernel).
 *
 * The function assumes that the input and index buffers contain at least Wt tiles,
 * and that Wt is a multiple of 2. It reserves space in the output buffers, processes
 * tiles in pairs, and pushes the results to the output buffers upon completion.
 */
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

/**
 * @brief Transposes and packs tiles from a source circular buffer to a destination circular buffer.
 *
 * This function performs the following steps:
 * 1. Reconfigures the data format of the source buffer for transposition.
 * 2. Initializes the transposition operation.
 * 3. Waits until the required number of tiles (Wt) are available in the source buffer.
 * 4. Iterates over each tile, performing:
 *    - Acquisition of tile registers.
 *    - Reservation of space in the destination buffer.
 *    - Transposition of the current tile from the source buffer.
 *    - Packing of the transposed tile into the destination buffer.
 *    - Committing and releasing tile registers.
 *    - Pushing the packed tile to the destination buffer.
 * 5. After processing all tiles, waits for the source buffer and pops the processed tiles.
 *
 * @param transposed_cb_index Index of the source circular buffer containing tiles to be transposed.
 * @param dest_cb_index Index of the destination circular buffer where packed tiles will be stored.
 * @param Wt Number of tiles to process (width in tiles).
 */
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

/**
 * @brief Helper function to manage copy_tile_to_dst_init_short_with_dt() calls.
 *
 * This function prepares the destination buffer for a new tile copy operation by
 * invoking a helper function to handle the initialization with the appropriate data type.
 * It then updates the global buffer index to reflect the new buffer being used.
 *
 * @param new_cb The index of the new circular buffer to which the tile will be copied.
 * @param global_old_cb Reference to the current global circular buffer index; will be updated to new_cb.
 */
FORCE_INLINE
void copy_tile_to_dst_init_with_cb_update(uint32_t new_cb, uint32_t& global_old_cb) {
    copy_tile_to_dst_init_short_with_dt(global_old_cb, new_cb);
    global_old_cb = new_cb;
}

/**
 * @brief Computes the integer base-2 logarithm of a 32-bit unsigned integer.
 *
 * This function returns the position of the highest set bit in the input value `n`,
 * effectively calculating ⌊log₂(n)⌋ for n > 0. If `n` is zero, the result is undefined.
 *
 * @param n The 32-bit unsigned integer input.
 * @return The integer part of the base-2 logarithm of `n`.
 */
constexpr uint32_t ilog2(uint32_t n) { return 31 - __builtin_clz(n); }

/**
 * @brief Synchronizes the packer and unpacker using a circular buffer.
 *
 * This function coordinates the synchronization between a packer and an unpacker
 * by performing a sequence of reserve, push, wait, and pop operations on a circular buffer
 * identified by the given index. The synchronization ensures that the packer and unpacker
 * operate in lockstep, processing one tile at a time.
 * This should be used when a compute kernel performs multiple sequential operations on data
 * from the same circular buffer (CB), to avoid situations where the packer from a previous
 * operation has not yet finished but the packer from the next operation has already started.
 * Using this synchronization prevents data hazards and ensures correct ordering of operations.
 *
 * @param packer_unpacker_sync_cb_index The index of the circular buffer used for synchronization.
 */
FORCE_INLINE
void sync_packer_unpacker(uint32_t packer_unpacker_sync_cb_index) {
    constexpr uint32_t ONE_TILE = 1;

    // This double sequence forces both the packer and the unpacker to wait for the other.
    // If we had a single sequence:
    //
    // If packer_unpacker_sync_cb_index is empty
    // - if packer is first, then it will push a tile and continue (it does not wait for the unpacker)
    // - if unpacker is first, then it will wait for packer
    //
    // If packer_unpacker_sync_cb_index is full
    // - if packer is first, then it will wait for unpacker
    // - if unpacker is first, then it will pop a tile and continue (it does not wait for the packer)
    //
    // By having a double sequence, we ensure that both the packer and the unpacker will wait for the other.
    // Indeed:
    // - if packer is first and CB is empty, then it will push a tile and continue until second sequence where it
    // will wait because CB will be full (it will wait for unpacker to pop tile).
    // - if unpacker is first and CB is full, then it will pop a tile and continue until second sequence where it
    // will wait because CB will be empty (it will wait for packer to push tile).

    cb_reserve_back(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_push_back(packer_unpacker_sync_cb_index, ONE_TILE);

    cb_wait_front(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_pop_front(packer_unpacker_sync_cb_index, ONE_TILE);

    cb_reserve_back(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_push_back(packer_unpacker_sync_cb_index, ONE_TILE);

    cb_wait_front(packer_unpacker_sync_cb_index, ONE_TILE);
    cb_pop_front(packer_unpacker_sync_cb_index, ONE_TILE);
}
}  // namespace NAMESPACE
