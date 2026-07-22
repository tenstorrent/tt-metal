// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"

/**
 * @brief Sorts Wt tiles from row-major order into a bitonic sequence using local sorting and transposition.
 *
 * This function processes tiles in pairs, transposes them for column-wise sorting,
 * performs an in-place local sort (bitonic sort phase), and then packs the sorted
 * and transposed tiles into output buffers. The sorting direction can be switched
 * between pairs to facilitate bitonic merge sort.
 *
 * @param input_cb Input circular buffer containing value tiles.
 * @param index_cb Input circular buffer containing index tiles.
 * @param input_transposed_cb Output circular buffer for transposed value tiles.
 * @param index_transposed_cb Output circular buffer for transposed index tiles.
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
    CircularBuffer& input_cb,
    CircularBuffer& index_cb,
    CircularBuffer& input_transposed_cb,
    CircularBuffer& index_transposed_cb,
    const uint32_t Wt,
    const bool switch_dir,
    const bool ascending,
    const int end_phase) {
    input_transposed_cb.reserve_back(Wt);
    index_transposed_cb.reserve_back(Wt);

    bool ascending_local = ascending;
    for (uint32_t wt = 0; wt < Wt; wt += 2) {
        tile_regs_acquire();

        input_cb.wait_front(2);
        index_cb.wait_front(2);

        // topk_local_sort sorts by columns - transpose input tiles for sorting
        reconfig_data_format_srca(input_cb.get_cb_id());
        transpose_init(input_cb.get_cb_id());
        transpose_tile(input_cb.get_cb_id(), 0, 0);
        transpose_tile(input_cb.get_cb_id(), 1, 1);

        reconfig_data_format_srca(index_cb.get_cb_id());
        transpose_init(index_cb.get_cb_id());
        transpose_tile(index_cb.get_cb_id(), 0, 2);
        transpose_tile(index_cb.get_cb_id(), 1, 3);

        // llk_topk_sort -> inplace
        ckernel::topk_local_sort(0, (int)ascending_local, end_phase);

        tile_regs_commit();
        tile_regs_wait();

        // pack value tiles into transposed buffer
        pack_reconfig_data_format(input_transposed_cb.get_cb_id());
        pack_tile(0, input_transposed_cb.get_cb_id());
        pack_tile(1, input_transposed_cb.get_cb_id());

        // pack index tiles into index transposed buffer
        pack_reconfig_data_format(index_transposed_cb.get_cb_id());
        pack_tile(2, index_transposed_cb.get_cb_id());
        pack_tile(3, index_transposed_cb.get_cb_id());
        input_cb.pop_front(2);
        index_cb.pop_front(2);

        tile_regs_release();

        // Switch sorting direction for bitonic merge sort
        ascending_local = switch_dir ? !ascending_local : ascending_local;
    }

    input_transposed_cb.push_back(Wt);
    index_transposed_cb.push_back(Wt);
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
 * @param transposed_cb Source circular buffer containing tiles to be transposed.
 * @param dest_cb Destination circular buffer where packed tiles will be stored.
 * @param Wt Number of tiles to process (width in tiles).
 */
void transpose_and_pack(CircularBuffer& transposed_cb, CircularBuffer& dest_cb, uint32_t Wt) {
    constexpr uint32_t one_tile = 1;

    // Transpose from sorting by column to right structure
    reconfig_data_format_srca(transposed_cb.get_cb_id());
    transpose_init(transposed_cb.get_cb_id());
    pack_reconfig_data_format(dest_cb.get_cb_id());

    transposed_cb.wait_front(Wt);

    for (uint32_t i = 0; i < Wt; ++i) {
        tile_regs_acquire();

        dest_cb.reserve_back(one_tile);
        transpose_tile(transposed_cb.get_cb_id(), i, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dest_cb.get_cb_id());
        dest_cb.push_back(one_tile);

        tile_regs_release();
    }

    transposed_cb.wait_front(Wt);
    transposed_cb.pop_front(Wt);
}

/**
 * @brief Helper function to manage reconfig_data_format_srca() + copy_init() calls.
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
    reconfig_data_format_srca(global_old_cb, new_cb);
    copy_init(new_cb);
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
 * @param packer_unpacker_sync_cb The circular buffer used for synchronization.
 */
FORCE_INLINE
void sync_packer_unpacker(CircularBuffer& packer_unpacker_sync_cb) {
    constexpr uint32_t ONE_TILE = 1;

    // This double sequence forces both the packer and the unpacker to wait for the other.
    // If we had a single sequence:
    //
    // If packer_unpacker_sync_cb is empty
    // - if packer is first, then it will push a tile and continue (it does not wait for the unpacker)
    // - if unpacker is first, then it will wait for packer
    //
    // If packer_unpacker_sync_cb is full
    // - if packer is first, then it will wait for unpacker
    // - if unpacker is first, then it will pop a tile and continue (it does not wait for the packer)
    //
    // By having a double sequence, we ensure that both the packer and the unpacker will wait for the other.
    // Indeed:
    // - if packer is first and CB is empty, then it will push a tile and continue until second sequence where it
    // will wait because CB will be full (it will wait for unpacker to pop tile).
    // - if unpacker is first and CB is full, then it will pop a tile and continue until second sequence where it
    // will wait because CB will be empty (it will wait for packer to push tile).

    packer_unpacker_sync_cb.reserve_back(ONE_TILE);
    packer_unpacker_sync_cb.push_back(ONE_TILE);

    packer_unpacker_sync_cb.wait_front(ONE_TILE);
    packer_unpacker_sync_cb.pop_front(ONE_TILE);

    packer_unpacker_sync_cb.reserve_back(ONE_TILE);
    packer_unpacker_sync_cb.push_back(ONE_TILE);

    packer_unpacker_sync_cb.wait_front(ONE_TILE);
    packer_unpacker_sync_cb.pop_front(ONE_TILE);
}

/**
 * @brief Copies a tile from a source circular buffer (CB) to a destination CB.
 *
 * This function acquires tile registers, copies a tile from the specified source CB and tile ID
 * to a destination register, commits and waits for the operation, then packs the tile into the
 * destination CB at the specified tile ID. It also handles reconfiguration of the data format
 * for the destination CB and releases the tile registers after the operation.
 *
 * @param last_used_cb_index Last used global circular buffer index.
 * @param src_cb Source circular buffer from which tile will be copied.
 * @param src_tile_id Index of the tile in the circular buffer.
 * @param dst_cb Destination circular buffer to which tile will be copied.
 * @param dst_tile_id Index of the tile in the destination circular buffer.
 */
void copy_tile_between_cbs(
    uint32_t& last_used_cb_index,
    CircularBuffer& src_cb,
    uint32_t src_tile_id,
    CircularBuffer& dst_cb,
    uint32_t dst_tile_id = 0) {
    // Constants
    constexpr uint32_t dest_idx = 0;
    constexpr uint32_t one_tile = 1;

    // Acquire DST for tile copy
    tile_regs_acquire();

    // Copy tile to DST register
    copy_tile_to_dst_init_with_cb_update(src_cb.get_cb_id(), last_used_cb_index);
    copy_tile(src_cb.get_cb_id(), src_tile_id, dest_idx);

    tile_regs_commit();
    tile_regs_wait();

    // Pack the tile into the destination circular buffer
    pack_reconfig_data_format(dst_cb.get_cb_id());
    if (dst_tile_id != 0) {
        pack_tile<true>(dest_idx, dst_cb.get_cb_id(), dst_tile_id);
    } else {
        pack_tile(dest_idx, dst_cb.get_cb_id(), 0);  // Append to the end of the CB
    }

    // Release tile registers
    tile_regs_release();
}
