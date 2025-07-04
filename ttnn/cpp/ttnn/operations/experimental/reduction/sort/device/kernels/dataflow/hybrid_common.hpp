// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/pause.h"

#include <cstdint>

using sem_ptr_t = volatile tt_l1_ptr uint32_t*;

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
 * @brief Exchanges Wt tiles between two cores over NoC.
 *
 * This function coordinates the exchange of Wt tiles between the calling core and a peer core.
 * It reads Wt tiles from the local value and index circular buffers, sends them to the peer core,
 * and then receives Wt tiles from the peer, writing them into the specified peer circular buffers.
 * Synchronization is handled using NoC semaphores to ensure correct ordering of tile transfers.
 *
 * @param value_tensor_this_cb_index   Circular buffer index for this core's value tensor tiles (read from).
 * @param index_tensor_this_cb_index   Circular buffer index for this core's index tensor tiles (read from).
 * @param cb_value_peer_index          Circular buffer index for peer's value tensor tiles (write to).
 * @param cb_index_peer_index          Circular buffer index for peer's index tensor tiles (write to).
 * @param Wt                           Number of tiles to exchange.
 * @param value_cb_tile_size           Size (in bytes) of each tile in value buffers.
 * @param index_cb_tile_size           Size (in bytes) of each tile in index buffers.
 * @param other_core_x                 Physical X coordinate of the peer core.
 * @param other_core_y                 Physical Y coordinate of the peer core.
 * @param sem_self_ptr                 Pointer to this core's semaphore for synchronization.
 */
FORCE_INLINE
void sort_noc_exchange_Wt_tiles(
    uint32_t value_tensor_this_cb_index,
    uint32_t index_tensor_this_cb_index,
    uint32_t cb_value_peer_index,
    uint32_t cb_index_peer_index,
    uint32_t Wt,
    uint32_t value_cb_tile_size,
    uint32_t index_cb_tile_size,
    uint32_t other_core_x,
    uint32_t other_core_y,
    sem_ptr_t sem_self_ptr) {
    constexpr uint32_t ONE_TILE = 1;

    const uint64_t sem_noc_addr = get_noc_addr(other_core_x, other_core_y, reinterpret_cast<uint32_t>(sem_self_ptr));

    for (uint32_t w = 0, sem_counter = 1; w < Wt; w++, sem_counter += 2) {
        // Reserve space for new tiles
        cb_reserve_back(cb_value_peer_index, ONE_TILE);
        cb_reserve_back(cb_index_peer_index, ONE_TILE);

        uint32_t cb_value_peer_local_write_addr = get_write_ptr(cb_value_peer_index);
        uint32_t cb_index_peer_local_write_addr = get_write_ptr(cb_index_peer_index);

        uint64_t cb_value_peer_noc_write_addr =
            get_noc_addr(other_core_x, other_core_y, cb_value_peer_local_write_addr);
        uint64_t cb_index_peer_noc_write_addr =
            get_noc_addr(other_core_x, other_core_y, cb_index_peer_local_write_addr);

        // Handshake for tile exchange
        noc_semaphore_inc(sem_noc_addr, 1);
        noc_semaphore_wait(sem_self_ptr, sem_counter);

        // Send local indices and values to peer
        cb_wait_front(value_tensor_this_cb_index, ONE_TILE);
        cb_wait_front(index_tensor_this_cb_index, ONE_TILE);
        uint32_t value_cb_self_read_addr = get_read_ptr(value_tensor_this_cb_index);
        uint32_t index_cb_self_read_addr = get_read_ptr(index_tensor_this_cb_index);

        // Write tiles to peer core
        noc_async_write(value_cb_self_read_addr, cb_value_peer_noc_write_addr, value_cb_tile_size);
        noc_async_write(index_cb_self_read_addr, cb_index_peer_noc_write_addr, index_cb_tile_size);

        noc_async_write_barrier();

        cb_pop_front(value_tensor_this_cb_index, ONE_TILE);
        cb_pop_front(index_tensor_this_cb_index, ONE_TILE);

        // Indicate finish reading and wait for other core to finish
        noc_semaphore_inc(sem_noc_addr, 1);
        noc_semaphore_wait(sem_self_ptr, sem_counter + 1);

        // Push incoming tiles to compute buffers
        cb_push_back(cb_value_peer_index, ONE_TILE);
        cb_push_back(cb_index_peer_index, ONE_TILE);
    }  // Wt

    // Reset semaphore value
    noc_semaphore_set(sem_self_ptr, 0);
}

/**
 * @brief Retrieves the physical coordinates (x, y) of a core given its core ID.
 *
 * This function looks up the physical coordinates of a core using a lookup table buffer.
 * It first checks if the core ID is valid for the given tile size. If the core ID is invalid,
 * it returns (0, 0) as an indicator. Otherwise, it reads the x and y coordinates from the
 * lookup table buffer at the specified index.
 *
 * @param core_id The logical ID of the core whose physical coordinates are to be retrieved.
 * @param lookup_table_buffer_cb_index The circular buffer index for the lookup table containing core coordinates.
 * @param tile_size The size of the tile (default is 1024). Used to validate the core ID.
 * @return std::pair<uint32_t, uint32_t> The physical (x, y) coordinates of the core. Returns (0, 0) if the core ID is
 * invalid.
 */
FORCE_INLINE std::pair<uint32_t, uint32_t> get_core_physical_coordinates(
    const uint32_t core_id, const uint32_t lookup_table_buffer_cb_index, const uint32_t tile_size = 1024) {
    // Initialize as invalid coordinates
    uint32_t core_x = 0;
    uint32_t core_y = 0;

    if (2 * core_id >= tile_size) {
        return {core_x, core_y};  // Invalid core ID
    }

    const uint32_t l1_read_addr = get_read_ptr(lookup_table_buffer_cb_index);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr);

    core_x = ptr[core_id * 2];
    core_y = ptr[core_id * 2 + 1];

    return {core_x, core_y};
}

/**
 * @brief Synchronizes a group of cores using a semaphore-based barrier.
 *
 * This function implements a barrier synchronization mechanism among multiple cores.
 * One core acts as the leader and waits for all other participating cores to reach the barrier.
 * Once all non-leader cores have signaled their arrival, the leader broadcasts a release signal,
 * allowing all cores to proceed past the barrier.
 *
 * @param physical_core_lookup_table_cb_index Index to the lookup table for physical core coordinates.
 * @param sem_barrier_addr Address of the semaphore used for the barrier.
 * @param this_core_id ID of the current core executing the function.
 * @param leader_core_id ID of the leader core responsible for coordinating the barrier.
 * @param num_cores Total number of participating cores in the barrier.
 * @param start_core_id ID of the first core in the participating group.
 *
 * @note If only one core is participating, the function returns immediately.
 * @note Assumes the existence of helper functions for semaphore and NoC operations.
 */
FORCE_INLINE
void sort_barrier(
    uint32_t physical_core_lookup_table_cb_index,
    uint32_t sem_barrier_addr,
    uint32_t this_core_id,
    uint32_t leader_core_id,
    uint32_t num_cores,
    uint32_t start_core_id) {
    // Early exit
    if (num_cores == 1) {
        return;
    }

    // Get semaphore pointer
    sem_ptr_t sem_self_barrier_ptr = reinterpret_cast<sem_ptr_t>(sem_barrier_addr);

    if (this_core_id == leader_core_id) {
        // Leader core logic - control other cores
        // Wait for all other cores to reach the barrier
        noc_semaphore_wait(sem_self_barrier_ptr, num_cores - 1);
        noc_semaphore_set(sem_self_barrier_ptr, 0);

        // Broadcast to all other cores
        for (uint32_t core_id = start_core_id; core_id < num_cores; core_id++) {
            if (core_id == this_core_id) {
                continue;
            }

            const std::pair<uint32_t, uint32_t> remote_core_physical =
                get_core_physical_coordinates(core_id, physical_core_lookup_table_cb_index);
            uint64_t sem_barrier_noc_addr =
                get_noc_addr(remote_core_physical.first, remote_core_physical.second, sem_barrier_addr);
            noc_semaphore_inc(sem_barrier_noc_addr, 1);
        }
    } else {
        // Indicate finish reading and wait for leader core to signal
        const std::pair<uint32_t, uint32_t> remote_core_physical =
            get_core_physical_coordinates(leader_core_id, physical_core_lookup_table_cb_index);
        uint64_t sem_barrier_noc_addr =
            get_noc_addr(remote_core_physical.first, remote_core_physical.second, sem_barrier_addr);
        noc_semaphore_inc(sem_barrier_noc_addr, 1);
        noc_semaphore_wait(sem_self_barrier_ptr, 1);
        noc_semaphore_set(sem_self_barrier_ptr, 0);
    }
}
