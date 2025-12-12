// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file noc_addr_utils.h
 * @brief Utility functions to compute NOC addresses for
 *        remote reading and multicasting for the sharded
 *        layernorm reader kernels.
 */

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace norm::layernorm::device::kernels::dataflow {

using NumNocAddrs = uint32_t;

// Would be better to use std::array, but it
// was causing the program to hang
template <NumNocAddrs N>
using RemoteNocAddrs = uint64_t[N];

using L1Ptr = volatile tt_l1_ptr uint32_t*;

/**
 * @brief Compute NOC addresses for a two-stage reduce.
 *        Populates `remote_noc_addrs_first_stage` and
 *        `remote_noc_addrs_second_stage` with the NOC addresses
 *        of the remote cores in its all-to-all or multicast
 *        network.
 * @tparam row_major Whether the cores should be indexed in
 *         row-major order
 * @tparam num_remote_workers_first_stage The number of remote
 *         workers in the first stage
 * @tparam num_remote_workers_second_stage The number of remote
 *         workers in the second stage
 * @param remote_noc_addrs_first_stage The array to populate with
 *        the NOC addresses of the remote workers in the first stage
 * @param remote_noc_addrs_second_stage The array to populate with
 *        the NOC addresses of the remote workers in the second stage
 * @param p_remote_noc_x Pointer to L1 memory pointing to an
 *        array of X coordinates of device worker cores
 * @param p_remote_noc_y Pointer to L1 memory pointing to an
 *        array of Y coordinates of device worker cores
 * @param start_core_x X coordinate of the starting core
 * @param start_core_y Y coordinate of the starting core
 * @param num_cores_x Total number of cores in the X dimension
 * @param num_cores_y Total number of cores in the Y dimension
 */
template <bool row_major, NumNocAddrs num_remote_workers_first_stage, NumNocAddrs num_remote_workers_second_stage>
inline void compute_two_stage_noc_addrs(
    RemoteNocAddrs<num_remote_workers_first_stage>& remote_noc_addrs_first_stage,
    RemoteNocAddrs<num_remote_workers_second_stage>& remote_noc_addrs_second_stage,
    L1Ptr p_remote_noc_x,
    L1Ptr p_remote_noc_y,
    uint32_t start_core_x,
    uint32_t start_core_y,
    uint32_t num_cores_x,
    uint32_t num_cores_y) {
    uint32_t x = start_core_x, y = start_core_y;
    for (uint32_t i = 0; i < num_remote_workers_first_stage; ++i) {
        remote_noc_addrs_first_stage[i] = get_noc_addr(p_remote_noc_x[x], p_remote_noc_y[y], 0);
        if constexpr (row_major) {
            ++x;
            if (x == num_cores_x) {
                x = 0;
            }
        } else {
            ++y;
            if (y == num_cores_y) {
                y = 0;
            }
        }
    }
    if constexpr (row_major) {
        x = start_core_x;
        y = 0;
    } else {
        x = 0;
        y = start_core_y;
    }
    for (uint32_t i = 0; i < num_remote_workers_second_stage; ++i) {
        remote_noc_addrs_second_stage[i] = get_noc_addr(p_remote_noc_x[x], p_remote_noc_y[y], 0);
        if constexpr (row_major) {
            ++y;
        } else {
            ++x;
        }
    }
}

/**
 * @brief Compute NOC addresses of a core's communication network
 *        for a single-stage reduce
 * @tparam row_major Whether the cores should be indexed in
 *         row-major order
 * @tparam num_remote_workers_first_stage The number of remote
 *        workers in the first stage
 * @tparam num_remote_workers_second_stage The number of remote
 *        workers in the second stage
 * @param remote_noc_addrs_first_stage The array to populate with
 *        the NOC addresses of the remote workers in the first stage
 * @param remote_noc_addrs_second_stage The array to populate with
 *        the NOC addresses of the remote workers in the second stage
 * @param p_remote_noc_x Pointer to L1 memory pointing to an
 *        array of X coordinates of device worker cores
 * @param p_remote_noc_y Pointer to L1 memory pointing to an
 *        array of Y coordinates of device worker cores
 * @param start_core_x X coordinate of the starting core
 * @param start_core_y Y coordinate of the starting core
 * @param num_cores_x Total number of cores in the X dimension
 * @param num_cores_y Total number of cores in the Y dimension
 */
template <bool row_major, NumNocAddrs num_remote_workers>
inline void compute_single_stage_noc_addrs(
    RemoteNocAddrs<num_remote_workers>& remote_noc_addrs,
    L1Ptr p_remote_noc_x,
    L1Ptr p_remote_noc_y,
    uint32_t start_core_x,
    uint32_t start_core_y,
    uint32_t num_cores_x,
    uint32_t num_cores_y) {
    uint32_t x = start_core_x, y = start_core_y;
    for (uint32_t i = 0; i < num_remote_workers; ++i) {
        remote_noc_addrs[i] = get_noc_addr(p_remote_noc_x[x], p_remote_noc_y[y], 0);
        if constexpr (row_major) {
            ++x;
            if (x == num_cores_x) {
                x = 0;
                ++y;
                if (y == num_cores_y) {
                    y = 0;
                }
            }
        } else {
            ++y;
            if (y == num_cores_y) {
                y = 0;
                ++x;
                if (x == num_cores_x) {
                    x = 0;
                }
            }
        }
    }
}

/**
 * @brief Keep a CB aligned with CB size by pushing
 * any excess tiles in the last block
 * @param cb CB to sync
 * @param block Block to sync
 */
template <typename Block>
inline void sync_remainder_block_tiles(uint32_t cb, const Block& block) {
    const auto remainder = block.remainder();
    if (remainder > 0) {
        cb_reserve_back(cb, remainder);
        cb_push_back(cb, remainder);
    }
}

/**
 * @brief A special push function that keeps the CB
 * in aligned with the CB size (assuming the CB is
 * an even nultiple of block.block_size())
 *
 * @tparam Block The block type
 * @param cb CB to pop
 * @param block Block to pop
 */
template <typename Block>
inline void push_block(uint32_t cb, const Block& block) {
    cb_push_back(cb, block.size() + block.remainder());
    // sync_remainder_block_tiles(cb, block);
}

/**
 * @brief Read a block of tiles from remote memory
 * to L1 for an input CB. Does a dummy reserve/push
 * to fill out reminder tiles in a block to keep the
 * total number of reserved tiles aligned with the CB size.
 *
 * @tparam T Type of the AddrGen object
 * @tparam Block The block type
 * @param cb_id The ID of the CB
 * @param addr AddrGen object for accessing tensor data
 * @param tile_bytes The size of a tile in bytes
 * @param offset Global offset for transaction ID
 * @param block Block object that defines the number of tiles to read
 */
template <typename T, typename Block>
void read_block_to_cb(
    const uint32_t cb_id, const T& addr, const uint32_t tile_bytes, const uint32_t offset, const Block& block) {
    cb_reserve_back(cb_id, block.size() + block.remainder());
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    for (uint32_t r = 0; r < block.size(); r++) {
        noc_async_read_tile(offset + r, addr, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    noc_async_read_barrier();
    push_block(cb_id, block);
}
}  // namespace norm::layernorm::device::kernels::dataflow
