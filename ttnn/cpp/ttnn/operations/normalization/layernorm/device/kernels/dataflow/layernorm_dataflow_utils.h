// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file layernorm_dataflow_utils.h
 * @brief Utility functions for the layernorm dataflow kernels.
 */

#pragma once

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>

#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h"

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
 * @brief Read a block of tiles from remote memory
 * to L1 for an input CB. Reserves space for a full
 * block of tiles for synchronization purposes, but
 * only reads tiles that contain data
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
inline void read_block_to_cb(
    const uint32_t cb_id, const T& addr, const uint32_t tile_bytes, const uint32_t offset, const Block& block) {
    // Need to reserve/push on intervals that nicely
    // divide the CB size. The CB and block size has been
    // configured to ensure this in the program setup
    cb_reserve_back(cb_id, block.full_block_size());
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    // Only read in the part of the block that has data
    for (auto r : block.local()) {
        noc_async_read_tile(offset + r, addr, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, block.full_block_size());
}

/**
 * @brief Read one column block of row-major input data from DRAM into cb_id_in_rm.
 *
 * Reads `num_valid_rows` rows, each of width `block.size() * tile_stride_bytes` bytes, starting
 * at column `block.start() * tile_stride_bytes` within each row. Rows are written into L1
 * contiguously with `rm_row_stride_bytes` stride (= full block width including padding tiles).
 * A full block slot (`block.full_block_size()`) is reserved/pushed for synchronization.
 */
template <typename T, typename Block, uint32_t TILE_W, uint32_t TILE_H>
inline void read_row_major_block_to_cb(
    const uint32_t cb_id_in_rm,
    const T& src_a,
    const uint32_t curr_tile_row,
    const uint32_t num_valid_rows,
    const uint32_t tile_stride_bytes,
    const uint32_t rm_row_stride_bytes,
    const Block& block) {
    const uint32_t col_byte_offset = block.start() * tile_stride_bytes;
    const uint32_t row_read_bytes = block.size() * tile_stride_bytes;
    cb_reserve_back(cb_id_in_rm, block.full_block_size());

    uint32_t l1_ptr = get_write_ptr(cb_id_in_rm);
    for (uint32_t row = 0; row < num_valid_rows; ++row) {
        const uint64_t noc_addr = get_noc_addr(curr_tile_row * TILE_H + row, src_a) + col_byte_offset;
        noc_async_read(noc_addr, l1_ptr, row_read_bytes);
        l1_ptr += rm_row_stride_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in_rm, block.full_block_size());
}

template <typename T, typename Block, uint32_t TILE_W, uint32_t TILE_H>
inline void write_row_major_block_from_cb(
    const uint32_t cb_id_out_rm,
    const T& dst_a,
    const uint32_t abs_row_base,
    const uint32_t num_valid_rows,
    const uint32_t tile_width_bytes,
    const uint32_t block_row_stride_bytes,
    const Block& block) {
    // Compute produces block_size tiles (full_block_size) in cb_out_rm; the last block
    // may have fewer valid tiles (block.size() <= blk), but blk slots are reserved.
    cb_wait_front(cb_id_out_rm, block.full_block_size());
    const uint32_t l1_base = get_read_ptr(cb_id_out_rm);

    // Column byte offset in the output row where this block starts.
    const uint32_t col_byte_offset = block.start() * tile_width_bytes;
    // Number of valid bytes to write per row (only block.size() valid tiles).
    const uint32_t valid_bytes = block.size() * tile_width_bytes;

    for (uint32_t r = 0; r < num_valid_rows; r++) {
        const uint32_t l1_src = l1_base + r * block_row_stride_bytes;
        const uint64_t noc_dst = get_noc_addr(abs_row_base + r, dst_a) + col_byte_offset;
        noc_async_write(l1_src, noc_dst, valid_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id_out_rm, block.full_block_size());
}

/**
 * @brief Push all column blocks of one tile-row of row-major input data into cb_id_in_rm.
 *
 * Iterates over all blocks (via `generic::blocks(Wt, block_size)`) and reads each block from DRAM
 * into cb_id_in_rm. Only `num_valid_rows` rows are read per block; padding rows are zero-filled
 * by the tilize step in the compute kernel. Handles the case where H is not tile-aligned.
 */
template <typename T, uint32_t TILE_W, uint32_t TILE_H>
inline void push_row_major_blocks_to_cb(
    const uint32_t cb_id_in_rm,
    const T& src_a,
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t curr_tile_row,
    const uint32_t elem_size_bytes,
    const uint32_t rm_row_stride_bytes,
    const uint32_t H_logical) {
    const uint32_t abs_row_start = curr_tile_row * TILE_H;
    const uint32_t tile_stride_bytes = TILE_W * elem_size_bytes;

    // Number of valid rows in this tile-row. When H is tile-aligned this equals
    // TILE_H for every tile-row and the zero-fill branch is never taken.
    uint32_t num_valid_rows = TILE_H;
    if (abs_row_start >= H_logical) {
        num_valid_rows = 0;
    } else if (H_logical - abs_row_start < TILE_H) {
        num_valid_rows = H_logical - abs_row_start;
    }

    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        const uint32_t col_byte_offset = block.start() * tile_stride_bytes;
        const uint32_t row_read_bytes = block.size() * tile_stride_bytes;

        cb_reserve_back(cb_id_in_rm, block.full_block_size());

        uint32_t l1_ptr = get_write_ptr(cb_id_in_rm);
        for (uint32_t row = 0; row < num_valid_rows; ++row) {
            const uint64_t noc_addr = get_noc_addr(curr_tile_row * TILE_H + row, src_a) + col_byte_offset;
            noc_async_read(noc_addr, l1_ptr, row_read_bytes);
            l1_ptr += rm_row_stride_bytes;
        }
        noc_async_read_barrier();

        cb_push_back(cb_id_in_rm, block.full_block_size());
    }
}

}  // namespace norm::layernorm::device::kernels::dataflow
