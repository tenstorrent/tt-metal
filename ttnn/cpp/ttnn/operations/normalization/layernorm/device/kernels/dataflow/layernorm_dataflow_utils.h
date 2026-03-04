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

/*
 * @brief Read a block of 32x32 tiles from remote memory to L1 for an input CB. Reserves space for a full
 *
 * Since input is row-major, we load partial sticks at a time.
 * For instance, for block size 4 and TILE_W = 32, we load 4x32 = 128 elements at a time from each stick
 * This means that, after tilizing in compute kerel, block.size() tiles will be processed.
 */
template <typename T, uint32_t TILE_W, uint32_t TILE_H>
void push_row_major_blocks_to_cb(
    const uint32_t cb_id_in_rm,
    const T& src_a,
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t abs_tile_row,
    const uint32_t elem_size_bytes,
    const uint32_t full_row_stride) {
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        const uint32_t col_byte_offset = block.start() * TILE_W * elem_size_bytes;
        const uint32_t row_read_bytes = block.size() * TILE_W * elem_size_bytes;
        DPRINT << "[rm_reader] pushing row-major block to cb_in_rm, block.start=" << block.start()
               << ", block size=" << block.size() << ", full block size=" << block.full_block_size()
               << ", row bytes = " << row_read_bytes << ENDL();

        cb_reserve_back(cb_id_in_rm, block.full_block_size());  // DEADLOCK HERE
        uint32_t l1_ptr = get_write_ptr(cb_id_in_rm);

        for (uint32_t row = 0; row < TILE_H; ++row) {
            // DPRINT << "[rm_reader] reading row " << row << " of block " << block.start() << ENDL();

            const uint64_t noc_addr = get_noc_addr(abs_tile_row * TILE_H + row, src_a) + col_byte_offset;
            noc_async_read(noc_addr, l1_ptr, row_read_bytes);
            l1_ptr += full_row_stride;
        }
        noc_async_read_barrier();

        // DPRINT: show what was read for first block of first ncht, to diagnose data corruption.
        // Enable by running with TT_METAL_DPRINT_CORES=0,0
        // if (ncht == 0 && block.is_first()) {
        //     DPRINT << "[rm_reader] src_addr=" << src_addr << " abs_tile_row=" << abs_tile_row
        //            << " col_byte_off=" << col_byte_offset << " row_read_bytes=" << row_read_bytes
        //            << " full_row_stride=" << full_row_stride << " l1_base=" << l1_base << ENDL();
        //     // Print first 4 BF16 values (first 2 rows, 2 elements each)
        //     volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_base);
        //     DPRINT << "[rm_reader] row0[0]=" << BF16(d[0]) << " row0[1]=" << BF16(d[1])
        //            << " row1[0]=" << BF16(d[full_row_stride / elem_size_bytes])
        //            << " row1[1]=" << BF16(d[full_row_stride / elem_size_bytes + 1]) << ENDL();
        // }
        DPRINT << "[rm_reader] pushed row-major block to cb_in_rm, block.start=" << block.start() << ENDL();

        cb_push_back(cb_id_in_rm, block.full_block_size());
    }
    DPRINT << "[rm_reader] done pushing row-major blocks to cb_in_rm" << ENDL();
}

}  // namespace norm::layernorm::device::kernels::dataflow
