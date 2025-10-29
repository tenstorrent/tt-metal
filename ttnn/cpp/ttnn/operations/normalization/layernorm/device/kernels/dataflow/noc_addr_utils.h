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

#include "dataflow_api.h"

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
}  // namespace norm::layernorm::device::kernels::dataflow
