// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reader for Sequential Add -> Matmul -> Add Test
 *
 * Reads:
 * - Input X: [1, Wt] tiles
 * - Bias1: [1, Wt] tiles
 * - Weights W: [Wt, Wt] tiles (streamed column by column)
 * - Bias2: [1, Wt] tiles
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t bias1_addr = get_arg_val<uint32_t>(1);
    uint32_t weights_addr = get_arg_val<uint32_t>(2);
    uint32_t bias2_addr = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = 0;       // input X
    constexpr uint32_t cb_bias1 = 1;    // bias1
    constexpr uint32_t cb_weights = 2;  // weights W
    constexpr uint32_t cb_bias2 = 3;    // bias2

    uint32_t tile_size_bytes = get_tile_size(cb_in);

    // Create interleaved address generators
    const InterleavedAddrGen<true> input_addrgen = {
        .bank_base_address = input_addr,
        .page_size = tile_size_bytes,
    };
    const InterleavedAddrGen<true> bias1_addrgen = {
        .bank_base_address = bias1_addr,
        .page_size = tile_size_bytes,
    };
    const InterleavedAddrGen<true> weights_addrgen = {
        .bank_base_address = weights_addr,
        .page_size = tile_size_bytes,
    };
    const InterleavedAddrGen<true> bias2_addrgen = {
        .bank_base_address = bias2_addr,
        .page_size = tile_size_bytes,
    };

    // ========== Read input X [1, Wt] tiles ==========
    for (uint32_t w = 0; w < Wt; w++) {
        uint64_t noc_addr = input_addrgen.get_noc_addr(w);
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in);
        noc_async_read(noc_addr, l1_write_addr, tile_size_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }

    // ========== Read bias1 [1, Wt] tiles ==========
    for (uint32_t w = 0; w < Wt; w++) {
        uint64_t noc_addr = bias1_addrgen.get_noc_addr(w);
        cb_reserve_back(cb_bias1, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_bias1);
        noc_async_read(noc_addr, l1_write_addr, tile_size_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_bias1, 1);
    }

    // ========== Read weights W for matmul phase ==========
    // For matmul [1, Wt] x [Wt, Wt] = [1, Wt]
    // Each output tile [0, nt] requires reading column nt of W
    // W[kt, nt] is at index kt*Wt + nt (row-major)
    for (uint32_t nt = 0; nt < Wt; nt++) {
        for (uint32_t kt = 0; kt < Wt; kt++) {
            uint32_t weight_tile_idx = kt * Wt + nt;
            uint64_t noc_addr = weights_addrgen.get_noc_addr(weight_tile_idx);
            cb_reserve_back(cb_weights, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_weights);
            noc_async_read(noc_addr, l1_write_addr, tile_size_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_weights, 1);
        }
    }

    // ========== Read bias2 [1, Wt] tiles ==========
    for (uint32_t w = 0; w < Wt; w++) {
        uint64_t noc_addr = bias2_addrgen.get_noc_addr(w);
        cb_reserve_back(cb_bias2, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_bias2);
        noc_async_read(noc_addr, l1_write_addr, tile_size_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_bias2, 1);
    }
}
