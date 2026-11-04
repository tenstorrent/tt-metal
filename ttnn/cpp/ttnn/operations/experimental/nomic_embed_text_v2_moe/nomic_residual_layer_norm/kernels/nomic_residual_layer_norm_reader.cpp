// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_residual_layer_norm — reader (load_block).
//
//  * x / residual: resident ND shards backing cb_x / cb_residual (zero copy). The reader only hands the
//    compute kernel the credits for the whole shard; no NoC read touches them.
//  * gamma / beta: only row 0 of each of this core's hidden_tiles tiles is ever read by the row-broadcast
//    unpacker, so only the face rows carrying row 0 are fetched (64 B at the face-0 and face-1 offsets).
//    The reads are issued right after the shard credits and retired with one barrier.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_residual = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(3);
    constexpr bool has_residual = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t beta_page_bytes = get_compile_time_arg_val(8);
    constexpr auto gamma_args = TensorAccessorArgs<9>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t beta_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_hidden_tile = get_arg_val<uint32_t>(2);

    // Row 0 of a tile lives in face 0 (columns 0-15) and face 1 (columns 16-31). A face is 16x16 elements.
    constexpr uint32_t row_read_bytes = 64;  // DRAM read granularity
    constexpr uint32_t gamma_face_bytes = gamma_page_bytes / 4;
    constexpr uint32_t beta_face_bytes = beta_page_bytes / 4;

    // Resident shard credits first: compute starts phase A at once; the gamma/beta reads overlap it.
    CircularBuffer x_cb(cb_x);
    x_cb.reserve_back(shard_tiles);
    x_cb.push_back(shard_tiles);
    if constexpr (has_residual) {
        CircularBuffer residual_cb(cb_residual);
        residual_cb.reserve_back(shard_tiles);
        residual_cb.push_back(shard_tiles);
    }

    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, beta_page_bytes);

    CircularBuffer gamma_cb(cb_gamma);
    CircularBuffer beta_cb(cb_beta);
    gamma_cb.reserve_back(hidden_tiles);
    beta_cb.reserve_back(hidden_tiles);
    const uint32_t gamma_l1 = gamma_cb.get_write_ptr();
    const uint32_t beta_l1 = beta_cb.get_write_ptr();
    for (uint32_t j = 0; j < hidden_tiles; ++j) {
        const uint32_t page = first_hidden_tile + j;
        for (uint32_t face = 0; face < 2; ++face) {
            noc_async_read(
                gamma_accessor.get_noc_addr(page, face * gamma_face_bytes),
                gamma_l1 + j * gamma_page_bytes + face * gamma_face_bytes,
                row_read_bytes);
            noc_async_read(
                beta_accessor.get_noc_addr(page, face * beta_face_bytes),
                beta_l1 + j * beta_page_bytes + face * beta_face_bytes,
                row_read_bytes);
        }
    }

    noc_async_read_barrier();
    gamma_cb.push_back(hidden_tiles);
    beta_cb.push_back(hidden_tiles);
}
