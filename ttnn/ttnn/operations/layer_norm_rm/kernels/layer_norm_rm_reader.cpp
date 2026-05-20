// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for layer_norm_rm (Refinement 2 — streaming + multi-core).
//
// Per-core slice = `Ht_local` consecutive tile-rows starting at
// `start_tile_row` (both passed via RT args).  For each tile-row, the reader
// streams the input THREE times (one full sweep per pass) plus optional
// per-Pass-3-block gamma/beta partial-row sticks.  All input streaming is
// chunked into NUM_BLOCKS blocks of BLOCK_SIZE tiles each:
//
//   RM input:    per pass, per block: push 32 partial-row sticks of
//                `block_row_bytes` bytes into cb_input_sticks (compute calls
//                tilize<BLOCK_SIZE,…>(1, 32) per block).
//   TILE input:  per pass, per block: push BLOCK_SIZE tiles into
//                cb_input_tiles (compute consumes per block).
//
// Gamma/beta (always RM, single row of W) are read per-Pass-3-block as
// partial-row sticks of `block_row_bytes` bytes (one stick per block per row);
// compute tilizes each into BLOCK_SIZE gamma/beta tiles.  cb_gamma_sticks /
// cb_beta_sticks are sized at 32 pages (tile-row capacity) so the LLK's
// 32-row read is safe — only the first stick per push is valid.
//
// The scaler tile pair is pushed once at startup; reduce<> waits-never-pops
// and the compute kernel issues a final cb_pop_front at exit.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(2);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t last_block_input_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t padded_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t input_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t elem_size = get_compile_time_arg_val(7);
    constexpr uint32_t inv_W_bits = get_compile_time_arg_val(8);
    constexpr uint32_t has_partial_w = get_compile_time_arg_val(9);
    constexpr uint32_t partial_w = get_compile_time_arg_val(10);
    constexpr uint32_t is_rm_input = get_compile_time_arg_val(11);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(12);
    constexpr uint32_t has_beta = get_compile_time_arg_val(13);

    constexpr auto input_args = TensorAccessorArgs<14>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t beta_addr = get_arg_val<uint32_t>(2);
    uint32_t start_tile_row = get_arg_val<uint32_t>(3);
    uint32_t Ht_local = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_input_sticks = 0;
    constexpr uint32_t cb_input_tiles = 1;
    constexpr uint32_t cb_gamma_sticks = 2;
    constexpr uint32_t cb_beta_sticks = 3;
    constexpr uint32_t cb_scaler = 4;
    constexpr uint32_t TILE_H = 32;

    // Cores past the work boundary may be assigned with Ht_local=0; skip them.
    if (Ht_local == 0) {
        return;
    }

    // Per-block read size from DRAM. For non-last blocks this is always
    // block_row_bytes; for the last block it may shrink to last_block_input_bytes
    // when partial_w != 0.  When has_partial_w is false, last_block_input_bytes
    // == block_row_bytes (the program descriptor sets it that way) so no
    // special casing is needed at runtime.
    constexpr uint32_t non_last_block_bytes = block_row_bytes;

    // ---------------- Scaler (once at startup) ----------------
    {
        float inv_W = __builtin_bit_cast(float, inv_W_bits);
        if constexpr (has_partial_w) {
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                partial_w>(inv_W);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(inv_W);
        }
    }

    // ---------------- Per-tile-row outer loop ----------------
    if constexpr (is_rm_input) {
        const auto input_accessor = TensorAccessor(input_args, input_addr, padded_row_bytes);
        // Gamma / beta accessors: always RM, single row of `padded_row_bytes`.
        [[maybe_unused]] const auto gamma_accessor_v = has_gamma
                                                           ? TensorAccessor(gamma_args, gamma_addr, padded_row_bytes)
                                                           : TensorAccessor(gamma_args, gamma_addr, padded_row_bytes);
        [[maybe_unused]] const auto beta_accessor_v = has_beta ? TensorAccessor(beta_args, beta_addr, padded_row_bytes)
                                                               : TensorAccessor(beta_args, beta_addr, padded_row_bytes);

        for (uint32_t tr = 0; tr < Ht_local; ++tr) {
            uint32_t global_tile_row = start_tile_row + tr;
            uint32_t base_row = global_tile_row * TILE_H;

            // --- Input: 3 passes × NUM_BLOCKS blocks × 32 partial sticks ---
            for (uint32_t pass = 0; pass < 3; ++pass) {
                for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                    uint32_t bytes_this_block = (b + 1 == NUM_BLOCKS) ? last_block_input_bytes : non_last_block_bytes;
                    uint32_t col_offset = b * block_row_bytes;
                    for (uint32_t r = 0; r < TILE_H; ++r) {
                        uint32_t page_id = base_row + r;
                        cb_reserve_back(cb_input_sticks, 1);
                        uint32_t l1_addr = get_write_ptr(cb_input_sticks);
                        uint64_t noc_addr = input_accessor.get_noc_addr(page_id) + col_offset;
                        noc_async_read(noc_addr, l1_addr, bytes_this_block);
                        noc_async_read_barrier();
                        cb_push_back(cb_input_sticks, 1);
                    }
                }
            }

            // --- Gamma / beta: 1 partial stick per Pass-3 block per row ---
            if constexpr (has_gamma) {
                for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                    uint32_t bytes_this_block = (b + 1 == NUM_BLOCKS) ? last_block_input_bytes : non_last_block_bytes;
                    uint32_t col_offset = b * block_row_bytes;
                    cb_reserve_back(cb_gamma_sticks, 1);
                    uint32_t l1_addr = get_write_ptr(cb_gamma_sticks);
                    uint64_t noc_addr = gamma_accessor_v.get_noc_addr(0) + col_offset;
                    noc_async_read(noc_addr, l1_addr, bytes_this_block);
                    noc_async_read_barrier();
                    cb_push_back(cb_gamma_sticks, 1);
                }
            }
            if constexpr (has_beta) {
                for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                    uint32_t bytes_this_block = (b + 1 == NUM_BLOCKS) ? last_block_input_bytes : non_last_block_bytes;
                    uint32_t col_offset = b * block_row_bytes;
                    cb_reserve_back(cb_beta_sticks, 1);
                    uint32_t l1_addr = get_write_ptr(cb_beta_sticks);
                    uint64_t noc_addr = beta_accessor_v.get_noc_addr(0) + col_offset;
                    noc_async_read(noc_addr, l1_addr, bytes_this_block);
                    noc_async_read_barrier();
                    cb_push_back(cb_beta_sticks, 1);
                }
            }
        }
    } else {
        // TILE input: stream BLOCK_SIZE tiles per (pass, block, tile-row).
        constexpr uint32_t tile_bytes_v = get_tile_size(cb_input_tiles);
        const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes_v);
        [[maybe_unused]] const auto gamma_accessor_v = has_gamma
                                                           ? TensorAccessor(gamma_args, gamma_addr, padded_row_bytes)
                                                           : TensorAccessor(gamma_args, gamma_addr, padded_row_bytes);
        [[maybe_unused]] const auto beta_accessor_v = has_beta ? TensorAccessor(beta_args, beta_addr, padded_row_bytes)
                                                               : TensorAccessor(beta_args, beta_addr, padded_row_bytes);

        for (uint32_t tr = 0; tr < Ht_local; ++tr) {
            uint32_t global_tile_row = start_tile_row + tr;

            for (uint32_t pass = 0; pass < 3; ++pass) {
                for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                    for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                        uint32_t tile_id = global_tile_row * Wt + b * BLOCK_SIZE + wt;
                        cb_reserve_back(cb_input_tiles, 1);
                        uint32_t l1_addr = get_write_ptr(cb_input_tiles);
                        noc_async_read_tile(tile_id, input_accessor, l1_addr);
                        noc_async_read_barrier();
                        cb_push_back(cb_input_tiles, 1);
                    }
                }
            }

            if constexpr (has_gamma) {
                for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                    uint32_t bytes_this_block = (b + 1 == NUM_BLOCKS) ? last_block_input_bytes : non_last_block_bytes;
                    uint32_t col_offset = b * block_row_bytes;
                    cb_reserve_back(cb_gamma_sticks, 1);
                    uint32_t l1_addr = get_write_ptr(cb_gamma_sticks);
                    uint64_t noc_addr = gamma_accessor_v.get_noc_addr(0) + col_offset;
                    noc_async_read(noc_addr, l1_addr, bytes_this_block);
                    noc_async_read_barrier();
                    cb_push_back(cb_gamma_sticks, 1);
                }
            }
            if constexpr (has_beta) {
                for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                    uint32_t bytes_this_block = (b + 1 == NUM_BLOCKS) ? last_block_input_bytes : non_last_block_bytes;
                    uint32_t col_offset = b * block_row_bytes;
                    cb_reserve_back(cb_beta_sticks, 1);
                    uint32_t l1_addr = get_write_ptr(cb_beta_sticks);
                    uint64_t noc_addr = beta_accessor_v.get_noc_addr(0) + col_offset;
                    noc_async_read(noc_addr, l1_addr, bytes_this_block);
                    noc_async_read_barrier();
                    cb_push_back(cb_beta_sticks, 1);
                }
            }
        }
    }
}
