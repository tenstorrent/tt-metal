// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre reader (NCRISC, NoC0).
//
// load_resident_constants (once):
//   - W slice of this rank -> cb_weight, K order p = c*n + i  (W page i*Ct + c_start + c). Filled per the
//     W role (R1: own DRAM read; R2: column sender reads + multicasts, receivers get it by mcast and issue
//     their X block 0 read first so it overlaps the W delivery).
//   - bias -> cb_bias_coef in coefficient-major form (slot k, every lane = b[k]; zero elsewhere).
//     The bias tile is staged through the not-yet-pushed first cb_x_resident slot (disjoint lifetime).
//   - reduce scaler (SUM / REDUCE_ROW, 1.0).
// load_x_block (per block): block_token_tiles x core_k_tiles X tiles of this rank's stream-column slice,
//   L1 slot (t, c, i) = t*core_k_tiles + c*n + i. One NoC burst + one barrier per block. The CB is always
//   pushed by the NOMINAL block size (block_token_tiles * core_k_tiles_max) so the FIFO never wraps inside a
//   block, whatever this rank's core_k_tiles or the ragged last block's extent.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "mhc_pre_layout.hpp"

// W fill roles (RT arg; regime R2 `W column broadcast`, op_design.md Regimes):
//   W_ROLE_DRAM     — this core reads its own W slice from DRAM (R1; no broadcast wire).
//   W_ROLE_SENDER   — rank r of the first group row: reads the slice from DRAM and multicasts each chunk
//                      down its physical column (every core of that column holds the same rank r).
//   W_ROLE_RECEIVER — every other core of the column: the slice lands by mcast, one Counter event per chunk.
constexpr uint32_t W_ROLE_DRAM = 0;
constexpr uint32_t W_ROLE_SENDER = 1;
constexpr uint32_t W_ROLE_RECEIVER = 2;
constexpr uint32_t W_MCAST_CT_BASE = 12;
constexpr uint32_t W_MCAST_RT_BASE = 9;

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias_coef = get_compile_time_arg_val(2);
    constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t n_streams = get_compile_time_arg_val(4);
    constexpr uint32_t block_token_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t core_k_tiles_max = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t mix_cols = get_compile_time_arg_val(8);  // n*(n+2)
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(9);  // W pushed in chunks (split pipelining)
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(10);
    constexpr bool needs_max_scaler = get_compile_time_arg_val(11) != 0;  // fp32 X grid split
    constexpr auto w_mc = dataflow_kernel_lib::McastArgs<W_MCAST_CT_BASE, W_MCAST_RT_BASE>();
    constexpr auto x_args = TensorAccessorArgs<w_mc.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_arg_val<uint32_t>(2);
    const uint32_t t_start = get_arg_val<uint32_t>(3);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(4);
    const uint32_t c_start = get_arg_val<uint32_t>(5);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(6);
    const uint32_t num_blocks = get_arg_val<uint32_t>(7);
    const uint32_t w_role = get_arg_val<uint32_t>(8);

    constexpr uint32_t tensor_k_tiles = n_streams * tensor_c_tiles;
    constexpr uint32_t x_block_pages = block_token_tiles * core_k_tiles_max;  // nominal push per block
    const uint32_t core_k_tiles = n_streams * core_c_tiles;

    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const uint32_t w_tile_bytes = get_tile_size(cb_weight);
    const uint32_t b_tile_bytes = get_tile_size(cb_bias_coef);

    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, b_tile_bytes);

    // ---------------- load_resident_constants: W ----------------
    // W slice, K order p = c*n + i, pushed in chunks of w_chunk_tiles so the compute kernel can start
    // consuming (fp32 W: the hi/lo split) while the rest of the slice is still in flight. cb_weight is
    // resident (never popped) at the same L1 address on every core, so a chunk multicast lands in place.
    auto read_w_chunk = [&](uint32_t w_base, uint32_t p0, uint32_t p1) {
        for (uint32_t p = p0; p < p1; ++p) {
            const uint32_t c = p / n_streams;
            const uint32_t i = p - c * n_streams;
            noc_async_read_page(i * tensor_c_tiles + c_start + c, w_acc, w_base + p * w_tile_bytes);
        }
    };
    auto chunk_end = [&](uint32_t p0) {
        return (p0 + w_chunk_tiles) < core_k_tiles ? (p0 + w_chunk_tiles) : core_k_tiles;
    };
    auto load_w = [&]() {
        cb_reserve_back(cb_weight, core_k_tiles);
        const uint32_t w_base = get_write_ptr(cb_weight);
        if (w_role == W_ROLE_DRAM) {
            for (uint32_t p0 = 0; p0 < core_k_tiles; p0 += w_chunk_tiles) {
                const uint32_t p1 = chunk_end(p0);
                read_w_chunk(w_base, p0, p1);
                noc_async_read_barrier();
                cb_push_back(cb_weight, p1 - p0);
            }
            return;
        }
        if constexpr (w_mc.active) {
            Noc noc;
            if (w_role == W_ROLE_SENDER) {
                // Chunk j+1's DRAM read is in flight while chunk j is multicast (reads and the mcast
                // are independent NoC transactions; send() only flushes writes).
                auto sender = w_mc.sender(noc);
                read_w_chunk(w_base, 0, chunk_end(0));
                noc_async_read_barrier();
                for (uint32_t p0 = 0; p0 < core_k_tiles; p0 += w_chunk_tiles) {
                    const uint32_t p1 = chunk_end(p0);
                    if (p1 < core_k_tiles) {
                        read_w_chunk(w_base, p1, chunk_end(p1));
                    }
                    const uint32_t a = w_base + p0 * w_tile_bytes;
                    cb_push_back(cb_weight, p1 - p0);  // compute only reads it: publish before the mcast
                    sender.send(a, a, (p1 - p0) * w_tile_bytes);
                    noc_async_read_barrier();
                }
            } else {
                auto receiver = w_mc.receiver(noc);
                for (uint32_t p0 = 0; p0 < core_k_tiles; p0 += w_chunk_tiles) {
                    receiver.receive();  // Counter: one event per chunk, lands in place
                    cb_push_back(cb_weight, chunk_end(p0) - p0);
                }
            }
        }
    };

    // ---------------- load_resident_constants: bias, scalers ----------------
    // Bias tile staged in the first (not yet pushed) cb_x_resident slot.
    cb_reserve_back(cb_x_resident, x_block_pages);
    const uint32_t stage_addr = get_write_ptr(cb_x_resident);
    noc_async_read_page(0, b_acc, stage_addr);

    // Zero the coefficient-major bias tile over the NoC (unused slots 24..31 must be 0).
    cb_reserve_back(cb_bias_coef, 1);
    {
        Noc noc;
        CircularBuffer bias_cb(cb_bias_coef);
        noc.async_write_zeros(bias_cb, b_tile_bytes);
        noc.write_zeros_l1_barrier();
    }
    noc_async_read_barrier();  // bias tile landed
    {
        volatile tt_l1_ptr float* src = reinterpret_cast<volatile tt_l1_ptr float*>(stage_addr);
        volatile tt_l1_ptr float* dst = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_bias_coef));
        for (uint32_t k = 0; k < mix_cols; ++k) {
            const float bk = src[mhc_layout::rc_index(0, k)];
            for (uint32_t l = 0; l < 32; ++l) {
                dst[mhc_layout::slot_index(k, l)] = bk;
            }
        }
    }
    cb_push_back(cb_bias_coef, 1);

    dataflow_kernel_lib::
        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    if constexpr (needs_max_scaler) {
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_SCALAR>(1.0f);
    }

    // ---------------- load_x_block ----------------
    auto issue_x_block = [&](uint32_t block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent =
            (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;
        cb_reserve_back(cb_x_resident, x_block_pages);
        const uint32_t base = get_write_ptr(cb_x_resident);
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t m = t_start + row0 + t;
            const uint32_t row_page = m * tensor_k_tiles + c_start;
            uint32_t l1 = base + t * core_k_tiles * x_tile_bytes;
            for (uint32_t c = 0; c < core_c_tiles; ++c) {
                for (uint32_t i = 0; i < n_streams; ++i) {
                    noc_async_read_page(row_page + i * tensor_c_tiles + c, x_acc, l1);
                    l1 += x_tile_bytes;
                }
            }
        }
    };

    // A W receiver has no DRAM work of its own before block 0: its X block 0 read is issued first and
    // overlaps the column mcast of W (the reader owns both; the W receive issues no reads).
    uint32_t first_block = 0;
    if (w_role == W_ROLE_RECEIVER && num_blocks > 0) {
        issue_x_block(0);
        load_w();
        noc_async_read_barrier();
        cb_push_back(cb_x_resident, x_block_pages);
        first_block = 1;
    } else {
        load_w();
    }
    for (uint32_t block_idx = first_block; block_idx < num_blocks; ++block_idx) {
        issue_x_block(block_idx);
        noc_async_read_barrier();
        cb_push_back(cb_x_resident, x_block_pages);
    }
}
