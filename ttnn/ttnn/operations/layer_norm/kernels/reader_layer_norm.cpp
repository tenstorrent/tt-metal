// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
//
// Three-pass pattern:
//   Startup : Fill c_1 (scaler 1/W) and c_2 (epsilon) once.
//   Pass 1  : Stream Wt input tiles -> c_0 (compute uses for mean).
//   Pass 2  : Re-stream Wt input tiles -> c_0 (compute uses for variance).
//   Pass 3  : Re-stream Wt input tiles -> c_0, plus gamma -> c_3, beta -> c_4.
//
// Compile-time args (via TensorAccessorArgs):
//   [0+] : TensorAccessorArgs for input tensor
//
// Runtime args:
//   [0] input_addr        : uint32  Input buffer base address
//   [1] num_rows_per_core : uint32  Number of tile-rows for this core
//   [2] Wt               : uint32  Width in tiles
//   [3] tile_offset      : uint32  Starting tile index in the input buffer
//   [4] gamma_addr       : uint32  Gamma buffer address (0 if no gamma)
//   [5] beta_addr        : uint32  Beta buffer address  (0 if no beta)
//   [6] eps_bits         : uint32  Epsilon packed as (bf16 << 16 | bf16)
//   [7] scaler_bits      : uint32  1/W   packed as (bf16 << 16 | bf16)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const uint32_t eps_bits = get_arg_val<uint32_t>(6);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(7);

    // Compile-time args: TensorAccessor for input
    constexpr auto input_args = TensorAccessorArgs<0>();

    // CB indices
    constexpr uint32_t cb_input = 0;   // c_0
    constexpr uint32_t cb_scaler = 1;  // c_1
    constexpr uint32_t cb_eps = 2;     // c_2

    const uint32_t tile_bytes = get_local_cb_interface(cb_input).fifo_page_size;

    // Create TensorAccessor for input
    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    // ======================================================================
    // Startup: Fill scaler CB (c_1) with 1/W packed value
    // ======================================================================
    {
        cb_reserve_back(cb_scaler, 1);
        auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
            ptr[i] = scaler_bits;
        }
        cb_push_back(cb_scaler, 1);
    }

    // ======================================================================
    // Startup: Fill eps CB (c_2) with epsilon packed value
    // ======================================================================
    {
        cb_reserve_back(cb_eps, 1);
        auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
            ptr[i] = eps_bits;
        }
        cb_push_back(cb_eps, 1);
    }

    // ======================================================================
    // Per tile-row: 3-pass streaming
    // ======================================================================
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Pass 1: stream Wt tiles to c_0
        for (uint32_t col = 0; col < Wt; ++col) {
            uint32_t tile_idx = tile_offset + row * Wt + col;
            cb_reserve_back(cb_input, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_input);
            noc_async_read_page(tile_idx, input_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_input, 1);
        }

        // Pass 2: re-stream same Wt tiles to c_0
        for (uint32_t col = 0; col < Wt; ++col) {
            uint32_t tile_idx = tile_offset + row * Wt + col;
            cb_reserve_back(cb_input, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_input);
            noc_async_read_page(tile_idx, input_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_input, 1);
        }

        // Pass 3: re-stream same Wt tiles to c_0
        // (gamma/beta reads will be added in Stage 4)
        for (uint32_t col = 0; col < Wt; ++col) {
            uint32_t tile_idx = tile_offset + row * Wt + col;
            cb_reserve_back(cb_input, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_input);
            noc_async_read_page(tile_idx, input_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_input, 1);
        }
    }
}
