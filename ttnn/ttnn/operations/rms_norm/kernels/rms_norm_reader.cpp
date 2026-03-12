// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Reader Kernel
// RM path: reads 32 sticks per tile-row into c_0 (Wt tile-pages)
// TILE path: reads Wt tiles per tile-row into c_1
// Also generates reduce scaler in c_2, epsilon in c_5
// If gamma: loads gamma sticks into c_0 before main loop (for compute to tilize)

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_eps = 5;

// Compile-time args
constexpr uint32_t stick_or_tile_size = get_compile_time_arg_val(0);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(1);
constexpr uint32_t eps_bits = get_compile_time_arg_val(2);
constexpr uint32_t input_is_rm = get_compile_time_arg_val(3);
constexpr uint32_t has_gamma = get_compile_time_arg_val(4);
constexpr uint32_t Wt = get_compile_time_arg_val(5);
constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(6);  // W * element_size (gamma is always RM)
constexpr auto input_accessor_args = TensorAccessorArgs<7>();

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t gamma_addr = get_arg_val<uint32_t>(3);

    if (num_rows == 0) {
        return;
    }

    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_or_tile_size);

    // Gamma accessor — always instantiated (PD appends [0] placeholder when gamma absent)
    [[maybe_unused]] constexpr auto gamma_accessor_args =
        TensorAccessorArgs<input_accessor_args.next_compile_time_args_offset()>();

    // ---- Generate reduce scaler tile in c_2 (1/W as bfloat16) ----
    constexpr float scaler_float = 1.0f / static_cast<float>(Wt * 32);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_float);

    // ---- Generate epsilon tile in c_5 ----
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_bits;
    float eps_f = eps_conv.f;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_f);

    // ---- Load gamma into c_0 before main loop (if gamma provided) ----
    // Gamma is (1,1,1,W) in ROW_MAJOR: 1 stick of W elements.
    // Load into c_0 as Wt tile-pages (32 sticks * W elements = full tile-row).
    // Replicate the single gamma stick across all 32 rows so that after tilize,
    // every row of every tile in c_7 has the gamma values. This is needed because
    // ROW broadcast reads per-row data from each B tile face, not just row 0.
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_accessor_args, gamma_addr, gamma_stick_size);

        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        // Read gamma stick 32 times to fill all rows of the tile-row
        constexpr uint32_t TILE_H = 32;
        uint64_t gamma_noc_addr = gamma_accessor.get_noc_addr(0);
        for (uint32_t s = 0; s < TILE_H; ++s) {
            noc_async_read(gamma_noc_addr, l1_write_addr, gamma_stick_size);
            l1_write_addr += gamma_stick_size;
        }
        noc_async_read_barrier();

        cb_push_back(cb_input_rm, Wt);
    }

    if constexpr (input_is_rm) {
        // RM path: read 32 sticks per tile-row into c_0
        constexpr uint32_t TILE_H = 32;

        for (uint32_t row = 0; row < num_rows; ++row) {
            // Reserve Wt tile-pages in c_0
            cb_reserve_back(cb_input_rm, Wt);
            uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

            // Read 32 sticks (one tile-row worth)
            for (uint32_t s = 0; s < TILE_H; ++s) {
                uint32_t stick_id = start_id + row * TILE_H + s;
                uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
                noc_async_read(noc_addr, l1_write_addr, stick_or_tile_size);
                l1_write_addr += stick_or_tile_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_rm, Wt);
        }
    } else {
        // TILE path: read Wt tiles per tile-row directly into c_1
        for (uint32_t row = 0; row < num_rows; ++row) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_reserve_back(cb_tilized, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_tilized);
                uint32_t tile_id = start_id + row * Wt + wt;
                uint64_t noc_addr = input_accessor.get_noc_addr(tile_id);
                noc_async_read(noc_addr, l1_write_addr, stick_or_tile_size);
                noc_async_read_barrier();
                cb_push_back(cb_tilized, 1);
            }
        }
    }
}
