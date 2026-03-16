// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Reader Kernel
// Runs on RISCV_0 (BRISC), reads data from DRAM via NOC0
//
// Two-pass reader per tile-row:
//   Pass 1: push Wt input tiles to cb_in (RM: read sticks; TILE: read tiles)
//   Pass 2: re-push same Wt tiles for normalization
//   Startup: fill cb_scaler with 1/W, fill cb_eps with epsilon
//   Optional: push gamma tiles per row (stage 4)

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// CB indices
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_in = 1;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_eps = 9;

// Compile-time args
constexpr uint32_t is_rm_input = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(3);

// TensorAccessor args for input start at CT index 4
constexpr auto input_ta_args = TensorAccessorArgs<4>();

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);
    uint32_t num_sticks = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t packed_scaler = get_arg_val<uint32_t>(6);
    uint32_t packed_eps = get_arg_val<uint32_t>(7);

    // Create TensorAccessor for input
    const uint32_t input_page_size = is_rm_input ? stick_size : get_tile_size(cb_in);
    const auto input_accessor = TensorAccessor(input_ta_args, src_addr, input_page_size);

    // Fill cb_scaler with 1/W for reduce
    float scaler_val = 1.0f / static_cast<float>(Wt * 32);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_val);

    // Fill cb_eps with epsilon value
    // packed_eps is the raw float32 bits of epsilon; reinterpret as float
    // and reuse prepare_reduce_scaler to fill row 0 of each face (sufficient for SCALAR broadcast)
    {
        union {
            uint32_t u;
            float f;
        } eps_conv;
        eps_conv.u = packed_eps;
        dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_conv.f);
    }

    // Two-pass reading per tile-row
    if constexpr (is_rm_input) {
        // RM input: read 32 sticks per tile-row, push Wt pages to cb_in_rm
        uint32_t stick_id = 0;
        for (uint32_t row = 0; row < num_rows; ++row) {
            uint32_t row_start_stick = stick_id;

            // Pass 1: read input sticks for square + reduce
            cb_reserve_back(cb_in_rm, Wt);
            uint32_t l1_write_addr = get_write_ptr(cb_in_rm);
            for (uint32_t s = 0; s < 32; ++s) {
                uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
                noc_async_read(noc_addr, l1_write_addr, stick_size);
                l1_write_addr += stick_size;
                stick_id++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in_rm, Wt);

            // Pass 2: re-read same sticks for normalization
            uint32_t reread_stick_id = row_start_stick;
            cb_reserve_back(cb_in_rm, Wt);
            l1_write_addr = get_write_ptr(cb_in_rm);
            for (uint32_t s = 0; s < 32; ++s) {
                uint64_t noc_addr = input_accessor.get_noc_addr(reread_stick_id);
                noc_async_read(noc_addr, l1_write_addr, stick_size);
                l1_write_addr += stick_size;
                reread_stick_id++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in_rm, Wt);
        }
    } else {
        // TILE input: read tiles directly to cb_in
        uint32_t tile_id = 0;
        uint32_t tile_size = get_tile_size(cb_in);
        for (uint32_t row = 0; row < num_rows; ++row) {
            uint32_t row_start_tile = tile_id;

            // Pass 1: read Wt tiles for square + reduce
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_reserve_back(cb_in, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_in);
                uint64_t noc_addr = input_accessor.get_noc_addr(tile_id);
                noc_async_read(noc_addr, l1_write_addr, tile_size);
                noc_async_read_barrier();
                cb_push_back(cb_in, 1);
                tile_id++;
            }

            // Pass 2: re-read same Wt tiles for normalization
            uint32_t reread_tile_id = row_start_tile;
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_reserve_back(cb_in, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_in);
                uint64_t noc_addr = input_accessor.get_noc_addr(reread_tile_id);
                noc_async_read(noc_addr, l1_write_addr, tile_size);
                noc_async_read_barrier();
                cb_push_back(cb_in, 1);
                reread_tile_id++;
            }
        }
    }
}
