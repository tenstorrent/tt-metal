// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Reader Kernel
// Stage 2+: Reads RM sticks, loads group_scaler mask tiles, fills reduce scaler CB.
// Also reads gamma/beta tiles and fills eps CB (later stages).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_accessor_args = TensorAccessorArgs<1>();
    constexpr auto gamma_accessor_args = TensorAccessorArgs<input_accessor_args.next_compile_time_args_offset()>();
    constexpr auto beta_accessor_args = TensorAccessorArgs<gamma_accessor_args.next_compile_time_args_offset()>();
    constexpr auto group_scaler_accessor_args =
        TensorAccessorArgs<beta_accessor_args.next_compile_time_args_offset()>();

    // ========== RUNTIME ARGS ==========
    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t group_scaler_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks = get_arg_val<uint32_t>(arg_idx++);  // N * HW
    const uint32_t Ct = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t group_scaler_num_tiles = get_arg_val<uint32_t>(arg_idx++);  // G * Ct
    const uint32_t inv_K_packed = get_arg_val<uint32_t>(arg_idx++);            // pre-packed bf16 1/K
    const uint32_t packed_eps = get_arg_val<uint32_t>(arg_idx++);

    // ========== CB INDICES ==========
    constexpr uint32_t cb_input_rm = 0;
    constexpr uint32_t cb_gamma = 2;
    constexpr uint32_t cb_beta = 3;
    constexpr uint32_t cb_eps = 4;
    constexpr uint32_t cb_scaler = 5;
    constexpr uint32_t cb_group_scaler = 26;

    // ========== SETUP TENSOR ACCESSORS ==========
    auto input_accessor = TensorAccessor(input_accessor_args, input_addr, stick_size);
    // group_scaler is TILE_LAYOUT so page_size comes from CB
    constexpr uint32_t group_scaler_tile_size = get_tile_size(cb_group_scaler);
    auto group_scaler_accessor = TensorAccessor(group_scaler_accessor_args, group_scaler_addr, group_scaler_tile_size);

    // ========== STARTUP: FILL REDUCE SCALER CB (value 1/K) ==========
    // inv_K_packed is a double-packed bf16 value of 1/K, prepared by host
    generate_reduce_scaler(cb_scaler, inv_K_packed);

    // ========== STARTUP: FILL EPSILON CB (value eps) ==========
    // packed_eps is a double-packed bf16 value of eps, prepared by host
    generate_reduce_scaler(cb_eps, packed_eps);

    // ========== STARTUP: READ GROUP SCALER MASK TILES INTO CB 26 ==========
    // These are persistent: G*Ct tiles loaded once, used by compute for all samples.
    cb_reserve_back(cb_group_scaler, group_scaler_num_tiles);
    uint32_t gs_l1_addr = get_write_ptr(cb_group_scaler);
    for (uint32_t t = 0; t < group_scaler_num_tiles; ++t) {
        uint64_t noc_addr = group_scaler_accessor.get_noc_addr(t);
        noc_async_read(noc_addr, gs_l1_addr, group_scaler_tile_size);
        gs_l1_addr += group_scaler_tile_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_group_scaler, group_scaler_num_tiles);

    // ========== MAIN LOOP: READ RM STICKS ==========
    // Process 32 sticks at a time (one tile-row), pushing Ct pages per block.
    // Each tile-row has 32 sticks * stick_size = Ct * tile_page_size bytes.
    uint32_t stick_id = 0;
    const uint32_t num_tile_rows = num_sticks / 32;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        cb_reserve_back(cb_input_rm, Ct);
        uint32_t l1_addr = get_write_ptr(cb_input_rm);

        for (uint32_t s = 0; s < 32; ++s) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_addr, stick_size);
            l1_addr += stick_size;
            stick_id++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_input_rm, Ct);
    }
}
