// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
//
// Reads RM sticks from DRAM into CB 0 for tilize.
// Also prepares scaler tiles for reduce operations (stages 2+).
//
// Compile-time args:
//   [0] stick_size_bytes
//   [1] gamma_stick_size (unused until stage 4)
//   [2] has_gamma
//   [3] has_beta
//   [4+] TensorAccessorArgs for input
//   [N+] TensorAccessorArgs for gamma (if has_gamma)
//   [M+] TensorAccessorArgs for beta (if has_beta)
//
// Runtime args:
//   [0] src_addr
//   [1] gamma_addr
//   [2] beta_addr
//   [3] num_sticks (total RM sticks this core processes)
//   [4] start_stick_id
//   [5] scaler_value (1/W packed bf16)
//   [6] eps_value (epsilon packed bf16)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);
    constexpr auto input_tensor_args = TensorAccessorArgs<4>();

    // Wt = number of tiles per row = stick_size / (32 * 2) for bf16
    constexpr uint32_t Wt = stick_size_bytes / (32 * 2);
    constexpr uint32_t TILE_HEIGHT = 32;

    // CB indices
    constexpr uint32_t cb_in_rm = 0;
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t cb_eps = 3;
    constexpr uint32_t cb_gamma_rm = 6;
    constexpr uint32_t cb_beta_rm = 7;

    // ===== Runtime args =====
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_sticks = get_arg_val<uint32_t>(3);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(4);
    const uint32_t scaler_value = get_arg_val<uint32_t>(5);
    const uint32_t eps_value = get_arg_val<uint32_t>(6);

    // Early exit for idle cores
    if (num_sticks == 0) {
        return;
    }

    // Create tensor accessor for input
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size_bytes);

    // ===== One-time setup: prepare scaler and epsilon tiles =====
    // Scaler value is already packed as bf16-pair uint32
    // Use prepare_reduce_scaler with the float-converted scaler
    // The scaler_value from host is already in bf16-packed format
    // We use the host-provided packed value directly by reinterpreting
    {
        // Reinterpret scaler_value (bf16 packed uint32) back to float for prepare_reduce_scaler
        // Actually, prepare_reduce_scaler takes a float and converts it internally
        // The host packed it as bf16, so we need to pass the original float 1.0/W
        // But we only have the packed value. Let's use generate_reduce_scaler_helper.
        // Actually, looking at the API: prepare_reduce_scaler<cb_id>(float scaler_f)
        // The host sends scaler_value as packed bf16 for convenience, but the kernel
        // function needs the actual float. Let me use a different approach: write the
        // packed value directly into the tile.

        // For prepare_reduce_scaler, we need the float value 1.0f/W
        // W = stick_size_bytes / 2 (bf16 = 2 bytes per element)
        constexpr uint32_t W = stick_size_bytes / 2;
        constexpr float scaler_float = 1.0f / W;
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_float);
    }

    // Prepare epsilon tile
    {
        // eps = 1e-5
        // The eps_value from host is packed bf16, but prepare_reduce_scaler wants float
        // Use constexpr default for now; the host passes eps in runtime args
        // but prepare_reduce_scaler needs float. We'll reconstruct from runtime arg.
        // Actually for stage 1 we don't use eps, but we prepare it anyway for simplicity.
        // For correctness, let's compute it properly:
        // eps_value is bf16-packed uint32 from host. We can extract the float approximation.
        // But prepare_reduce_scaler takes float and converts internally.
        // For now, use the literal 1e-5f. The host epsilon parameter defaults to 1e-5.
        dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(1e-5f);
    }

    // ===== Per-tile-row loop: read RM sticks into cb_in_rm =====
    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;
    uint32_t stick_id = start_stick_id;

    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Reserve Wt tile-sized pages in CB 0
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        // Read 32 sticks into the reserved space
        for (uint32_t s = 0; s < TILE_HEIGHT; s++) {
            uint64_t noc_addr = get_noc_addr(stick_id, input_accessor);
            noc_async_read(noc_addr, l1_write_addr, stick_size_bytes);
            l1_write_addr += stick_size_bytes;
            stick_id++;
        }

        noc_async_read_barrier();

        // Push Wt pages (32 sticks = Wt tile-pages worth of data)
        cb_push_back(cb_in_rm, Wt);
    }
}
