// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Reads RM sticks from DRAM using TensorAccessor, generates reduce scaler and epsilon tiles.
// Optionally reads gamma/beta RM sticks with zero-padding for tilize.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t cb_input_rm = get_compile_time_arg_val(0);       // c_0
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(1);          // c_2
    constexpr uint32_t cb_beta = get_compile_time_arg_val(2);           // c_3
    constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(3);  // c_8
    constexpr uint32_t cb_eps = get_compile_time_arg_val(4);            // c_9
    constexpr uint32_t stick_size = get_compile_time_arg_val(5);        // W * 2 bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(6);                // tiles per row
    constexpr uint32_t has_gamma = get_compile_time_arg_val(7);
    constexpr uint32_t has_beta = get_compile_time_arg_val(8);

    // TensorAccessor args - declare all unconditionally, chained
    constexpr auto input_args = TensorAccessorArgs<9>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // ========== Runtime args ==========
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);
    const uint32_t packed_eps = get_arg_val<uint32_t>(3);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);

    // ========== Construct TensorAccessors ==========
    auto input_accessor = TensorAccessor(input_args, input_addr, stick_size);

    // ========== Generate reduce scaler tile (1/W for AVG) ==========
    // W = Wt * 32, so reduce_factor = Wt * 32
    // AVG with REDUCE_ROW computes scaler = 1/reduce_factor = 1/W
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduce_scaler,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        32,
        Wt * 32>();

    // ========== Generate epsilon tile ==========
    // packed_eps is (bf16 << 16 | bf16). We use prepare_reduce_scaler to fill the tile.
    // Convert packed_eps back to float for the helper
    {
        // Extract bf16 from the packed value (upper 16 bits)
        uint16_t bf16_bits = static_cast<uint16_t>(packed_eps >> 16);
        // Convert bf16 to float: bf16 is upper 16 bits of float32
        uint32_t float_bits = static_cast<uint32_t>(bf16_bits) << 16;
        float eps_float;
        // Use union for type punning
        union {
            uint32_t u;
            float f;
        } converter;
        converter.u = float_bits;
        eps_float = converter.f;
        dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_float);
    }

    // ========== Read gamma/beta RM sticks (once at program start) ==========
    if constexpr (has_gamma) {
        auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, stick_size);
        // Reserve Wt pages in cb_gamma for one tile-row worth of data
        cb_reserve_back(cb_gamma, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_gamma);

        // Read 1 stick (gamma shape is (1,1,1,W) = 1 RM stick)
        uint64_t noc_addr = gamma_accessor.get_noc_addr(0);
        noc_async_read(noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        l1_write_addr += stick_size;

        // Zero-pad remaining 31 sticks (tilize needs 32 sticks per block)
        // Each tile page is tile_size = 2048 bytes, total CB space = Wt * 2048
        // We need 32 * stick_size bytes total, first stick_size already written
        uint32_t remaining_bytes = 31 * stick_size;
        // Use memset-like zero fill by writing zeros from L1
        // Actually, let's just fill the remaining space with noc_async_read from a zero location
        // Simpler: use the fact that L1 might not be zeroed, so we write explicit zeros
        for (uint32_t s = 0; s < 31; s++) {
            // Zero out each stick
            for (uint32_t w = 0; w < stick_size / 4; w++) {
                reinterpret_cast<volatile uint32_t*>(l1_write_addr)[w] = 0;
            }
            l1_write_addr += stick_size;
        }

        cb_push_back(cb_gamma, Wt);
    }

    if constexpr (has_beta) {
        auto beta_accessor = TensorAccessor(beta_args, beta_addr, stick_size);
        cb_reserve_back(cb_beta, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_beta);

        uint64_t noc_addr = beta_accessor.get_noc_addr(0);
        noc_async_read(noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        l1_write_addr += stick_size;

        for (uint32_t s = 0; s < 31; s++) {
            for (uint32_t w = 0; w < stick_size / 4; w++) {
                reinterpret_cast<volatile uint32_t*>(l1_write_addr)[w] = 0;
            }
            l1_write_addr += stick_size;
        }

        cb_push_back(cb_beta, Wt);
    }

    // ========== Main loop: read RM sticks per tile-row ==========
    uint32_t page_id = start_page_id;

    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Reserve Wt pages in input CB (tile-sized pages, but filled with RM sticks)
        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        // Read 32 RM sticks (one tile-row)
        for (uint32_t s = 0; s < 32; s++) {
            uint64_t noc_addr = input_accessor.get_noc_addr(page_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            page_id++;
        }
        noc_async_read_barrier();

        cb_push_back(cb_input_rm, Wt);
    }
}
