// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Reader Kernel
// RM path: reads 32 sticks per tile-row into c_0 (Wt tile-pages)
// TILE path: reads Wt tiles per tile-row into c_1
// Also generates reduce scaler in c_2, epsilon in c_5

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
constexpr auto input_accessor_args = TensorAccessorArgs<6>();

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

    // ---- Generate reduce scaler tile in c_2 (1/W as bfloat16) ----
    // Use prepare_reduce_scaler with the float value reconstructed from bits
    // The scaler_bits is a packed bf16 value (bf16 << 16 | bf16), but we need
    // to pass a float to prepare_reduce_scaler. The scaler CB is always bf16 format.
    // Actually, the scaler value 1/W was packed as bf16 by the host. We need to
    // pass the actual float 1/W. Let's reconstruct from the bits.
    // scaler_bits is actually (bf16 << 16 | bf16). We can just pass 1.0f/W.
    // But we don't have W at compile time. We have Wt = W/32.
    // Actually 1/W = 1/(Wt * 32).
    constexpr float scaler_float = 1.0f / static_cast<float>(Wt * 32);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_float);

    // ---- Generate epsilon tile in c_5 ----
    // eps_bits is float32 representation of epsilon
    // We need to fill tile[0][0] with epsilon. Use similar pattern to scaler.
    // Actually epsilon needs to be in the input data format. For bf16, we fill row0.
    // Re-interpret eps_bits as float
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_bits;
    float eps_f = eps_conv.f;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_f);

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
