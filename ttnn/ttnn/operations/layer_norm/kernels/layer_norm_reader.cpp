// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Reader Kernel
// Runs on BRISC (RISCV_0), reads data from DRAM to L1 circular buffers via NOC0.
//
// Responsibilities:
//   1. Generate cb_scaler (1/W) and cb_eps tiles once at program start
//   2. Optionally load cb_gamma and cb_beta tiles once at program start
//   3. Per tile-row (Ht iterations): load Wt input tiles into cb_input

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Compile-time args
constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);
constexpr auto input_tensor_args = TensorAccessorArgs<4>();

// CB indices
constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_eps = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_beta = 4;

void kernel_main() {
    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t eps_u32 = get_arg_val<uint32_t>(3);

    const uint32_t tile_bytes = get_tile_size(cb_input);

    // Reinterpret eps_u32 bits as float
    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_u32;
    float eps_f = eps_conv.f;

    // 1. Generate scaler tile (1/W) into cb_scaler
    float scaler_val = 1.0f / static_cast<float>(Wt * 32);  // 1/W where W = Wt * 32
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_val);

    // 2. Generate eps constant tile into cb_eps
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_f);

    // 3. Optionally load gamma tiles
    if constexpr (has_gamma) {
        // gamma is [1, 1, 1, W] = Wt tiles, starting at gamma_addr
        // We need a TensorAccessor for gamma; but gamma uses the same compile-time
        // args structure. For simplicity, use InterleavedAddrGenFast for gamma/beta
        // since they are simple Wt-tile reads from DRAM interleaved.
        // Actually, we must use get_noc_addr with a separate TensorAccessor or
        // InterleavedAddrGenFast. Since the design says use TensorAccessor only for
        // input/output, and gamma/beta are separate tensors without accessor args,
        // we use the noc_async_read_page approach with a simple interleaved addr gen.

        const auto gamma_accessor = InterleavedAddrGenFast</*is_dram=*/true>{
            .bank_base_address = gamma_addr,
            .page_size = tile_bytes,
            .data_format = get_dataformat(cb_gamma),
        };

        cb_reserve_back(cb_gamma, Wt);
        uint32_t gamma_l1_addr = get_write_ptr(cb_gamma);
        for (uint32_t w = 0; w < Wt; ++w) {
            uint64_t gamma_noc_addr = get_noc_addr(w, gamma_accessor);
            noc_async_read(gamma_noc_addr, gamma_l1_addr, tile_bytes);
            gamma_l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, Wt);
    }

    // 4. Optionally load beta tiles
    if constexpr (has_beta) {
        const auto beta_accessor = InterleavedAddrGenFast</*is_dram=*/true>{
            .bank_base_address = beta_addr,
            .page_size = tile_bytes,
            .data_format = get_dataformat(cb_beta),
        };

        cb_reserve_back(cb_beta, Wt);
        uint32_t beta_l1_addr = get_write_ptr(cb_beta);
        for (uint32_t w = 0; w < Wt; ++w) {
            uint64_t beta_noc_addr = get_noc_addr(w, beta_accessor);
            noc_async_read(beta_noc_addr, beta_l1_addr, tile_bytes);
            beta_l1_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, Wt);
    }

    // 5. Build TensorAccessor for input reads
    const auto input_accessor = TensorAccessor(input_tensor_args, input_addr, tile_bytes);

    // 6. Main loop: for each tile-row, read Wt input tiles
    uint32_t tile_id = 0;
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        cb_reserve_back(cb_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc_async_read_page(tile_id, input_accessor, l1_write_addr);
            l1_write_addr += tile_bytes;
            tile_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input, Wt);
    }
}
