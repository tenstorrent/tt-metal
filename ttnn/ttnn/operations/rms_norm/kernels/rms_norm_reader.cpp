// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (Regime A).
//   - prepares the SUM/REDUCE_ROW scaler tile (value 1.0, col-0 matmul fill)
//   - reads gamma once (TILE: tiles into cb_gamma; ROW_MAJOR: sticks into
//     cb_gamma_rm for compute to tilize — Refinement 3 mixed gamma layout)
//   - reads each owned tile-row's Wt tiles into cb_input_resident (read once, P1)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t n = nbytes >> 2;
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = 0;
    }
}
}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_input_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t gamma_is_rm = get_compile_time_arg_val(5);  // gamma.layout == ROW_MAJOR
    constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(6);
    constexpr uint32_t reduce_block = get_compile_time_arg_val(7);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(8);
    constexpr uint32_t W = get_compile_time_arg_val(9);  // true element count along W
    constexpr uint32_t gamma_elem = get_compile_time_arg_val(10);
    constexpr auto input_args = TensorAccessorArgs<11>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    using dataflow_kernel_lib::PoolType;
    using dataflow_kernel_lib::ReduceDim;

    // SUM scaler = 1.0, col-0 (matmul) fill for SUM + REDUCE_ROW.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    const uint32_t tile_bytes = get_tile_size(cb_input_resident);
    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    // gamma read once, held resident.
    if constexpr (has_gamma) {
        if constexpr (gamma_is_rm) {
            // ROW_MAJOR gamma (1,1,1,W): one stick. Push num_chunks chunks of
            // reduce_block tile-pages (row-major), row 0 carries data; compute
            // tilizes them into cb_gamma. Pad columns are zeroed (don't-care for
            // the row-broadcast multiply, kept well-formed).
            constexpr uint32_t TILE_W = 32;
            constexpr uint32_t gamma_tile_row_bytes = TILE_W * gamma_elem;
            constexpr uint32_t gamma_padded_chunk_bytes = reduce_block * gamma_tile_row_bytes;
            constexpr uint32_t chunk_cols = reduce_block * TILE_W;
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
            for (uint32_t c = 0; c < num_chunks; ++c) {
                const uint32_t col0 = c * chunk_cols;
                uint32_t valid_cols = (col0 < W) ? (W - col0) : 0;
                if (valid_cols > chunk_cols) {
                    valid_cols = chunk_cols;
                }
                const uint32_t chunk_row_bytes = valid_cols * gamma_elem;
                const uint32_t byte_off = col0 * gamma_elem;
                cb_reserve_back(cb_gamma_rm, reduce_block);
                uint32_t l1 = get_write_ptr(cb_gamma_rm);
                if (chunk_row_bytes > 0) {
                    const uint32_t zstart = chunk_row_bytes & ~3u;
                    if (zstart < gamma_padded_chunk_bytes) {
                        zero_l1(l1 + zstart, gamma_padded_chunk_bytes - zstart);
                    }
                    noc_async_read(gamma_accessor.get_noc_addr(0, byte_off), l1, chunk_row_bytes);
                } else {
                    zero_l1(l1, gamma_padded_chunk_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma_rm, reduce_block);
            }
        } else {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
            cb_reserve_back(cb_gamma, Wt);
            uint32_t l1 = get_write_ptr(cb_gamma);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                noc_async_read_tile(wt, gamma_accessor, l1);
                l1 += gamma_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, Wt);
        }
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        const uint32_t global_row = start_row + row;
        const uint32_t page_base = global_row * Wt;

        cb_reserve_back(cb_input_resident, Wt);
        uint32_t l1 = get_write_ptr(cb_input_resident);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc_async_read_tile(page_base + wt, input_accessor, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_resident, Wt);
    }
}
