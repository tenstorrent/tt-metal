// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SDPA writer (BRISC).
//
// Generates the matmul-reduce tile (col-0 ones, elsewhere zero) used by
// the compute kernel for the in-place row-sum step. Then drains the
// cb_output queue (Dt tiles per query tile-row) to DRAM.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint16_t BF16_ONE_BITS = 0x3F80;  // bfloat16 1.0

// Fill a tile with col-0 = one (bf16) and zero elsewhere. The tile is
// stored in faced format: 4 16x16 faces. Faces 0 and 2 are "left"
// faces (cover columns 0..15); faces 1 and 3 are "right" faces. The
// canonical "matmul row-reduce" tile sets the FIRST column (w==0) of
// each left face to 1, everything else 0. When this tile is used as
// in1 of a matmul, the result is a row-sum-as-column-vector.
inline void generate_matmul_row_reduce_tile_bf16(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    auto* tile_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t h = 0; h < 16; ++h) {
            for (uint32_t w = 0; w < 16; ++w) {
                *tile_ptr++ = (!(face & 1u) && (w == 0)) ? BF16_ONE_BITS : 0u;
            }
        }
    }
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    // --- Compile-time args -------------------------------------------------
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Qt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    // --- Runtime args ------------------------------------------------------
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);

    // --- CB indices --------------------------------------------------------
    constexpr uint32_t cb_matmul_reduce = 6;
    constexpr uint32_t cb_output = 16;

    // --- One-shot col-0-ones tile for matmul-as-reduce ---------------------
    generate_matmul_row_reduce_tile_bf16(cb_matmul_reduce);

    // --- Tensor accessor ---------------------------------------------------
    const uint32_t tile_bytes = get_tile_size(cb_output);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    // --- Drain cb_output to DRAM (Dt tiles per row) ------------------------
    const uint32_t end_row = start_row + num_rows;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Decode r -> (b, h, qt). Output layout (B, H, Qt, Dt).
        const uint32_t b = r / (H * Qt);
        const uint32_t h = (r / Qt) % H;
        const uint32_t qt = r % Qt;
        const uint32_t out_base = ((b * H + h) * Qt + qt) * Dt;

        cb_wait_front(cb_output, Dt);
        {
            uint32_t l1_addr = get_read_ptr(cb_output);
            for (uint32_t d = 0; d < Dt; ++d) {
                noc_async_write_tile(out_base + d, out_acc, l1_addr);
                l1_addr += tile_bytes;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_output, Dt);
    }
}
