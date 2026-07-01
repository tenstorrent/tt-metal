// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// transpose_rm_writer.cpp — BRISC1 / writer for transpose_rm.
//
// Consumes 32×32 already-transposed blocks from CB_TR_BLOCK (reader did
// the in-place swap) and scatters them to the destination tensor of shape
// (B, C, A).  32 small NoC writes per block, one per destination row.
//
// Runtime args:
//   0: dst_addr
//   1: base_unit
//   2: num_units
//   3: dst_page_size_bytes   (= A * elem_size, ROW_MAJOR row stride of dst)
// Compile-time args:
//   0: A_TILES
//   1: C_TILES
//   2: IS_BF16

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "transpose_rm_common.h"

void kernel_main() {
    const uint32_t dst_addr            = get_arg_val<uint32_t>(0);
    const uint32_t base_unit           = get_arg_val<uint32_t>(1);
    const uint32_t num_units           = get_arg_val<uint32_t>(2);
    const uint32_t dst_page_size_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t A_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t C_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t IS_BF16 = get_compile_time_arg_val(2);

    constexpr uint32_t elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr uint32_t row_bytes  = T_BLOCK * elem_bytes;

    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr, .page_size = dst_page_size_bytes};

    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t unit_idx = base_unit + u;
        const uint32_t tile_c =  unit_idx %  C_TILES;
        const uint32_t tile_a = (unit_idx /  C_TILES) % A_TILES;
        const uint32_t b      =  unit_idx / (C_TILES * A_TILES);

        // Destination is (B, C, A) — note swap of A_TILES / C_TILES.
        const uint32_t dst_row_base   = b * (C_TILES * T_BLOCK) + tile_c * T_BLOCK;
        const uint32_t dst_col_offset = tile_a * T_BLOCK * elem_bytes;

        cb_wait_front(CB_TR_BLOCK, 1);
        const uint32_t l1_base = get_read_ptr(CB_TR_BLOCK);

        // Reader has already done the in-L1 transpose, so L1 row i now
        // holds dst row (dst_row_base + i)'s contribution to this tile.
        for (uint32_t r = 0; r < T_BLOCK; ++r) {
            const uint32_t dst_row = dst_row_base + r;
            const uint64_t dst_noc_addr = dst_gen.get_noc_addr(dst_row, dst_col_offset);
            noc_async_write(l1_base + r * row_bytes, dst_noc_addr, row_bytes);
        }
        noc_async_write_barrier();

        cb_pop_front(CB_TR_BLOCK, 1);
    }
}
