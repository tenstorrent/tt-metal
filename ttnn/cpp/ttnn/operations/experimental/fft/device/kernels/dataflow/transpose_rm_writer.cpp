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
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "transpose_rm_common.h"

void kernel_main() {
    const uint32_t base_unit = get_arg(args::base_unit);
    const uint32_t num_units = get_arg(args::num_units);

    constexpr uint32_t A_TILES = get_arg(args::a_tiles);
    constexpr uint32_t C_TILES = get_arg(args::c_tiles);
    constexpr uint32_t IS_BF16 = get_arg(args::is_bf16);

    constexpr uint32_t elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr uint32_t row_bytes = T_BLOCK * elem_bytes;

    const auto dst_gen = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer block(dfb::block);

    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t unit_idx = base_unit + u;
        const uint32_t tile_c = unit_idx % C_TILES;
        const uint32_t tile_a = (unit_idx / C_TILES) % A_TILES;
        const uint32_t b = unit_idx / (C_TILES * A_TILES);

        // Destination is (B, C, A) — note swap of A_TILES / C_TILES.
        const uint32_t dst_row_base = b * (C_TILES * T_BLOCK) + tile_c * T_BLOCK;
        const uint32_t dst_col_offset = tile_a * T_BLOCK * elem_bytes;

        block.wait_front(1);

        // Reader has already done the in-L1 transpose, so L1 row i now
        // holds dst row (dst_row_base + i)'s contribution to this tile.
        // CB as source resolves to its read pointer (+ offset_bytes).
        for (uint32_t r = 0; r < T_BLOCK; ++r) {
            const uint32_t dst_row = dst_row_base + r;
            noc.async_write(
                block,
                dst_gen,
                row_bytes,
                {.offset_bytes = r * row_bytes},
                {.page_id = dst_row, .offset_bytes = dst_col_offset});
        }
        noc.async_write_barrier();

        block.pop_front(1);
    }
}
