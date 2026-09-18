// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// topk_route_finish reader: per work unit (one 16-row face-pair half of one output tile),
// gather THIS RISC'S SHARE of the selected bf16 logits straight out of the TILE-layout
// source and assemble it into the unit's value and index staging halves. The gather is
// split across both data-movement RISCs: this reader (BRISC) owns unit rows [0, 8), the
// writer (NCRISC) owns rows [8, 16) — see topk_route_finish_gather_common.hpp for the
// row-disjointness argument and the trid double-wave pipeline both sides share.
//
// Per unit(row_tile, kt, half):
//   1. Issue the reads for this RISC's <=8 index-stick segments: for each valid owned row,
//      a valid_cols*4 B slice of that row's RM u32 index stick at byte offset kt*128
//      (both 64 B aligned -- Blackhole DRAM reads require 64 B alignment on both ends).
//   2. Zero THIS RISC's row ranges of both staging halves while the stick reads fly
//      (always: cheaper than tracking which of the four padding cases applies, and it
//      guarantees the zero-filled-tile-padding contract). The writer zeroes rows [8, 16).
//   3. Barrier the stick reads, then gather each selected element with a 64 B NoC read
//      from the source tile's face-row region into a rotating bounce slot — 32-deep waves
//      on alternating trids, retiring the previous wave while the current one flies.
//   4. Push both halves to the writer. The push carries only "rows [0, 8) and their zero
//      fill are done"; the writer completes rows [8, 16) itself before writing out.
//
// TILE face math (32x32 tile = four 16x16 faces stored contiguously, [f0|f1 / f2|f3],
// bf16 face = 512 B, face row = 32 B):
//
//   SOURCE side -- element (wr, wc) of a tile, wr = row & 31, wc = idx & 31:
//     byte = (wr>>4)<<10 | (wc>>4)<<9 | (wr&15)<<5 | (wc&15)<<1
//   (face index (wr>>4)*2 + (wc>>4) selects a 512 B face, (wr&15) the 32 B face row,
//   (wc&15) the 2 B element). The unit spans one source row-tile, so the source page is
//   row_tile*width_tiles + (idx>>5) and wr = half*16 + lr for local row lr in [0,16).
//
//   OUTPUT side -- the unit IS one face-pair of the output tile: rows
//   [half*16, half*16+16) are faces {2*half, 2*half+1}, i.e. bytes
//   [half*1024, half*1024+1024) of the 2048 B bf16 tile page. For output element
//   (lr, c) (c = column within the output tile, c < 32): wr_out = half*16 + lr, so
//   wr_out>>4 == half and wr_out&15 == lr, and the full-tile formula splits into the
//   page base half*1024 (applied by the WRITER as the page offset) plus the staging
//   offset
//     off16 = (c>>4)<<9 | lr<<5 | (c&15)<<1
//   -- the staging half is byte-for-byte the tile's face-pair range, so the writer can
//   blast it with one contiguous write. The u32 index staging doubles every term
//   (4 B elements, 1024 B faces, 64 B face rows), i.e. off32 = off16 << 1; u16 index
//   staging uses off16 unchanged.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "topk_route_finish_gather_common.hpp"

void kernel_main() {
    using namespace topk_route_finish;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);  // TILE bf16 logits
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);  // RM u32 index sticks
    const uint32_t start_unit = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t logical_rows = get_arg_val<uint32_t>(4);         // R
    const uint32_t row_tiles_per_batch = get_arg_val<uint32_t>(5);  // R_p / 32
    const uint32_t k_rounded = get_arg_val<uint32_t>(6);

    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);      // div_up(k_rounded, 32)
    constexpr uint32_t width_tiles = get_compile_time_arg_val(1);  // W_p / 32
    constexpr uint32_t cb_stick = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bounce = get_compile_time_arg_val(3);
    constexpr uint32_t cb_values = get_compile_time_arg_val(4);
    constexpr uint32_t cb_indices = get_compile_time_arg_val(5);
    constexpr bool index_is_u32 = get_compile_time_arg_val(6) == 1;
    constexpr auto src_args = TensorAccessorArgs<7>();
    constexpr auto idx_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    // Page sizes are baked compile-time by the host's TensorAccessorArgs (2048 B tiles /
    // k_rounded*4 B sticks).
    const auto src = TensorAccessor(src_args, src_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    Noc noc;
    DataflowBuffer dfb_stick(cb_stick);    // reader-private scratch: never pushed
    DataflowBuffer dfb_bounce(cb_bounce);  // reader-private scratch: never pushed
    DataflowBuffer dfb_values(cb_values);
    DataflowBuffer dfb_indices(cb_indices);

    // Scratch bases are fixed for the whole kernel (1-page CBs, nothing pushed). The CB
    // allocator aligns CB bases to the 64 B DRAM alignment, which the 64 B bounce slots
    // and 128 B stick rows rely on.
    const uint32_t stick_base = dfb_stick.get_write_ptr();
    const uint32_t bounce_base = dfb_bounce.get_write_ptr();
    const CoreLocalMem<uint32_t> stick_dst(stick_base);
    const CoreLocalMem<uint32_t> bounce_dst(bounce_base);
    volatile tt_l1_ptr uint32_t* const stick_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stick_base);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t row_tile = u / (k_tiles * 2);  // global (all batches)
        const uint32_t rem = u % (k_tiles * 2);
        const uint32_t kt = rem >> 1;
        const uint32_t half = rem & 1;

        const uint32_t batch = row_tile / row_tiles_per_batch;
        const uint32_t row_in_batch0 = (row_tile % row_tiles_per_batch) * 32 + half * half_rows;

        // Row clamp: rows at or past logical R are tile-height padding (stay zero). Units
        // that are ALL padding still run: they exist to zero their face-pair. This RISC
        // owns rows [0, 8) of the unit.
        const uint32_t rows_left = row_in_batch0 < logical_rows ? logical_rows - row_in_batch0 : 0;
        const uint32_t valid_rows = rows_left < half_rows ? rows_left : half_rows;
        const uint32_t my_rows = valid_rows < rows_per_risc ? valid_rows : rows_per_risc;
        // Column clamp: when k_rounded % 32 == 16, the last output tile's right face pair
        // is k-padding (stays zero).
        const uint32_t cols_left = k_rounded - kt * tile_width;
        const uint32_t valid_cols = cols_left < tile_width ? cols_left : tile_width;

        dfb_values.reserve_back(1);
        dfb_indices.reserve_back(1);
        const uint32_t val_base = dfb_values.get_write_ptr();
        const uint32_t idx_out_base = dfb_indices.get_write_ptr();

        // Issue this RISC's stick-segment reads first so their flight overlaps the
        // zero-fill below: row lr's stick is page batch*R + row_in_batch0 + lr; the unit's
        // columns start at u32 offset kt*32.
        for (uint32_t lr = 0; lr < my_rows; ++lr) {
            noc.async_read(
                idx,
                stick_dst,
                valid_cols * 4,
                {.page_id = batch * logical_rows + row_in_batch0 + lr, .offset_bytes = kt * stick_seg_bytes},
                {.offset_bytes = lr * stick_seg_bytes});
        }

        // Zero this RISC's row ranges of both staging halves. Covers all padding cases at
        // once (row padding, k-padding, R==0 units, garbage-guarding fresh CB pages); the
        // writer zeroes rows [8, 16) — every staging byte is zeroed by exactly one RISC.
        zero_half_rows<2>(val_base, 0, rows_per_risc);
        if constexpr (index_is_u32) {
            zero_half_rows<4>(idx_out_base, 0, rows_per_risc);
        } else {
            zero_half_rows<2>(idx_out_base, 0, rows_per_risc);
        }

        if (my_rows > 0) {
            // Plain barrier: only the stick reads are outstanding here (both gather trids
            // were drained before the previous unit's push).
            noc.async_read_barrier();
            gather_unit_rows<index_is_u32>(
                noc,
                src,
                bounce_dst,
                bounce_base,
                stick_l1,
                val_base,
                idx_out_base,
                row_tile,
                width_tiles,
                half,
                0,  // lr_begin: reader owns rows [0, 8)
                my_rows,
                valid_cols);
        }

        dfb_values.push_back(1);
        dfb_indices.push_back(1);
    }
}
