// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// topk_route_finish writer: gather THIS RISC'S SHARE of the per-unit load (unit rows
// [8, 16); the reader owns rows [0, 8) — see topk_route_finish_gather_common.hpp for the
// row-disjointness argument and the shared trid double-wave pipeline), then drain the
// completed staged face-pair halves into the two TILE outputs. Each staged page is
// byte-for-byte the output tile's face-pair range (see the reader's face-math comment), so
// a unit is exactly two contiguous page writes:
//
//   unit(row_tile, kt, half) -> output page row_tile * k_tiles + kt,
//     values:  1024 B at byte offset half * 1024 (bf16 tile page = 2048 B)
//     indices: index_half_bytes at offset half * index_half_bytes (u16 page = 2048 B,
//              u32 page = 4096 B)
//
// Both offsets are multiples of 16 B, satisfying the (write-side) DRAM/L1 NoC alignment.
//
// Split protocol (why this cannot deadlock or race): per unit u this kernel
//   1. computes the unit's staging page addresses BEFORE wait_front — get_read_ptr() is
//      plain local pointer state (fifo_rd_ptr), it never blocks, and it already points at
//      the page unit u will occupy: both kernels walk the same unit order, this kernel is
//      the CBs' only consumer, and its own pop of unit u-2 is what freed that page (units
//      0 and 1 use never-touched pages). Writing into it before the reader's push is safe
//      because everything this kernel writes (its zero-fill ranges and its gather stores)
//      lives in rows [8, 16) — 32 B face rows the reader never touches (the reader's
//      reserve_back only moves credits, it writes no bytes);
//   2. zeroes its row ranges, reads its own <=8 index-stick segments into private scratch,
//      and gathers rows [8, 16) into the staging page (per-trid barriers inside
//      gather_unit_rows drain all of its reads before returning);
//   3. THEN calls wait_front — the reader's push guarantees rows [0, 8) and their zero
//      fill are complete — and only then issues the output writes, write-barriers, pops.
// The only cross-RISC blocking edges are the single producer/consumer pair: the reader
// blocks only in reserve_back (waiting on this kernel's pops) and its own NoC barriers
// (hardware-bounded); this kernel blocks only in wait_front (waiting on the reader's
// pushes) and its own NoC barriers. The gather happens strictly BEFORE wait_front and
// blocks only on hardware-bounded barriers, so by induction on the unit index every unit
// terminates — no wait cycle exists.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "topk_route_finish_gather_common.hpp"

void kernel_main() {
    using namespace topk_route_finish;

    const uint32_t values_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_unit = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t src_addr = get_arg_val<uint32_t>(4);             // TILE bf16 logits
    const uint32_t idx_addr = get_arg_val<uint32_t>(5);             // RM u32 index sticks
    const uint32_t logical_rows = get_arg_val<uint32_t>(6);         // R
    const uint32_t row_tiles_per_batch = get_arg_val<uint32_t>(7);  // R_p / 32
    const uint32_t k_rounded = get_arg_val<uint32_t>(8);

    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(1);  // W_p / 32
    constexpr uint32_t cb_values = get_compile_time_arg_val(2);
    constexpr uint32_t cb_indices = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stick = get_compile_time_arg_val(4);          // writer-private scratch
    constexpr uint32_t cb_bounce = get_compile_time_arg_val(5);         // writer-private scratch
    constexpr uint32_t value_half_bytes = get_compile_time_arg_val(6);  // 1024
    constexpr uint32_t index_half_bytes = get_compile_time_arg_val(7);  // 1024 (u16) / 2048 (u32)
    constexpr bool index_is_u32 = get_compile_time_arg_val(8) == 1;
    constexpr auto values_args = TensorAccessorArgs<9>();
    constexpr auto indices_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
    constexpr auto src_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    // Page sizes (2048 B bf16 / 2048 or 4096 B index tiles, 2048 B source tiles /
    // k_rounded*4 B sticks) are baked compile-time by the host's TensorAccessorArgs.
    const auto values_out = TensorAccessor(values_args, values_addr);
    const auto indices_out = TensorAccessor(indices_args, indices_addr);
    const auto src = TensorAccessor(src_args, src_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    Noc noc;
    DataflowBuffer dfb_values(cb_values);
    DataflowBuffer dfb_indices(cb_indices);
    DataflowBuffer dfb_stick(cb_stick);    // writer-private scratch: never pushed
    DataflowBuffer dfb_bounce(cb_bounce);  // writer-private scratch: never pushed

    // Scratch bases are fixed for the whole kernel (1-page CBs, nothing pushed); CB bases
    // are 64 B aligned, which the bounce slots and 128 B stick rows rely on.
    const uint32_t stick_base = dfb_stick.get_write_ptr();
    const uint32_t bounce_base = dfb_bounce.get_write_ptr();
    const CoreLocalMem<uint32_t> stick_dst(stick_base);
    const CoreLocalMem<uint32_t> bounce_dst(bounce_base);
    volatile tt_l1_ptr uint32_t* const stick_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stick_base);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t row_tile = u / (k_tiles * 2);
        const uint32_t rem = u % (k_tiles * 2);
        const uint32_t kt = rem >> 1;
        const uint32_t half = rem & 1;
        const uint32_t page = row_tile * k_tiles + kt;

        // Same clamps as the reader; this RISC owns rows [8, 16) of the unit.
        const uint32_t batch = row_tile / row_tiles_per_batch;
        const uint32_t row_in_batch0 = (row_tile % row_tiles_per_batch) * 32 + half * half_rows;
        const uint32_t rows_left = row_in_batch0 < logical_rows ? logical_rows - row_in_batch0 : 0;
        const uint32_t valid_rows = rows_left < half_rows ? rows_left : half_rows;
        const uint32_t my_rows = valid_rows > rows_per_risc ? valid_rows - rows_per_risc : 0;
        const uint32_t cols_left = k_rounded - kt * tile_width;
        const uint32_t valid_cols = cols_left < tile_width ? cols_left : tile_width;

        // Unit u's staging page addresses, read BEFORE wait_front (see the split-protocol
        // comment at the top for why this is safe).
        const uint32_t val_base = dfb_values.get_read_ptr();
        const uint32_t idx_out_base = dfb_indices.get_read_ptr();

        // Issue this RISC's stick-segment reads (rows [8, 8+my_rows), locally indexed from
        // 0 in the private scratch) so their flight overlaps the zero-fill below.
        for (uint32_t j = 0; j < my_rows; ++j) {
            noc.async_read(
                idx,
                stick_dst,
                valid_cols * 4,
                {.page_id = batch * logical_rows + row_in_batch0 + rows_per_risc + j,
                 .offset_bytes = kt * stick_seg_bytes},
                {.offset_bytes = j * stick_seg_bytes});
        }

        // Zero this RISC's row ranges [8, 16) of both staging halves (the reader zeroes
        // [0, 8) — every staging byte is zeroed by exactly one RISC).
        zero_half_rows<2>(val_base, rows_per_risc, half_rows);
        if constexpr (index_is_u32) {
            zero_half_rows<4>(idx_out_base, rows_per_risc, half_rows);
        } else {
            zero_half_rows<2>(idx_out_base, rows_per_risc, half_rows);
        }

        if (my_rows > 0) {
            // Plain barrier: only the stick reads are outstanding here (both gather trids
            // were drained before the previous unit's output writes, and the previous
            // unit's writes were write-barriered before its pop).
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
                rows_per_risc,  // lr_begin: writer owns rows [8, 16)
                my_rows,
                valid_cols);
        }

        // The reader's push guarantees rows [0, 8) and their zero fill are complete; this
        // kernel's own rows [8, 16) were completed above. The staged halves are now whole.
        dfb_values.wait_front(1);
        dfb_indices.wait_front(1);

        noc.async_write(
            CoreLocalMem<uint32_t>(val_base),
            values_out,
            value_half_bytes,
            {.offset_bytes = 0},
            {.page_id = page, .offset_bytes = half * value_half_bytes});
        noc.async_write(
            CoreLocalMem<uint32_t>(idx_out_base),
            indices_out,
            index_half_bytes,
            {.offset_bytes = 0},
            {.page_id = page, .offset_bytes = half * index_half_bytes});

        // Both staged pages are about to be recycled by the reader; the writes must have
        // fully landed before the credits go back.
        noc.async_write_barrier();
        dfb_values.pop_front(1);
        dfb_indices.pop_front(1);
    }
}
