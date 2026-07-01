// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// transpose_rm_reader.cpp — BRISC0 / reader for transpose_rm.
//
// For each work unit u ∈ [base, base + num_units) this kernel:
//   1. Decodes the linear index into (b, tile_a, tile_c) using compile-
//      time A_TILES, C_TILES.
//   2. Gathers a 32×32 fp32/bf16 sub-block from src tensor (B, A, C):
//      32 small NoC reads, one per source row.
//   3. Does an in-place row↔column swap of the L1 block so the writer
//      can emit it as 32 contiguous destination rows.
//
// Per-row addressing uses InterleavedAddrGen<true>::get_noc_addr(row, offset)
// where offset = tile_c*T_BLOCK*elem_size — gives us 1 NoC read per row
// segment without needing per-element gather.  Same bank-stride safety
// as batch_fft (per-bank stride = aligned_page_size, not tile size).
//
// Runtime args:
//   0: src_addr
//   1: base_unit             (first work unit linear idx)
//   2: num_units             (work units this core handles)
//   3: src_page_size_bytes   (= C * elem_size, ROW_MAJOR row stride)
// Compile-time args:
//   0: A_TILES               (= A / T_BLOCK)
//   1: C_TILES               (= C / T_BLOCK)
//   2: IS_BF16               (0 = fp32, 1 = bf16)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "transpose_rm_common.h"

void kernel_main() {
    const uint32_t src_addr           = get_arg_val<uint32_t>(0);
    const uint32_t base_unit          = get_arg_val<uint32_t>(1);
    const uint32_t num_units          = get_arg_val<uint32_t>(2);
    const uint32_t src_page_size_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t A_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t C_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t IS_BF16 = get_compile_time_arg_val(2);

    constexpr uint32_t elem_bytes  = IS_BF16 ? 2u : 4u;
    constexpr uint32_t row_bytes   = T_BLOCK * elem_bytes;   // 128B fp32, 64B bf16
    constexpr uint32_t block_bytes = T_BLOCK * row_bytes;    // one block in L1

    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr, .page_size = src_page_size_bytes};

    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t unit_idx = base_unit + u;
        // Linear → (b, tile_a, tile_c).  Inner loop is over tile_c so
        // consecutive work units operate on the same row-of-tiles which
        // helps DRAM bank locality for the source reads.
        const uint32_t tile_c =  unit_idx %  C_TILES;
        const uint32_t tile_a = (unit_idx /  C_TILES) % A_TILES;
        const uint32_t b      =  unit_idx / (C_TILES * A_TILES);

        const uint32_t src_row_base    = b * (A_TILES * T_BLOCK) + tile_a * T_BLOCK;
        const uint32_t src_col_offset  = tile_c * T_BLOCK * elem_bytes;

        cb_reserve_back(CB_TR_BLOCK, 1);
        const uint32_t l1_base = get_write_ptr(CB_TR_BLOCK);

        // 32 row-segment reads (each = 32 elements).  All in-flight at
        // once — NoC tolerates many outstanding requests; single barrier
        // at the end.
        for (uint32_t r = 0; r < T_BLOCK; ++r) {
            const uint32_t src_row = src_row_base + r;
            const uint64_t src_noc_addr = src_gen.get_noc_addr(src_row, src_col_offset);
            noc_async_read(src_noc_addr, l1_base + r * row_bytes, row_bytes);
        }
        noc_async_read_barrier();

        // In-place transpose of the 32×32 L1 block.
        //   L1[i*T_BLOCK + j] ↔ L1[j*T_BLOCK + i]   for i < j
        // Pure scalar BRISC code; ~496 swaps per block ≈ <2 µs.
        if constexpr (IS_BF16) {
            volatile tt_l1_ptr uint16_t* const buf =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_base);
            for (uint32_t i = 0; i < T_BLOCK; ++i) {
                for (uint32_t j = i + 1; j < T_BLOCK; ++j) {
                    const uint32_t a = i * T_BLOCK + j;
                    const uint32_t bidx = j * T_BLOCK + i;
                    const uint16_t tmp = buf[a];
                    buf[a]    = buf[bidx];
                    buf[bidx] = tmp;
                }
            }
        } else {
            volatile tt_l1_ptr uint32_t* const buf =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base);
            for (uint32_t i = 0; i < T_BLOCK; ++i) {
                for (uint32_t j = i + 1; j < T_BLOCK; ++j) {
                    const uint32_t a = i * T_BLOCK + j;
                    const uint32_t bidx = j * T_BLOCK + i;
                    const uint32_t tmp = buf[a];
                    buf[a]    = buf[bidx];
                    buf[bidx] = tmp;
                }
            }
        }

        cb_push_back(CB_TR_BLOCK, 1);
    }
}
