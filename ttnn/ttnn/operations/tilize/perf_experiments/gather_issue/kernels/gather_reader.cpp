// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH KERNEL — the cross-core L1 gather, and nothing else.
//
// One NCRISC reader per destination core. It gathers `num_blocks` blocks of
// `tile_h` source rows x `row_bytes` bytes out of a ROW_MAJOR source whose shard
// is NARROWER than a tensor row (so a row is `row_pages` pages living on
// `row_pages` DIFFERENT cores' L1), and lays each block down in this core's own
// L1 at stride `row_bytes` — byte-for-byte the destination layout the production
// reader produces in its input CB.
//
// No CB handshake, no compute, no writer: the ONLY thing that varies between
// variants is how the read transactions are addressed and issued, so the
// measured delta is attributable to that alone.
//
// RAW DATAFLOW, deliberately: the library helper
// `dataflow_kernel_lib::read_sticks_for_tilize` walks CONSECUTIVE PAGE IDS as
// consecutive sticks, which is exactly the identity a narrower-than-a-row shard
// breaks (page id p -> shard p % row_pages, row p / row_pages), so the gather
// cannot be expressed through it at all. That is why the production op also hand
// -rolls this loop; this bench reconstructs it faithfully as VARIANT 0.
//
// Variants (compile-time arg 0):
//   0 BASELINE  — the production R_PAD inner loop: per source row, per page
//                 slice, `accessor.get_noc_addr(page, off)` then a read.
//   1 HOIST     — the address math is hoisted out of the row loop: one
//                 `get_noc_addr` per (block, page-slice column), then step the
//                 NoC address by the source page stride. Loop order is inverted
//                 so every row of ONE source shard is issued back-to-back.
//   2 COALESCE  — only expressible when `row_bytes == page_bytes` (the block is
//                 exactly one source shard's width): the block's `tile_h` source
//                 rows are then `tile_h` CONSECUTIVE pages of one shard, and the
//                 destination stride equals the transfer size, so the whole
//                 block is ONE contiguous transfer.
//   4 STRIP     — one wide contiguous transfer per (block, source shard),
//                 straight into a STRIP-MAJOR block slot. Coalesce's fabric
//                 behaviour without coalesce's block-width precondition.
//   3 SCRATCH   — one wide contiguous transfer per (block, shard) into an L1
//                 scratch, then a local L1 word permutation into place. Prices
//                 "fewest possible NoC transactions, paid for in RISC copies".

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

// Local L1 word copy — same mechanism the production retile path uses for its
// face-row moves (a volatile 32-bit load/store loop; there is no DMA engine for
// L1->L1 on this hardware).
template <uint32_t n_bytes>
FORCE_INLINE void copy_l1_words(uint32_t src_addr, uint32_t dst_addr) {
    static_assert(n_bytes % 4 == 0, "L1 word copy needs a 4-byte multiple");
    auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
#pragma GCC unroll 8
    for (uint32_t i = 0; i < n_bytes / 4; ++i) {
        dst[i] = src[i];
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t tile_h = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t row_pages = get_compile_time_arg_val(4);  // pages per TENSOR row
    constexpr uint32_t n_chunks = get_compile_time_arg_val(5);   // W chunks per tile-row
    constexpr uint32_t scratch_cb = 0;
    constexpr auto src_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_row0 = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);

    constexpr uint32_t block_bytes = tile_h * row_bytes;
    // Slices one source row is cut into for a block of `row_bytes`. Every block in
    // this bench starts on a page boundary (col_off is a multiple of row_bytes and
    // row_bytes is a multiple of, or a divisor of, page_bytes), so this is exact.
    constexpr uint32_t slices = row_bytes >= page_bytes ? row_bytes / page_bytes : 1;
    constexpr uint32_t slice_bytes = row_bytes >= page_bytes ? page_bytes : row_bytes;
    // Both wide-transfer variants need a source that is CONTIGUOUS across the
    // block's rows, i.e. the slice must be a whole page. The host only selects
    // them where that holds; assert it here so a mis-selection is a compile error
    // rather than a silently shifted gather.
    static_assert(variant != 2 || row_bytes == page_bytes, "coalesce needs block width == source shard width");
    static_assert(variant != 3 || slice_bytes == page_bytes, "scratch needs whole-page slices");
    static_assert(variant != 4 || slice_bytes == page_bytes, "strip needs whole-page slices");

    const auto accessor = TensorAccessor(src_args, src_addr);

    // The byte distance between two source rows INSIDE one shard. Advancing the
    // page id by `row_pages` moves one row down within the SAME shard (page p ->
    // shard p % row_pages, local index p / row_pages), so this is the shard's
    // aligned page size — derived from the accessor itself rather than assumed,
    // and both endpoints are on the same core so the NoC xy bits cancel.
    const uint64_t page_stride = accessor.get_noc_addr(row_pages, 0) - accessor.get_noc_addr(0u, 0);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t r_blk = b / n_chunks;
        const uint32_t c_blk = b - r_blk * n_chunks;
        const uint32_t row0 = (tile_row0 + r_blk) * tile_h;
        const uint32_t col_off = c_blk * row_bytes;
        const uint32_t dst = dst_addr + b * block_bytes;

        if constexpr (variant == 0) {
            // ── BASELINE: the production R_PAD loop, verbatim in structure ──
            MaybeDeviceZoneScope("gather_issue");
            uint32_t l1_addr = dst;
            for (uint32_t r = 0; r < tile_h; ++r) {
                const uint32_t row = row0 + r;
                uint32_t page_in_row = col_off / page_bytes;
                uint32_t page = row * row_pages + page_in_row;
                uint32_t off = col_off - page_in_row * page_bytes;
                uint32_t n_bytes = row_bytes;
                uint32_t a = l1_addr;
                while (n_bytes > 0) {
                    uint32_t n = page_bytes - off;
                    if (n > n_bytes) {
                        n = n_bytes;
                    }
                    noc_async_read(accessor.get_noc_addr(page, off), a, n);
                    a += n;
                    n_bytes -= n;
                    ++page;
                    off = 0;
                }
                l1_addr += row_bytes;
            }
        } else if constexpr (variant == 1) {
            // ── HOIST: one address computation per (block, slice) ───────────
            MaybeDeviceZoneScope("gather_issue");
            const uint32_t page_in_row0 = col_off / page_bytes;
            const uint32_t off0 = col_off - page_in_row0 * page_bytes;
#pragma GCC unroll 1
            for (uint32_t s = 0; s < slices; ++s) {
                // One shard, one endpoint, one base address — then just step it.
                uint64_t src = accessor.get_noc_addr(row0 * row_pages + page_in_row0 + s, s == 0 ? off0 : 0);
                uint32_t a = dst + s * slice_bytes;
                for (uint32_t r = 0; r < tile_h; ++r) {
                    noc_async_read(src, a, slice_bytes);
                    src += page_stride;
                    a += row_bytes;
                }
            }
        } else if constexpr (variant == 2) {
            // ── COALESCE: the whole block is one contiguous transfer ────────
            MaybeDeviceZoneScope("gather_issue");
            noc_async_read(accessor.get_noc_addr(row0 * row_pages + col_off / page_bytes, 0), dst, block_bytes);
        } else if constexpr (variant == 4) {
            // ── STRIP: one contiguous transfer per (block, source shard) ────
            // Same fabric behaviour as COALESCE but at the op's CURRENT block
            // width: the block's L1 slot is laid out STRIP-major (all `tile_h`
            // rows of source shard 0, then all of shard 1, ...) instead of
            // row-major, which makes each strip both contiguous at the source
            // AND contiguous at the destination. The consumer then tilizes
            // `slices` sub-blocks of `page_bytes` width per slot instead of one
            // block of `row_bytes` — the SAME tile sequence, since strip s holds
            // exactly tile-columns [s*page_tiles, (s+1)*page_tiles).
            MaybeDeviceZoneScope("gather_issue");
            const uint32_t pir0 = col_off / page_bytes;
            for (uint32_t s = 0; s < slices; ++s) {
                noc_async_read(
                    accessor.get_noc_addr(row0 * row_pages + pir0 + s, 0),
                    dst + s * tile_h * slice_bytes,
                    tile_h * slice_bytes);
            }
        } else {
            // ── SCRATCH: fewest NoC transactions, paid in RISC word copies ──
            const uint32_t scratch = get_write_ptr(scratch_cb);
            {
                MaybeDeviceZoneScope("gather_issue");
                for (uint32_t s = 0; s < slices; ++s) {
                    noc_async_read(
                        accessor.get_noc_addr(row0 * row_pages + col_off / page_bytes + s, 0),
                        scratch + s * tile_h * slice_bytes,
                        tile_h * slice_bytes);
                }
            }
            {
                MaybeDeviceZoneScope("gather_barrier");
                noc_async_read_barrier();
            }
            {
                MaybeDeviceZoneScope("gather_permute");
                for (uint32_t s = 0; s < slices; ++s) {
                    uint32_t a = scratch + s * tile_h * slice_bytes;
                    uint32_t d = dst + s * slice_bytes;
                    for (uint32_t r = 0; r < tile_h; ++r) {
                        copy_l1_words<slice_bytes>(a, d);
                        a += slice_bytes;
                        d += row_bytes;
                    }
                }
            }
            continue;  // its barrier already ran
        }

        {
            MaybeDeviceZoneScope("gather_barrier");
            noc_async_read_barrier();  // ONE barrier per block (the op's policy)
        }
    }
}
