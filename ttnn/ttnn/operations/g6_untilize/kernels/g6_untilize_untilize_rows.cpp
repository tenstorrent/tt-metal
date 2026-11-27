// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// untilize_rows: gather-on-read untilize, the op's only kernel (regime R1 `gather_rows`).
//
// Every face-row piece (16 elements, contiguous in both the TILE input and the ROW_MAJOR output) is read from
// DRAM straight to its row-major position in cb_row_staging, then each staged row is written to its output page
// with one NoC write. Bytes never enter an engine register, so every 32-bit pattern survives.
//
// Raw NoC instead of kernel_lib helpers: no dataflow helper gathers face rows of tile pages into row-major sticks
// (write_sticks_after_untilize consumes whole tile rows through a CB handshake; untilize() routes through SrcA,
// where UInt32 is not representable). See op_design.md, "Helpers considered and rejected".
//
// cb_row_staging is scratch owned by this one RISC: no CB push/pop. Ordering comes from NoC barriers only.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    // Rectangle and work split (plan()).
    constexpr uint32_t x0 = get_compile_time_arg_val(0);
    constexpr uint32_t y0 = get_compile_time_arg_val(1);
    constexpr uint32_t cx = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores = get_compile_time_arg_val(3);
    constexpr uint32_t total_rows = get_compile_time_arg_val(4);
    // Geometry.
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t W = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t elem = get_compile_time_arg_val(9);
    // Block knobs.
    constexpr uint32_t block_rows = get_compile_time_arg_val(10);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(11);
    constexpr uint32_t staging_depth = get_compile_time_arg_val(12);
    constexpr bool rotate_tile_order = get_compile_time_arg_val(13) != 0;
    // Page sizes the accessors index banks with.
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t cb_row_staging = get_compile_time_arg_val(16);
    constexpr auto in_args = TensorAccessorArgs<17>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile_w = 32;
    constexpr uint32_t tile_h = 32;
    constexpr uint32_t face_w = 16;
    constexpr uint32_t face_h = 16;
    constexpr uint32_t face_row_bytes = face_w * elem;
    constexpr uint32_t face_bytes = face_h * face_row_bytes;
    constexpr uint32_t tile_col_bytes = tile_w * elem;
    constexpr uint32_t staged_row_bytes = block_tile_cols * tile_col_bytes;
    constexpr uint32_t row_bytes = W * elem;
    constexpr uint32_t num_col_blocks = (Wt + block_tile_cols - 1) / block_tile_cols;
    // Output addresses computed before the read barrier: bounded so a large block_rows costs no stack.
    constexpr uint32_t addr_prefetch = block_rows < 8 ? block_rows : 8;
    static_assert(Ht * tile_h >= H, "Ht must cover H");

    // Balanced contiguous row runs over the rectangle: core i owns [start, start + n).
    constexpr uint32_t q = total_rows / num_cores;
    constexpr uint32_t rem = total_rows % num_cores;
    const uint32_t core_idx = (get_absolute_logical_x() - x0) + (get_absolute_logical_y() - y0) * cx;
    const uint32_t row_start = core_idx * q + (core_idx < rem ? core_idx : rem);
    const uint32_t num_rows = q + (core_idx < rem ? 1 : 0);
    if (num_rows == 0) {
        return;
    }
    const uint32_t num_row_blocks = (num_rows + block_rows - 1) / block_rows;
    const uint32_t num_blocks = num_row_blocks * num_col_blocks;
    // Different cores start on different tile columns so their first reads hit different DRAM banks.
    const uint32_t rot = rotate_tile_order ? core_idx % block_tile_cols : 0;

    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), in_page_bytes);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), out_page_bytes);
    const uint32_t staging_base = get_write_ptr(cb_row_staging);

    uint32_t block_idx = 0;
    for (uint32_t rb = 0; rb < num_row_blocks; ++rb) {
        const uint32_t rho0 = row_start + rb * block_rows;
        const uint32_t rows_this_block = (rb + 1 < num_row_blocks) ? block_rows : num_rows - rb * block_rows;
        for (uint32_t col_block = 0; col_block < num_col_blocks; ++col_block, ++block_idx) {
            const uint32_t c0 = col_block * block_tile_cols;
            const uint32_t cols_this_block = (c0 + block_tile_cols <= Wt) ? block_tile_cols : Wt - c0;
            const uint32_t slot = staging_base + (block_idx % staging_depth) * block_rows * staged_row_bytes;
            const uint32_t col_offset_bytes = c0 * tile_col_bytes;
            const uint32_t store_bytes =
                (col_offset_bytes + staged_row_bytes <= row_bytes) ? staged_row_bytes : row_bytes - col_offset_bytes;

            // Single slot: the previous block's store reads it until its writes have left L1.
            if constexpr (staging_depth == 1) {
                if (block_idx > 0) {
                    DeviceZoneScopedN("slot-flush-wait");
                    noc_async_writes_flushed();
                }
            }

            // gather_block: column-outer so each tile page's NoC address is computed once per tile row touched.
            {
                DeviceZoneScopedN("gather-issue");
                uint32_t c_rel = rot % cols_this_block;
                for (uint32_t k = 0; k < cols_this_block; ++k) {
                    const uint32_t c = c0 + c_rel;
                    const uint32_t dst_col = slot + c_rel * tile_col_bytes;
                    uint32_t cached_tile_row = UINT32_MAX;
                    uint64_t tile_noc = 0;
                    for (uint32_t i = 0; i < rows_this_block; ++i) {
                        const uint32_t rho = rho0 + i;
                        const uint32_t b = rho / H;
                        const uint32_t r = rho - b * H;
                        const uint32_t tile_row = b * Ht + r / tile_h;
                        if (tile_row != cached_tile_row) {
                            cached_tile_row = tile_row;
                            tile_noc = in.get_noc_addr(tile_row * Wt + c);
                        }
                        const uint32_t band = (r % tile_h) / face_h;
                        const uint32_t piece_src = 2 * band * face_bytes + (r % face_h) * face_row_bytes;
                        const uint32_t dst = dst_col + i * staged_row_bytes;
                        noc_async_read_one_packet(tile_noc + piece_src, dst, face_row_bytes);
                        // A piece starting at or past W holds only tile padding: never read, never written.
                        if (c * tile_w + face_w < W) {
                            noc_async_read_one_packet(
                                tile_noc + piece_src + face_bytes, dst + face_row_bytes, face_row_bytes);
                        }
                    }
                    c_rel = (c_rel + 1 == cols_this_block) ? 0 : c_rel + 1;
                }
            }

            // Output addresses do not depend on the reads: compute them in the read round trip's shadow.
            uint64_t out_noc[addr_prefetch];
            const uint32_t prefetched = rows_this_block < addr_prefetch ? rows_this_block : addr_prefetch;
            for (uint32_t i = 0; i < prefetched; ++i) {
                out_noc[i] = out.get_noc_addr(rho0 + i, col_offset_bytes);
            }

            {
                DeviceZoneScopedN("gather-wait");
                noc_async_read_barrier();
            }

            // Two or more slots: flushing store(k-1) here frees its slot for gather(k+1) while store(k) runs.
            if constexpr (staging_depth > 1) {
                if (block_idx + 1 >= staging_depth && block_idx + 1 < num_blocks) {
                    DeviceZoneScopedN("slot-flush-wait");
                    noc_async_writes_flushed();
                }
            }

            // store_block: one write per staged row.
            {
                DeviceZoneScopedN("store-issue");
                for (uint32_t i = 0; i < rows_this_block; ++i) {
                    const uint64_t dst_noc = i < prefetched ? out_noc[i] : out.get_noc_addr(rho0 + i, col_offset_bytes);
                    noc_async_write(slot + i * staged_row_bytes, dst_noc, store_bytes);
                }
            }
        }
    }

    // drain: every non-posted write acked before the next program may read the output.
    {
        DeviceZoneScopedN("drain-wait");
        noc_async_write_barrier();
    }
}
