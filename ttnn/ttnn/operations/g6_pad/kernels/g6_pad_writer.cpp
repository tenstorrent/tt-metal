// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad new_storage writer (RISCV_0 / NOC1), tiles and rows (CT flag rows_mode).
// rows: unit = (output row, row chunk); N4 fills the row tail [d_w, D_w) of the last source chunk.
// Block operations (op_design.md, Block schedule / new_storage_tiles):
//   N1 make_fill_page   -- fill_l1_range of fill_bits into cb_fill_tile, once per kernel
//   N3 write_fill_tiles -- fill-only output tiles of the block written from cb_fill_tile BEFORE the
//                          wait on cb_input_tiles (they do not depend on the input: stall shadow)
//   N4 patch_block      -- wait block_tiles; in each edge slot RISC-store fill_bits into rows
//                          [edge_row, 32) (last input tile row) and/or cols [edge_col, 32) (last input
//                          tile col), face-aware, in place (bit-exact, no unpack/pack)
//   N5 store_block      -- write each source-bearing slot to its output page; writes_flushed; pop
//   N6 drain_writes     -- noc_async_write_barrier (BB-03)
// Every output page is written exactly once.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "g6_pad_common.hpp"

namespace {

constexpr uint32_t FACE = 16;

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_fill_tile = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(3);  // CB page: one tile, or one row-chunk slot
    constexpr uint32_t block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles_h = get_compile_time_arg_val(5);
    constexpr uint32_t out_tiles_w = get_compile_time_arg_val(6);
    constexpr uint32_t out_lead1 = get_compile_time_arg_val(7);
    constexpr uint32_t in_tiles_h = get_compile_time_arg_val(8);
    constexpr uint32_t in_tiles_w = get_compile_time_arg_val(9);
    constexpr uint32_t in_lead0 = get_compile_time_arg_val(10);
    constexpr uint32_t in_lead1 = get_compile_time_arg_val(11);
    constexpr uint32_t edge_row = get_compile_time_arg_val(12);
    constexpr uint32_t edge_col = get_compile_time_arg_val(13);
    constexpr bool rows_mode = get_compile_time_arg_val(14) != 0;      // new_storage_rows (ROW_MAJOR)
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(15);     // rows: bytes per row chunk
    constexpr uint32_t in_row_bytes = get_compile_time_arg_val(16);    // rows: logical input row bytes
    constexpr uint32_t out_row_bytes = get_compile_time_arg_val(17);   // rows: logical output row bytes
    constexpr uint32_t dst_page_bytes = get_compile_time_arg_val(18);  // aligned output page size
    constexpr auto dst_args = TensorAccessorArgs<19>();

    constexpr uint32_t row_bytes = FACE * elem_bytes;
    constexpr uint32_t face_bytes = FACE * row_bytes;
    static_assert(rows_mode || slot_bytes == 4 * face_bytes, "tile geometry");
    constexpr uint32_t NOC_ALIGN = 16;  // L1 NoC write alignment: sub-16 B heads are RISC stores

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t fill_bits = get_arg_val<uint32_t>(3);

    if (num_tiles == 0) {
        return;
    }

    const auto dst = TensorAccessor(dst_args, dst_addr, dst_page_bytes);
    using Walker =
        g6_pad::OutTileWalker<out_tiles_h, out_tiles_w, out_lead1, in_tiles_h, in_tiles_w, in_lead0, in_lead1>;

    // N1: fill page (scratch owned by this kernel only; never pushed).
    const uint32_t fill_base = get_write_ptr(cb_fill_tile);
    {
        DeviceZoneScopedN("make_fill_page");
        // RISC stores seed at most one tile's worth; a larger (row-chunk) page is grown by local NoC
        // doubling copies (tile pages are <= 4 KB, so the tiles path is unchanged: one RISC fill).
        constexpr uint32_t seed_bytes = slot_bytes < 4096 ? slot_bytes : 4096;
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(fill_base, seed_bytes, fill_bits);
        if constexpr (slot_bytes > seed_bytes) {
            for (uint32_t have = seed_bytes; have < slot_bytes; have *= 2) {
                const uint32_t n = (slot_bytes - have) < have ? (slot_bytes - have) : have;
                noc_async_write(fill_base, get_noc_addr(fill_base + have), n);
                noc_async_write_barrier();
            }
        }
    }

    // rows: byte extent of the current unit's chunk (the last chunk of a row may be short).
    auto chunk_len = [](const Walker& w) -> uint32_t {
        const uint32_t left = out_row_bytes - w.wt * chunk_bytes;
        return left < chunk_bytes ? left : chunk_bytes;
    };
    auto dst_noc_addr = [&](const Walker& w, uint32_t flat) -> uint64_t {
        if constexpr (rows_mode) {
            return dst.get_noc_addr(w.out_row(), w.wt * chunk_bytes);
        } else {
            return dst.get_noc_addr(flat);
        }
    };
    auto unit_bytes = [&](const Walker& w) -> uint32_t {
        if constexpr (rows_mode) {
            return chunk_len(w);
        } else {
            return slot_bytes;
        }
    };

    Walker walk(start_tile);
    uint32_t t = start_tile;
    const uint32_t num_blocks = (num_tiles + block_tiles - 1) / block_tiles;
    uint32_t remaining = num_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;

        // N3: fill-only tiles of this block, issued before the input wait.
        {
            DeviceZoneScopedN("write_fill_tiles");
            Walker w = walk;
            for (uint32_t k = 0; k < extent; ++k) {
                if (!w.source_bearing()) {
                    noc_async_write(fill_base, dst_noc_addr(w, t + k), unit_bytes(w));
                }
                w.advance();
            }
        }

        // N4: patch edge slots in place (zone includes the wait).
        {
            DeviceZoneScopedN("patch_block");
            cb_wait_front(cb_input_tiles, block_tiles);
            if constexpr (rows_mode) {
                // Tail [in_len, chunk_len) of the last source-bearing chunk of a row becomes c: RISC stores
                // up to the next 16 B boundary, then one local NoC copy from the same offset of the fill
                // page (both bases 64 B-aligned, so src == dst mod 16). Never a misaligned NoC transfer.
                if constexpr (in_row_bytes < out_row_bytes) {
                    const uint32_t slot_base = get_read_ptr(cb_input_tiles);
                    bool noc_patched = false;
                    Walker w = walk;
                    for (uint32_t k = 0; k < extent; ++k) {
                        if (w.source_bearing() && w.last_input_tile_col()) {
                            const uint32_t base = slot_base + k * slot_bytes;
                            const uint32_t in_len = in_row_bytes - w.wt * chunk_bytes;
                            const uint32_t len = chunk_len(w);
                            if (in_len < len) {
                                uint32_t head_end = (in_len + NOC_ALIGN - 1) & ~(NOC_ALIGN - 1);
                                head_end = head_end < len ? head_end : len;
                                if (head_end > in_len) {
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + in_len, head_end - in_len, fill_bits);
                                }
                                if (len > head_end) {
                                    noc_async_write(
                                        fill_base + head_end, get_noc_addr(base + head_end), len - head_end);
                                    noc_patched = true;
                                }
                            }
                        }
                        w.advance();
                    }
                    if (noc_patched) {
                        noc_async_write_barrier();
                    }
                }
            } else if constexpr (edge_row != 0 || edge_col != 0) {
                const uint32_t slot_base = get_read_ptr(cb_input_tiles);
                // Face-row-aligned runs are copied from the same offset of the fill page by local NoC
                // writes (DMA, 32 B-granular, src == dst mod 16); only sub-face-row column fragments
                // (2 B-granular for bf16) are RISC stores. Both write c, so overlap order is irrelevant.
                bool noc_patched = false;
                auto noc_fill = [&](uint32_t base, uint32_t off, uint32_t bytes) {
                    noc_async_write(fill_base + off, get_noc_addr(base + off), bytes);
                    noc_patched = true;
                };
                Walker w = walk;
                for (uint32_t k = 0; k < extent; ++k) {
                    if (w.source_bearing()) {
                        const uint32_t base = slot_base + k * slot_bytes;
                        if constexpr (edge_row != 0) {
                            if (w.last_input_tile_row()) {
                                if constexpr (edge_row < FACE) {
                                    // face 0 rows [edge_row,16), then face 1 rows [edge_row,16) + faces 2,3
                                    noc_fill(base, edge_row * row_bytes, (FACE - edge_row) * row_bytes);
                                    noc_fill(
                                        base,
                                        face_bytes + edge_row * row_bytes,
                                        (FACE - edge_row) * row_bytes + 2 * face_bytes);
                                } else {
                                    constexpr uint32_t r0 = edge_row - FACE;
                                    noc_fill(base, 2 * face_bytes + r0 * row_bytes, (FACE - r0) * row_bytes);
                                    noc_fill(base, 3 * face_bytes + r0 * row_bytes, (FACE - r0) * row_bytes);
                                }
                            }
                        }
                        if constexpr (edge_col != 0) {
                            if (w.last_input_tile_col()) {
                                constexpr uint32_t c0 = edge_col % FACE;
                                // faces holding columns [edge_col % 16, 16) partially: 0/2 if edge_col < 16
                                constexpr uint32_t part_a = edge_col < FACE ? 0 : face_bytes;
                                constexpr uint32_t part_b = part_a + 2 * face_bytes;
                                if constexpr (edge_col < FACE) {
                                    noc_fill(base, face_bytes, face_bytes);      // face 1 whole
                                    noc_fill(base, 3 * face_bytes, face_bytes);  // face 3 whole
                                }
                                for (uint32_t r = 0; r < FACE; ++r) {
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + part_a + r * row_bytes + c0 * elem_bytes,
                                        (FACE - c0) * elem_bytes,
                                        fill_bits);
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + part_b + r * row_bytes + c0 * elem_bytes,
                                        (FACE - c0) * elem_bytes,
                                        fill_bits);
                                }
                            }
                        }
                    }
                    w.advance();
                }
                if (noc_patched) {
                    // Local patch writes must land before the slot is sourced by store_block.
                    noc_async_write_barrier();
                }
            }
        }

        // N5: store source-bearing slots.
        {
            DeviceZoneScopedN("store_block");
            const uint32_t slot_base = get_read_ptr(cb_input_tiles);
            for (uint32_t k = 0; k < extent; ++k) {
                if (walk.source_bearing()) {
                    noc_async_write(slot_base + k * slot_bytes, dst_noc_addr(walk, t + k), unit_bytes(walk));
                }
                walk.advance();
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_input_tiles, block_tiles);
        }
        t += extent;
        remaining -= extent;
    }

    {
        DeviceZoneScopedN("drain_writes");
        noc_async_write_barrier();
    }
}
