// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_concat reader (NCRISC, NoC0).
//
// Output-stationary: this core owns output pages [core_start_page, core_start_page + core_pages),
// processed in blocks of `block_pages`. Each block is reserved/pushed NOMINALLY (block_pages) in
// cb_out_tiles; a ragged last block only narrows the NoC transfers.
//
//   page_relocation: load_block        — every output page = one whole input page
//   row_gather:      stage_rows_block  — aligned DRAM windows of each row piece -> cb_row_stage
//                    place_rows_block  — L1->L1 self-reads stage -> cb_out_tiles slots; rows >= N zeroed
//
// Raw dataflow API only (no helper covers a K-source page map / sub-tile row pieces; see
// op_design.md "Helpers considered and rejected"). The L1->L1 self-read source uses
// get_noc_addr(local_addr) on this RISC's NoC (equivalent of dataflow_kernel_lib::local_addr for
// the legacy noc_async_read API).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/alignment.h"

constexpr uint32_t is_row_gather = get_compile_time_arg_val(0);
constexpr uint32_t num_inputs = get_compile_time_arg_val(1);
constexpr uint32_t tensor_slices = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t out_h_tiles = get_compile_time_arg_val(4);
constexpr uint32_t total_rows = get_compile_time_arg_val(5);
constexpr uint32_t block_pages = get_compile_time_arg_val(6);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
constexpr uint32_t dram_align = get_compile_time_arg_val(8);
constexpr uint32_t face_bytes = get_compile_time_arg_val(9);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(10);
constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(11);
constexpr uint32_t cb_row_stage = get_compile_time_arg_val(12);
constexpr uint32_t per_input_base = 13;  // per_input[k] at kernel_compile_time_args[13 + k]
constexpr auto input_args = TensorAccessorArgs<per_input_base + num_inputs>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;
// Output pages per slice (row_gather); clamped to 1 so the page_relocation build has no /0.
constexpr uint32_t pages_per_slice = out_h_tiles > 0 ? out_h_tiles* Wt : 1;
constexpr uint32_t stage_tile_bytes = 2 * tile_bytes;  // STAGE_PAGES_PER_TILE * tile_bytes

FORCE_INLINE uint32_t per_input(uint32_t k) { return kernel_compile_time_args[per_input_base + k]; }

FORCE_INLINE auto input_accessor(uint32_t k) {
    return TensorAccessor(input_args, get_common_arg_val<uint32_t>(k), tile_bytes);
}

// ----------------------------------------------------------------------------------------------
// page_relocation
// ----------------------------------------------------------------------------------------------
FORCE_INLINE void run_page_relocation(uint32_t core_start_page, uint32_t core_pages) {
    uint32_t R = 0;
    for (uint32_t k = 0; k < num_inputs; ++k) {
        R += per_input(k);
    }
    // Cursor (p, k, j) from core_start_page.
    uint32_t p = core_start_page / R;
    uint32_t rem = core_start_page - p * R;
    uint32_t k = 0;
    while (rem >= per_input(k)) {
        rem -= per_input(k);
        ++k;
    }
    uint32_t j = rem;

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    uint32_t remaining = core_pages;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t extent = remaining < block_pages ? remaining : block_pages;
        remaining -= extent;
        {
            DeviceZoneScopedN("reader_reserve");
            cb_reserve_back(cb_out_tiles, block_pages);
        }
        {
            DeviceZoneScopedN("load_block");
            uint32_t l1_addr = get_write_ptr(cb_out_tiles);
            uint32_t i = 0;
            while (i < extent) {
                const uint32_t run = per_input(k);
                uint32_t n = run - j;
                if (n > extent - i) {
                    n = extent - i;
                }
                const auto acc = input_accessor(k);
                uint32_t page = p * run + j;
                for (uint32_t t = 0; t < n; ++t) {
                    noc_async_read_page(page + t, acc, l1_addr);
                    l1_addr += tile_bytes;
                }
                i += n;
                j += n;
                if (j == run) {
                    j = 0;
                    if (++k == num_inputs) {
                        k = 0;
                        ++p;
                    }
                }
            }
        }
        {
            DeviceZoneScopedN("reader_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_out_tiles, block_pages);
    }
}

// ----------------------------------------------------------------------------------------------
// row_gather
// ----------------------------------------------------------------------------------------------
// Walk every row piece of output tile (p, ht, wt). A piece is a maximal run of m rows of one input
// with constant src half (s/16), dst half (t/16) and input tile. For each piece and each column face
// c in {0,1}, the callback gets (k, src_page, src_off, dst_off, len).
template <typename F>
FORCE_INLINE void for_each_piece(uint32_t p, uint32_t ht, uint32_t wt, F&& f) {
    const uint32_t r0 = ht * TILE_ROWS;
    uint32_t r1 = r0 + TILE_ROWS;
    if (r1 > total_rows) {
        r1 = total_rows;
    }
    uint32_t o_k = 0;
    for (uint32_t k = 0; k < num_inputs; ++k) {
        const uint32_t n_k = per_input(k);
        const uint32_t end_k = o_k + n_k;
        if (end_k > r0) {
            if (o_k >= r1) {
                break;
            }
            const uint32_t in_h_tiles = (n_k + TILE_ROWS - 1) / TILE_ROWS;
            uint32_t rho = o_k > r0 ? o_k : r0;
            const uint32_t b = end_k < r1 ? end_k : r1;
            while (rho < b) {
                const uint32_t s_abs = rho - o_k;
                const uint32_t s = s_abs % TILE_ROWS;
                const uint32_t t = rho % TILE_ROWS;
                uint32_t m = b - rho;
                const uint32_t ms = FACE_ROWS - (s % FACE_ROWS);
                const uint32_t mt = FACE_ROWS - (t % FACE_ROWS);
                if (m > ms) {
                    m = ms;
                }
                if (m > mt) {
                    m = mt;
                }
                const uint32_t src_page = (p * in_h_tiles + s_abs / TILE_ROWS) * Wt + wt;
                const uint32_t len = m * face_row_bytes;
                for (uint32_t c = 0; c < 2; ++c) {
                    const uint32_t src_off = ((s / FACE_ROWS) * 2 + c) * face_bytes + (s % FACE_ROWS) * face_row_bytes;
                    const uint32_t dst_off = ((t / FACE_ROWS) * 2 + c) * face_bytes + (t % FACE_ROWS) * face_row_bytes;
                    f(k, src_page, src_off, dst_off, len);
                }
                rho += m;
            }
        }
        o_k = end_k;
    }
}

FORCE_INLINE uint32_t align_down(uint32_t x) { return x & ~(dram_align - 1); }

FORCE_INLINE void run_row_gather(uint32_t core_start_page, uint32_t core_pages) {
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    constexpr uint32_t pad_row0 = total_rows % TILE_ROWS;  // first padding row of the last tile row

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    uint32_t remaining = core_pages;
    uint32_t g0 = core_start_page;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t extent = remaining < block_pages ? remaining : block_pages;
        remaining -= extent;
        {
            DeviceZoneScopedN("reader_reserve");
            cb_reserve_back(cb_out_tiles, block_pages);
        }
        const uint32_t out_base = get_write_ptr(cb_out_tiles);
        const uint32_t stage_base = get_write_ptr(cb_row_stage);

        // ---- stage_rows_block: aligned DRAM windows -> cb_row_stage ----
        {
            DeviceZoneScopedN("stage_rows_block");
            uint32_t p = g0 / pages_per_slice;
            uint32_t rem = g0 - p * pages_per_slice;
            uint32_t ht = rem / Wt;
            uint32_t wt = rem - ht * Wt;
            for (uint32_t i = 0; i < extent; ++i) {
                uint32_t stage_cursor = stage_base + i * stage_tile_bytes;
                for_each_piece(p, ht, wt, [&](uint32_t k, uint32_t src_page, uint32_t src_off, uint32_t, uint32_t len) {
                    const uint32_t win_start = align_down(src_off);
                    const uint32_t win_len = align_power_of_2(src_off + len, dram_align) - win_start;
                    const auto acc = input_accessor(k);
                    noc_async_read(acc.get_noc_addr(src_page, win_start), stage_cursor, win_len);
                    stage_cursor += win_len;
                });
                if (++wt == Wt) {
                    wt = 0;
                    if (++ht == out_h_tiles) {
                        ht = 0;
                        ++p;
                    }
                }
            }
        }
        {
            DeviceZoneScopedN("stage_barrier");
            noc_async_read_barrier();
        }

        // ---- place_rows_block: L1->L1 stage -> slots; zero rows >= N ----
        {
            DeviceZoneScopedN("place_rows_block");
            uint32_t p = g0 / pages_per_slice;
            uint32_t rem = g0 - p * pages_per_slice;
            uint32_t ht = rem / Wt;
            uint32_t wt = rem - ht * Wt;
            for (uint32_t i = 0; i < extent; ++i) {
                uint32_t stage_cursor = stage_base + i * stage_tile_bytes;
                const uint32_t slot = out_base + i * tile_bytes;
                for_each_piece(p, ht, wt, [&](uint32_t, uint32_t, uint32_t src_off, uint32_t dst_off, uint32_t len) {
                    const uint32_t win_start = align_down(src_off);
                    const uint32_t win_len = align_power_of_2(src_off + len, dram_align) - win_start;
                    noc_async_read(get_noc_addr(stage_cursor + (src_off - win_start)), slot + dst_off, len);
                    stage_cursor += win_len;
                });
                if constexpr (pad_row0 != 0) {
                    if (ht == out_h_tiles - 1) {
                        // Rows [pad_row0, 32) of both column faces: <= 4 contiguous runs, NoC-read
                        // from the core's zeros region (MEM_ZEROS_SIZE bytes per read).
                        for (uint32_t h = pad_row0 / FACE_ROWS; h < 2; ++h) {
                            const uint32_t first = (h * FACE_ROWS > pad_row0) ? h * FACE_ROWS : pad_row0;
                            const uint32_t run_bytes = (h * FACE_ROWS + FACE_ROWS - first) * face_row_bytes;
                            for (uint32_t c = 0; c < 2; ++c) {
                                uint32_t dst = slot + (h * 2 + c) * face_bytes + (first % FACE_ROWS) * face_row_bytes;
                                uint32_t left = run_bytes;
                                while (left > 0) {
                                    const uint32_t chunk = left < MEM_ZEROS_SIZE ? left : MEM_ZEROS_SIZE;
                                    noc_async_read(zeros_noc_addr, dst, chunk);
                                    dst += chunk;
                                    left -= chunk;
                                }
                            }
                        }
                    }
                }
                if (++wt == Wt) {
                    wt = 0;
                    if (++ht == out_h_tiles) {
                        ht = 0;
                        ++p;
                    }
                }
            }
        }
        {
            DeviceZoneScopedN("place_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_out_tiles, block_pages);
        g0 += extent;
    }
}

void kernel_main() {
    const uint32_t core_start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    if constexpr (is_row_gather) {
        run_row_gather(core_start_page, core_pages);
    } else {
        run_page_relocation(core_start_page, core_pages);
    }
}
