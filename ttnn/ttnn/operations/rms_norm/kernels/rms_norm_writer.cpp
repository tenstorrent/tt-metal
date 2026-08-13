// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (NoC1).  Realizes `store_block` and the *contributor half* of
// `combine_block` (the gather: this core's per-slice partial -> the row-group
// root's cb_gathered_partials page (row * s + slice_index), plus one progress
// increment).
//
// Raw-API notes: the gather is a scatter of s DIFFERENT sources into s DIFFERENT
// destination pages on one core — the opposite shape from mcast_pipe's
// one-source-to-a-rectangle broadcast, and kernel_lib has no gather helper.  The
// destination address is derived from THIS core's own cb_gathered_partials write
// pointer, which is valid because every CB in this program is declared on one
// common core set, so the L1 map is identical on every participating core.
//
// The root is NOT special-cased: it writes its own partial through the same NoC
// path (a local loopback write) and increments the same counter, so the root
// waits for exactly s arrivals and the code stays uniform.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_output_tiles = 9;
constexpr uint32_t cb_rm_stage_out = 11;

constexpr uint32_t TILE_DIM = 32;

void kernel_main() {
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);   // s
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_TILE_BYTES = get_compile_time_arg_val(4);
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(5);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(6);
    constexpr uint32_t TENSOR_HIDDEN_TILES = get_compile_time_arg_val(7);  // page stride only
    constexpr uint32_t OUT_ELEM_BYTES = get_compile_time_arg_val(8);
    constexpr uint32_t DM_CHUNK_TILES = get_compile_time_arg_val(9);
    constexpr auto out_args = TensorAccessorArgs<10>();

    constexpr uint32_t RM_STICK_PITCH = SLICE_HIDDEN_TILES * TILE_DIM * OUT_ELEM_BYTES;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t slice_base = get_arg_val<uint32_t>(4);
    const uint32_t valid_tiles = get_arg_val<uint32_t>(5);
    const uint32_t valid_w = get_arg_val<uint32_t>(6);
    const uint32_t root_noc_x = get_arg_val<uint32_t>(7);
    const uint32_t root_noc_y = get_arg_val<uint32_t>(8);
    const uint32_t slice_index = get_arg_val<uint32_t>(9);
    const uint32_t total_sticks = get_arg_val<uint32_t>(10);

    Noc noc;
    const auto output_accessor = TensorAccessor(out_args, output_addr);
    Semaphore<> gather_progress(GATHER_SEM_ID);

    // Captured before any push/pop touches the CB, so this is its base address —
    // identical on every core in the row-group rect.
    uint32_t gather_base = 0;
    if constexpr (NUM_HIDDEN_SLICES > 1) {
        gather_base = get_write_ptr(cb_gathered_partials);
    }

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_row = block * BLOCK_ROWS;

        // ---- combine_block: contribute this slice's partials to the root ----
        if constexpr (NUM_HIDDEN_SLICES > 1) {
            cb_wait_front(cb_sq_partials, BLOCK_ROWS);
            const uint32_t src = get_read_ptr(cb_sq_partials);
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                const uint32_t page = r * NUM_HIDDEN_SLICES + slice_index;
                noc_async_write(
                    src + r * STAT_TILE_BYTES,
                    get_noc_addr(root_noc_x, root_noc_y, gather_base + page * STAT_TILE_BYTES),
                    STAT_TILE_BYTES);
            }
            noc_async_write_barrier();
            gather_progress.up(noc, root_noc_x, root_noc_y, 1);
            cb_pop_front(cb_sq_partials, BLOCK_ROWS);
        }

        // ---- store_block ----
        if constexpr (IS_ROW_MAJOR) {
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                cb_wait_front(cb_rm_stage_out, SLICE_HIDDEN_TILES);
                const uint32_t l1 = get_read_ptr(cb_rm_stage_out);
                const uint32_t local_row = first_row + r;
                if (local_row < core_row_tiles) {
                    const uint32_t stick_base = (row_start + local_row) * TILE_DIM;
                    uint32_t pending = 0;
                    for (uint32_t k = 0; k < TILE_DIM; ++k) {
                        const uint32_t stick = stick_base + k;
                        if (stick >= total_sticks) {
                            break;
                        }
                        noc_async_write(
                            l1 + k * RM_STICK_PITCH,
                            output_accessor.get_noc_addr(stick, slice_base * TILE_DIM * OUT_ELEM_BYTES),
                            valid_w * OUT_ELEM_BYTES);
                        if (++pending == DM_CHUNK_TILES) {
                            noc_async_write_barrier();
                            pending = 0;
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_rm_stage_out, SLICE_HIDDEN_TILES);
            }
        } else {
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                cb_wait_front(cb_output_tiles, SLICE_HIDDEN_TILES);
                const uint32_t l1 = get_read_ptr(cb_output_tiles);
                const uint32_t local_row = first_row + r;
                if (local_row < core_row_tiles) {
                    const uint32_t page = (row_start + local_row) * TENSOR_HIDDEN_TILES + slice_base;
                    uint32_t pending = 0;
                    for (uint32_t j = 0; j < valid_tiles; ++j) {
                        noc_async_write(
                            l1 + j * OUT_TILE_BYTES, output_accessor.get_noc_addr(page + j), OUT_TILE_BYTES);
                        if (++pending == DM_CHUNK_TILES) {
                            noc_async_write_barrier();
                            pending = 0;
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, SLICE_HIDDEN_TILES);
            }
        }
    }
}
