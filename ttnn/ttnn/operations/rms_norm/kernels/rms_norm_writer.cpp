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

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_output_tiles = 9;
constexpr uint32_t cb_rm_stage_out = 11;
constexpr uint32_t cb_shard_out = 14;  // ROW_MAJOR + sharded: the resident output shard

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
    // Sharded: the output shard is already this core's L1, so `store_block` is
    // either a no-op (TILE — cb_output_tiles IS the shard) or a core-local L1
    // stick write (ROW_MAJOR — untilize's staging drains into the shard).
    constexpr uint32_t IS_SHARDED = get_compile_time_arg_val(10);
    constexpr uint32_t SHARD_PAGE_BYTES = get_compile_time_arg_val(11);
    constexpr auto out_args = TensorAccessorArgs<12>();

    constexpr uint32_t RM_STICK_PITCH = SLICE_HIDDEN_TILES * TILE_DIM * OUT_ELEM_BYTES;
    static_assert(RM_STICK_PITCH % 16 == 0, "ROW_MAJOR staging stick pitch must be L1-aligned");

    // Same byte-budget conversion as the reader: DM_CHUNK_TILES tiles' worth of
    // bytes expressed in sticks (a stick is S*32 elements = S/32 of a tile),
    // clamped to the 32 sticks of one tile-row.  Keeps both halves of the
    // dataflow batching at the same granularity on both layouts.
    constexpr uint32_t RM_CHUNK_STICKS_RAW = (DM_CHUNK_TILES * TILE_DIM) / SLICE_HIDDEN_TILES;
    constexpr uint32_t RM_CHUNK_STICKS =
        RM_CHUNK_STICKS_RAW < 1 ? 1 : (RM_CHUNK_STICKS_RAW > TILE_DIM ? TILE_DIM : RM_CHUNK_STICKS_RAW);

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
    const uint32_t slice_elem_base = get_arg_val<uint32_t>(11);

    Noc noc;
    const auto output_accessor = TensorAccessor(out_args, output_addr);
    Semaphore<> gather_progress(GATHER_SEM_ID);
    const uint64_t self_noc = get_noc_addr(my_x[noc_index], my_y[noc_index], 0);

    uint32_t shard_out_base = 0;
    if constexpr (IS_SHARDED && IS_ROW_MAJOR) {
        shard_out_base = get_write_ptr(cb_shard_out);
    }

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
            {
                // Waiting on THIS core's own Sum(x^2).  High here == the local
                // compute feeding the combine is the critical path.
                MaybeDeviceZoneScope("wr_stat_wait");
                cb_wait_front(cb_sq_partials, BLOCK_ROWS);
            }
            const uint32_t src = get_read_ptr(cb_sq_partials);
            {
                // ISSUE: B stat-tile writes into the root's landing pages.  This
                // is RISC-serial address generation + command-buffer work and
                // scales with the TRANSACTION COUNT (B per contributor).
                MaybeDeviceZoneScope("wr_gather_issue");
                for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                    const uint32_t page = r * NUM_HIDDEN_SLICES + slice_index;
                    noc_async_write(
                        src + r * STAT_TILE_BYTES,
                        get_noc_addr(root_noc_x, root_noc_y, gather_base + page * STAT_TILE_BYTES),
                        STAT_TILE_BYTES);
                }
            }
            {
                // BARRIER: the bytes actually landing on the root.  This is the
                // half that carries the incast congestion (s contributors x B
                // tiles converging on one core's L1 write port).
                MaybeDeviceZoneScope("wr_gather_barrier");
                noc_async_write_barrier();
            }
            gather_progress.up(noc, root_noc_x, root_noc_y, 1);
            cb_pop_front(cb_sq_partials, BLOCK_ROWS);
        }

        // ---- store_block ----
        // Outer zone = the whole store region's occupancy; the inner
        // `wr_store_wait` / `wr_store_barrier` pair separates "starved by
        // compute" from "waiting on the NoC", leaving issue cost as the
        // remainder.  On a resident output shard the store moves no bytes at all
        // and the whole region should read as pure wait.
        {
            MaybeDeviceZoneScope("wr_store_total");
            if constexpr (IS_ROW_MAJOR) {
                for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                    {
                        MaybeDeviceZoneScope("wr_store_wait");
                        cb_wait_front(cb_rm_stage_out, SLICE_HIDDEN_TILES);
                    }
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
                            const uint64_t dst =
                                IS_SHARDED ? (self_noc + shard_out_base + stick * SHARD_PAGE_BYTES)
                                           : output_accessor.get_noc_addr(stick, slice_elem_base * OUT_ELEM_BYTES);
                            noc_async_write(l1 + k * RM_STICK_PITCH, dst, valid_w * OUT_ELEM_BYTES);
                            if (++pending == RM_CHUNK_STICKS) {
                                noc_async_write_barrier();
                                pending = 0;
                            }
                        }
                    }
                    {
                        MaybeDeviceZoneScope("wr_store_barrier");
                        noc_async_write_barrier();
                    }
                    cb_pop_front(cb_rm_stage_out, SLICE_HIDDEN_TILES);
                }
            } else {
                for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                    {
                        MaybeDeviceZoneScope("wr_store_wait");
                        cb_wait_front(cb_output_tiles, SLICE_HIDDEN_TILES);
                    }
                    const uint32_t l1 = get_read_ptr(cb_output_tiles);
                    const uint32_t local_row = first_row + r;
                    // Sharded: cb_output_tiles IS the caller's resident output shard,
                    // so compute already packed this tile-row into its final home.
                    // The pop is the whole store — it just advances the CB window
                    // through the shard, moving no bytes.
                    if (IS_SHARDED) {
                        cb_pop_front(cb_output_tiles, SLICE_HIDDEN_TILES);
                        continue;
                    }
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
                    {
                        MaybeDeviceZoneScope("wr_store_barrier");
                        noc_async_write_barrier();
                    }
                    cb_pop_front(cb_output_tiles, SLICE_HIDDEN_TILES);
                }
            }
        }  // wr_store_total
    }
}
