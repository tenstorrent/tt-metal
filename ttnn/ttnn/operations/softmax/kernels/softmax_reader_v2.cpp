// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax reader kernel — V2 streaming path.
//
// Reads input tiles in chunks of BLOCK_SIZE along the reduce dimension.
// Makes three DRAM passes per tile-row (dim=-1) or tile-column (dim=-2):
//   Pass 0: reads tiles for max reduction
//   Pass 1: re-reads tiles for sum(exp(x-max)) reduction
//   Pass 2: re-reads tiles for final apply (exp(x-max) * recip_sum)
//
// TILE path: reads tiles directly into cb_input_tiles
// RM path:   reads sticks into cb_rm_in (compute tilizes)
//
// At kernel start: prepares scaler tiles (cb_scaler_max, cb_scaler_sum).
//
// read_sticks_for_tilize<cb_rm_in> pushes `width_in_tiles` pages per block
// (one block = 32 sticks). The compute kernel's tilize helper consumes
// the same count.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_scaler_max = 1;
constexpr uint32_t cb_scaler_sum = 2;
constexpr uint32_t cb_rm_in = 3;
}  // namespace

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_slabs = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr int32_t dim = static_cast<int32_t>(get_compile_time_arg_val(2));
    constexpr uint32_t is_rm = get_compile_time_arg_val(3);
    constexpr uint32_t origin_W = get_compile_time_arg_val(4);
    constexpr uint32_t origin_H = get_compile_time_arg_val(5);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(6);
    constexpr uint32_t chunk_along_reduce = get_compile_time_arg_val(7);

    constexpr auto src_args = TensorAccessorArgs<8>();
    const auto src_accessor = TensorAccessor(src_args, input_buffer_address);

    // Prepare scaler tiles once at kernel start.
    constexpr uint32_t partial_W = origin_W % 32;
    constexpr uint32_t partial_H = origin_H % 32;
    constexpr bool has_partial = (dim == -1) ? (partial_W > 0) : (partial_H > 0);

    if constexpr (dim == -1) {
        if constexpr (has_partial) {
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_scaler_max,
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                partial_W>(1.0f);
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_scaler_sum,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                partial_W>(1.0f);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        }
    } else {
        if constexpr (has_partial) {
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_scaler_max,
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_COL,
                partial_H>(1.0f);
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_scaler_sum,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                partial_H>(1.0f);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_COL>(1.0f);
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(1.0f);
        }
    }

    constexpr uint32_t reduce_dim_tiles = (dim == -1) ? Wt : Ht;
    constexpr uint32_t non_reduce_dim = (dim == -1) ? Ht : Wt;
    constexpr uint32_t num_chunks =
        chunk_along_reduce ? (reduce_dim_tiles / BLOCK_SIZE) : (non_reduce_dim / BLOCK_SIZE);
    constexpr uint32_t num_passes = chunk_along_reduce ? 3 : 1;

    if constexpr (!is_rm) {
        // ===== TILE path: read tiles directly into cb_input_tiles =====
        CircularBuffer input_cb(cb_input_tiles);
        Noc noc;
        const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
        uint32_t slab_start_tile = start_id;

        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            if constexpr (chunk_along_reduce) {
                // 3-pass approach: chunk along reduce dim
                if constexpr (dim == -1) {
                    for (uint32_t ht = 0; ht < Ht; ++ht) {
                        uint32_t row_base = slab_start_tile + ht * Wt;
                        for (uint32_t pass = 0; pass < num_passes; ++pass) {
                            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                                uint32_t chunk_base = row_base + chunk * BLOCK_SIZE;
                                for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                                    input_cb.reserve_back(1);
                                    noc.async_read(
                                        src_accessor,
                                        input_cb,
                                        tile_bytes,
                                        {.page_id = chunk_base + i},
                                        {.offset_bytes = 0});
                                    noc.async_read_barrier();
                                    input_cb.push_back(1);
                                }
                            }
                        }
                    }
                } else {
                    for (uint32_t wt = 0; wt < Wt; ++wt) {
                        for (uint32_t pass = 0; pass < num_passes; ++pass) {
                            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                                for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                                    uint32_t ht = chunk * BLOCK_SIZE + i;
                                    uint32_t tile_id = slab_start_tile + ht * Wt + wt;
                                    input_cb.reserve_back(1);
                                    noc.async_read(
                                        src_accessor, input_cb, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
                                    noc.async_read_barrier();
                                    input_cb.push_back(1);
                                }
                            }
                        }
                    }
                }
            } else {
                // chunk_along_non_reduce: 1 pass, chunks along non-reduce dim
                if constexpr (dim == -1) {
                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                            uint32_t ht = chunk * BLOCK_SIZE + i;
                            uint32_t row_base = slab_start_tile + ht * Wt;
                            for (uint32_t wt = 0; wt < Wt; ++wt) {
                                input_cb.reserve_back(1);
                                noc.async_read(
                                    src_accessor,
                                    input_cb,
                                    tile_bytes,
                                    {.page_id = row_base + wt},
                                    {.offset_bytes = 0});
                                noc.async_read_barrier();
                                input_cb.push_back(1);
                            }
                        }
                    }
                } else {
                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        for (uint32_t ht = 0; ht < Ht; ++ht) {
                            for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
                                uint32_t wt = chunk * BLOCK_SIZE + i;
                                uint32_t tile_id = slab_start_tile + ht * Wt + wt;
                                input_cb.reserve_back(1);
                                noc.async_read(
                                    src_accessor, input_cb, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
                                noc.async_read_barrier();
                                input_cb.push_back(1);
                            }
                        }
                    }
                }
            }
            slab_start_tile += Ht * Wt;
        }
    } else {
        // ===== ROW_MAJOR path: read sticks into cb_rm_in =====
        //
        // read_sticks_for_tilize<cb_rm_in> pushes `width_in_tiles` pages per
        // block (32 rows). Each push is consumed by the compute kernel's
        // tilize helper, which reads `width_in_tiles` pages and produces
        // `width_in_tiles` tile-pages.
        //
        // For chunk_along_reduce (dim=-1):
        //   Each chunk reads 32 sticks (1 tile-row), BLOCK_SIZE tiles wide.
        //   Using byte_offset_within_page to select the W-slice for each chunk.
        //   3 passes per tile-row (max → sum → apply).
        //
        // For chunk_along_reduce (dim=-2):
        //   Each chunk reads 32*BLOCK_SIZE sticks, 1 tile column wide.
        //   Using byte_offset_within_page to select the 1-tile-wide column.
        //   3 passes per tile-column (max → sum → apply).
        //
        // For chunk_along_non_reduce (dim=-1):
        //   Each chunk reads BLOCK_SIZE tile-rows (BLOCK_SIZE*32 sticks),
        //   full W width. 1 pass per chunk (V1-style 4-phase per chunk).
        //
        // For chunk_along_non_reduce (dim=-2):
        //   Each chunk reads full H, BLOCK_SIZE tile-columns wide.
        //   1 pass per chunk.
        constexpr uint32_t tile_h = 32;
        const uint32_t tile_size = get_tile_size(cb_rm_in);
        // Full row bytes = origin_W * elem_size = origin_W * tile_size / (32*32)
        const uint32_t full_row_bytes = origin_W * tile_size / (tile_h * tile_h);

        uint32_t slab_start_stick = start_id;

        for (uint32_t slab = 0; slab < num_slabs; ++slab) {
            if constexpr (chunk_along_reduce) {
                if constexpr (dim == -1) {
                    // Chunk width in bytes: BLOCK_SIZE tiles = BLOCK_SIZE * 32 elements
                    constexpr uint32_t chunk_row_bytes = BLOCK_SIZE * tile_size / tile_h;

                    for (uint32_t ht = 0; ht < Ht; ++ht) {
                        // Each tile-row starts at stick: slab_start + ht * 32
                        uint32_t base_stick = slab_start_stick + ht * tile_h;

                        for (uint32_t pass = 0; pass < 3; ++pass) {
                            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                                uint32_t byte_offset = chunk * chunk_row_bytes;
                                // Clamp row_bytes to not exceed the page boundary.
                                // For non-aligned W, the last chunk may extend past
                                // the actual page — read_sticks_for_tilize pads the
                                // remainder in L1 (padded_row_bytes > row_bytes).
                                uint32_t actual_row_bytes =
                                    (byte_offset + chunk_row_bytes <= full_row_bytes)
                                        ? chunk_row_bytes
                                        : (full_row_bytes > byte_offset ? (full_row_bytes - byte_offset) : 0);
                                dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in>(
                                    src_accessor,
                                    tile_h,            // total_num_rows (one tile-height of sticks)
                                    actual_row_bytes,  // row_bytes for this chunk
                                    base_stick,        // start_page (stick index)
                                    byte_offset        // byte_offset_within_page
                                );
                            }
                        }
                    }
                } else {
                    // dim=-2: each chunk reads BLOCK_SIZE tile-rows, 1 tile column wide
                    constexpr uint32_t chunk_row_bytes = tile_size / tile_h;  // 1 tile column

                    for (uint32_t wt = 0; wt < Wt; ++wt) {
                        uint32_t byte_offset = wt * chunk_row_bytes;
                        // Clamp row_bytes for non-aligned W (last column may
                        // extend past the actual page boundary)
                        uint32_t actual_row_bytes =
                            (byte_offset + chunk_row_bytes <= full_row_bytes)
                                ? chunk_row_bytes
                                : (full_row_bytes > byte_offset ? (full_row_bytes - byte_offset) : 0);

                        for (uint32_t pass = 0; pass < 3; ++pass) {
                            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                                uint32_t base_stick = slab_start_stick + chunk * tile_h * BLOCK_SIZE;
                                dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in>(
                                    src_accessor,
                                    tile_h * BLOCK_SIZE,  // total_num_rows
                                    actual_row_bytes,     // row_bytes (1 tile column)
                                    base_stick,           // start_page
                                    byte_offset           // byte_offset_within_page
                                );
                            }
                        }
                    }
                }
            } else {
                // chunk_along_non_reduce: 1 pass, full reduce dim per chunk
                if constexpr (dim == -1) {
                    // Each chunk: BLOCK_SIZE tile-rows × full W width
                    // Read BLOCK_SIZE*32 sticks, full row width
                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        uint32_t base_stick = slab_start_stick + chunk * tile_h * BLOCK_SIZE;
                        dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in>(
                            src_accessor,
                            tile_h * BLOCK_SIZE,  // total_num_rows (BLOCK_SIZE tile-rows)
                            full_row_bytes,       // row_bytes (full width)
                            base_stick,           // start_page
                            0                     // byte_offset_within_page
                        );
                    }
                } else {
                    // dim=-2: each chunk reads full H, BLOCK_SIZE tile-columns wide
                    constexpr uint32_t chunk_row_bytes = BLOCK_SIZE * tile_size / tile_h;

                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        uint32_t byte_offset = chunk * chunk_row_bytes;
                        dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in>(
                            src_accessor,
                            origin_H,          // total_num_rows (full H)
                            chunk_row_bytes,   // row_bytes (BLOCK_SIZE tile columns)
                            slab_start_stick,  // start_page
                            byte_offset        // byte_offset_within_page
                        );
                    }
                }
            }
            slab_start_stick += origin_H;
        }
    }
}
