// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul reader: streams the A and B panels for this cluster's C blocks.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N] = A[M x K] x B[K x N]. The unit of work is one C block
// (per_core_M x per_core_N tiles, numbered row-major over the C block grid) of one batch; work item w is
// batch w / num_C_blocks, C block w % num_C_blocks, and this cluster owns a contiguous run of items. For
// each work item and each K step the reader pushes
//   - one A panel: this block's rows of A, K_step_tiles wide    -> [per_core_M][K_step_tiles] tiles, row-major
//   - one B panel: this block's columns of B, K_step_tiles tall -> [K_step_tiles][per_core_N] tiles, row-major
// which is the layout the compute kernel indexes. Loop order (work item, K step) matches it.
//
// Edge blocks: tiles past M_tiles / N_tiles are never read; their slots keep stale L1, which only reaches
// C tiles the writer drops (a valid C tile uses valid A rows and valid B columns only). When K is not a
// multiple of the tile dim, the padding columns of A's last K tile are zeroed so they contribute nothing.
//
// A and B are addressed by tile index through the tensor accessor (page id == row-major tile index within
// the tensor), so interleaved, L1-sharded and DRAM-sharded inputs are one code path.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    // Per-core runtime args: this cluster's contiguous run of work items.
    const uint32_t first_work_item = get_arg(args::first_work_item);
    const uint32_t num_work_items = get_arg(args::num_work_items);

    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t K_tiles = get_arg(args::K_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);
    constexpr uint32_t num_C_blocks = get_arg(args::num_C_blocks);  // per batch
    constexpr bool broadcast_B_over_batch = get_arg(args::broadcast_B_over_batch) != 0;
    constexpr uint32_t per_core_M = get_arg(args::per_core_M);
    constexpr uint32_t per_core_N = get_arg(args::per_core_N);
    constexpr uint32_t K_step_tiles = get_arg(args::K_step_tiles);
    constexpr uint32_t num_K_steps = get_arg(args::num_K_steps);
    constexpr uint32_t num_C_block_columns = get_arg(args::num_C_block_columns);
    // Valid element columns in A's last K tile; 0 when K is a multiple of the tile dim.
    constexpr uint32_t A_last_K_tile_valid_columns = get_arg(args::A_last_K_tile_valid_columns);

    constexpr uint32_t A_panel_tiles = per_core_M * K_step_tiles;
    constexpr uint32_t B_panel_tiles = K_step_tiles * per_core_N;
    constexpr uint32_t A_tiles_per_batch = M_tiles * K_tiles;
    constexpr uint32_t B_tiles_per_batch = K_tiles * N_tiles;

    const auto A = TensorAccessor(tensor::A);
    const auto B = TensorAccessor(tensor::B);
    Noc noc;
    DataflowBuffer A_panel(dfb::A_panel);
    DataflowBuffer B_panel(dfb::B_panel);

    const uint32_t A_tile_bytes = get_tile_size(dfb::A_panel);
    const uint32_t B_tile_bytes = get_tile_size(dfb::B_panel);
    // A ring slot is one tile at the DRAM-aligned stride the factory sized the ring with (== tile bytes
    // for every 32x32 format).
    const uint32_t A_slot_bytes = A_panel.get_entry_size();
    const uint32_t B_slot_bytes = B_panel.get_entry_size();

    for (uint32_t work_item = first_work_item; work_item < first_work_item + num_work_items; ++work_item) {
        const uint32_t batch = work_item / num_C_blocks;
        const uint32_t C_block = work_item % num_C_blocks;
        const uint32_t A_batch_first_tile = batch * A_tiles_per_batch;
        const uint32_t B_batch_first_tile = broadcast_B_over_batch ? 0 : batch * B_tiles_per_batch;
        const uint32_t first_M_tile = (C_block / num_C_block_columns) * per_core_M;
        const uint32_t first_N_tile = (C_block % num_C_block_columns) * per_core_N;
        {
            // Rows / columns of this block that lie inside the matrices (edge blocks are clipped).
            const uint32_t valid_M_tiles =
                (M_tiles - first_M_tile < per_core_M) ? (M_tiles - first_M_tile) : per_core_M;
            const uint32_t valid_N_tiles =
                (N_tiles - first_N_tile < per_core_N) ? (N_tiles - first_N_tile) : per_core_N;

            for (uint32_t K_step = 0; K_step < num_K_steps; ++K_step) {
                const uint32_t first_K_tile = K_step * K_step_tiles;

                // A panel: rows first_M_tile.., columns first_K_tile.. (invalid rows trail, so they are
                // simply not written).
                A_panel.reserve_back(A_panel_tiles);
                {
                    uint32_t slot_offset = 0;
                    for (uint32_t m = 0; m < valid_M_tiles; ++m) {
                        uint32_t A_tile_index = A_batch_first_tile + (first_M_tile + m) * K_tiles + first_K_tile;
                        for (uint32_t k = 0; k < K_step_tiles; ++k, ++A_tile_index, slot_offset += A_slot_bytes) {
                            noc.async_read(
                                A, A_panel, A_tile_bytes, {.page_id = A_tile_index}, {.offset_bytes = slot_offset});
                        }
                    }
                }

                // B panel: rows first_K_tile.., columns first_N_tile.. (invalid columns keep their slot).
                B_panel.reserve_back(B_panel_tiles);
                {
                    uint32_t slot_offset = 0;
                    for (uint32_t k = 0; k < K_step_tiles; ++k) {
                        uint32_t B_tile_index = B_batch_first_tile + (first_K_tile + k) * N_tiles + first_N_tile;
                        for (uint32_t n = 0; n < per_core_N; ++n, ++B_tile_index, slot_offset += B_slot_bytes) {
                            if (n < valid_N_tiles) {
                                noc.async_read(
                                    B, B_panel, B_tile_bytes, {.page_id = B_tile_index}, {.offset_bytes = slot_offset});
                            }
                        }
                    }
                }

                noc.async_read_barrier();

                if constexpr (A_last_K_tile_valid_columns > 0) {
                    // Zero the padding columns of the last K tile in every valid row (reads have landed).
                    if (K_step == num_K_steps - 1) {
                        constexpr DataFormat A_format = get_dataformat(dfb::A_panel);
                        const uint32_t last_K_tile_of_row_0 =
                            A_panel.get_write_ptr() + (K_step_tiles - 1) * A_slot_bytes;
                        for (uint32_t m = 0; m < valid_M_tiles; ++m) {
                            pad_last_ktile<A_format, A_last_K_tile_valid_columns>(
                                last_K_tile_of_row_0 + m * K_step_tiles * A_slot_bytes);
                        }
                    }
                }

                A_panel.push_back(A_panel_tiles);
                B_panel.push_back(B_panel_tiles);
            }
        }
    }
}
