// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_core_data_exchange_common.hpp"

#include "experimental/kernel_args.h"

#include <cstdint>
#include <utility>

void kernel_main() {
    Noc noc;

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr uint32_t compute_with_storage_grid_size_y = get_arg(args::compute_with_storage_grid_size_y);

    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t number_of_tiles_per_core = get_arg(args::number_of_tiles_per_core);
    constexpr uint32_t number_of_cores_used = get_arg(args::number_of_cores_used);
    constexpr bool ascending = get_arg(args::ascending) == 1;

    constexpr uint32_t W_value_slice_bytes = get_arg(args::W_value_slice_bytes);
    constexpr uint32_t W_index_slice_bytes = get_arg(args::W_index_slice_bytes);

    // TILE-format tile sizes for the input and index tensors. They size both the DRAM transfers on
    // the TILE path and the peer-to-peer tile exchange, which runs in both configurations. In
    // ROW_MAJOR this kernel binds no dataflow buffer paged at those sizes, so they arrive as
    // compile-time arguments rather than buffer metadata.
    constexpr uint32_t input_tensor_tile_size_bytes = get_arg(args::input_tensor_tile_size_bytes);
    constexpr uint32_t index_tensor_tile_size_bytes = get_arg(args::index_tensor_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;
    constexpr uint32_t start_core_id = 0;
    constexpr uint32_t leader_core_id = start_core_id;

    // Input tensor config
    const auto input_tensor_accessor = TensorAccessor(tensor::input_tensor);
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_input_dfb(dfb::rm_input);
#else
    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
#endif

    // Index tensor config
    const auto index_tensor_output_accessor = TensorAccessor(tensor::index_tensor);
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_index_output_dfb(dfb::rm_index_output);
#else
    DataflowBuffer index_output_dfb(dfb::index_tensor_output);
#endif

    // Physical core lookup table config
    const auto physical_core_lookup_table_accessor = TensorAccessor(tensor::physical_core_lookup_table);
    DataflowBuffer physical_core_lookup_table_dfb(dfb::physical_core_lookup_table);
    const uint32_t physical_core_lookup_table_tile_size_bytes = physical_core_lookup_table_dfb.get_tile_size();

    // Read lookup table for physical core IDs
    physical_core_lookup_table_dfb.reserve_back(one_tile);
    noc.async_read(
        physical_core_lookup_table_accessor,
        physical_core_lookup_table_dfb,
        physical_core_lookup_table_tile_size_bytes,
        {.page_id = 0, .offset_bytes = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    // Semaphore setup
    Semaphore sem_exchange(sem::exchange);
    Semaphore sem_barrier(sem::barrier);

    // ROW_MAJOR per-core slice byte offset within each input/index DRAM row.
    // Each core owns a contiguous strip of `number_of_tiles_per_core` tiles
    // (= W_value_slice_bytes for values, W_index_slice_bytes for indices).
    constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT
    const uint32_t value_slice_offset_bytes = core_id * W_value_slice_bytes;
    const uint32_t index_slice_offset_bytes = core_id * W_index_slice_bytes;

    for (uint32_t h = 0; h < Ht; h++) {
        // Read input value data
#ifdef IS_ROW_MAJOR
        {
            // ROW_MAJOR input: read TILE_H rows of the per-core W-slice from DRAM
            // into rm_input_dfb. Compute kernel will tilize them into
            // the tile-format input buffer for the existing TILE-format sort flow.
            const uint32_t row_base = h * TILE_H;
            for (uint32_t row = 0; row < TILE_H; row++) {
                rm_input_dfb.reserve_back(one_tile);
                noc.async_read(
                    input_tensor_accessor,
                    rm_input_dfb,
                    W_value_slice_bytes,
                    {.page_id = row_base + row, .offset_bytes = value_slice_offset_bytes},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                rm_input_dfb.push_back(one_tile);
            }
        }
#else
        {
            // TILE input path
            for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
                input_tensor_dfb.reserve_back(one_tile);
                const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
                noc.async_read(
                    input_tensor_accessor,
                    input_tensor_dfb,
                    input_tensor_tile_size_bytes,
                    {.page_id = tile_offset, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                input_tensor_dfb.push_back(one_tile);
            }  // w loop
        }
#endif

        const uint32_t stages = ilog2(Wt);
        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                const uint32_t sub_dist = 1 << (sub - 1);

                const uint32_t i = global_tile_start;
                const uint32_t j = i ^ sub_dist;

                if (!(i >= global_tile_start && i < global_tile_end && j >= global_tile_start && j < global_tile_end)) {
                    // Without this barrier, a faster core (in this scenario core C) could start a new exchange
                    // before its peer has finished the previous one, causing a conflict
                    // on the shared semaphore. For example, with three cores A, B, and C:
                    //  A     B     C
                    //  |     |     |
                    //  E <-> E     |   (A and B exchanging tiles)
                    //  E <-> E     |
                    //  E <-> E     |
                    //  E <---E-----|   (C starts exchange with A)
                    //  X     E     |   (A is now in an invalid state)
                    //  X     E     |
                    //
                    // This barrier ensures all cores reach the same stage before proceeding,
                    // preventing such conflicts.
                    sort_barrier(
                        noc,
                        sem_barrier,
                        dfb::physical_core_lookup_table,
                        core_id,
                        leader_core_id,
                        number_of_cores_used,
                        start_core_id);

                    const uint32_t other_core_id = j / number_of_tiles_per_core;
                    const std::pair<uint32_t, uint32_t> remote_core_physical =
                        get_core_physical_coordinates(other_core_id, dfb::physical_core_lookup_table);

                    sort_noc_exchange_Wt_tiles(
                        noc,
                        sem_exchange,
                        dfb::value_tensor_intermediate,
                        dfb::index_tensor_intermediate,
                        dfb::value_tensor_peer,
                        dfb::index_tensor_peer,
                        number_of_tiles_per_core,
                        input_tensor_tile_size_bytes,
                        index_tensor_tile_size_bytes,
                        remote_core_physical.first,
                        remote_core_physical.second);
                }  // if !(i >= global_tile_start && i < ...
            }  // sub
        }  // stages

        // Write output index data
#ifdef IS_ROW_MAJOR
        {
            // ROW_MAJOR output indices: drain TILE_H untilized index rows
            // from rm_index_output_dfb (compute pack_untilize'd them) and write
            // each row's per-core W-slice back to DRAM. pack_untilize_block
            // produces little-endian uint16/uint32 elements, so no byte swap.
            const uint32_t row_base = h * TILE_H;
            for (uint32_t row = 0; row < TILE_H; row++) {
                rm_index_output_dfb.wait_front(one_tile);
                noc.async_write(
                    rm_index_output_dfb,
                    index_tensor_output_accessor,
                    W_index_slice_bytes,
                    {.offset_bytes = 0},
                    {.page_id = row_base + row, .offset_bytes = index_slice_offset_bytes});
                noc.async_write_barrier();
                rm_index_output_dfb.pop_front(one_tile);
            }
        }
#else
        {
            // Write output index data (TILE path)
            for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
                index_output_dfb.wait_front(one_tile);
                const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
                noc.async_write(
                    index_output_dfb,
                    index_tensor_output_accessor,
                    index_tensor_tile_size_bytes,
                    {.offset_bytes = 0},
                    {.page_id = tile_offset, .offset_bytes = 0});
                noc.async_write_barrier();
                index_output_dfb.pop_front(one_tile);
            }  // Wt loop
        }
#endif

    }  // h loop
    physical_core_lookup_table_dfb.push_back(one_tile);
}
