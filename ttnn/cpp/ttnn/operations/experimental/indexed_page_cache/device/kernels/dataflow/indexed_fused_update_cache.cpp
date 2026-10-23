// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

FORCE_INLINE std::pair<uint32_t, uint32_t> divmod(uint32_t dividend, uint32_t divisor) {
    return {dividend / divisor, dividend % divisor};
}

void kernel_main() {
    constexpr auto num_heads = get_arg(args::num_heads);
    constexpr auto width_tiles = get_arg(args::width_tiles);
    constexpr auto source_height_tiles = get_arg(args::source_height_tiles);
    constexpr auto cache_page_rows = get_arg(args::cache_page_rows);
    constexpr auto total_cache_rows = get_arg(args::total_cache_rows);
    constexpr auto worker_count = get_arg(args::worker_count);
    constexpr auto bytes_per_element = get_arg(args::bytes_per_element);
    constexpr auto scratch_buffer_depth = get_arg(args::scratch_buffer_depth);

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_height = 16;
    constexpr uint32_t face_bytes = face_width * face_height * bytes_per_element;
    constexpr uint32_t face_line_bytes = face_width * bytes_per_element;
    constexpr uint32_t cache_height_tiles = cache_page_rows / tile_height;

    const auto source_rows = get_arg(args::source_rows);
    const auto worker_start = get_arg(args::worker_start);
    const auto worker_stride = get_arg(args::worker_stride);

    DataflowBuffer scratch_dfb(dfb::scratch);
    DataflowBuffer positions_dfb(dfb::positions);
    Noc noc;

    const auto cache1 = TensorAccessor(tensor::cache1);
    const auto cache2 = TensorAccessor(tensor::cache2);
    const auto input1 = TensorAccessor(tensor::input1);
    const auto input2 = TensorAccessor(tensor::input2);
    const auto positions = TensorAccessor(tensor::positions);

    // Indices stay on device so a captured trace can target different physical
    // cache rows on every replay. Negative entries are per-row no-ops.
    positions_dfb.reserve_back(1);
    noc.async_read(positions, positions_dfb, positions_dfb.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    positions_dfb.push_back(1);
    CoreLocalMem<volatile int32_t> physical_positions(positions_dfb.get_read_ptr());

    auto copy_rows = [&](const auto& source, const auto& cache, uint32_t head, uint32_t width_tile) {
        bool writes_in_flight[scratch_buffer_depth] = {};

        auto wait_for_slot_writes = [&](uint32_t slot) {
            if (writes_in_flight[slot]) {
                noc.async_write_barrier<NocOptions::TXN_ID>({.trid = slot + 1});
                writes_in_flight[slot] = false;
            }
        };

        auto issue_source_read = [&](uint32_t source_height_tile, uint32_t slot) {
            // A slot can be overwritten only after the cache writes sourced from it have completed.
            wait_for_slot_writes(slot);
            const uint32_t source_tile = (head * source_height_tiles + source_height_tile) * width_tiles + width_tile;
            scratch_dfb.reserve_back(1);
            noc.async_read<NocOptions::TXN_ID>(
                source,
                scratch_dfb,
                scratch_dfb.get_entry_size(),
                {.page_id = source_tile},
                {.offset_bytes = 0},
                {.trid = slot + 1});
        };

        if constexpr (source_height_tiles == 0) {
            return;
        }

        issue_source_read(/*source_height_tile=*/0, /*slot=*/0);
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1});
        scratch_dfb.push_back(1);

        for (uint32_t source_height_tile = 0; source_height_tile < source_height_tiles; ++source_height_tile) {
            const uint32_t current_slot = source_height_tile % scratch_buffer_depth;
            const bool has_next_tile = source_height_tile + 1 < source_height_tiles;
            const uint32_t next_slot = (current_slot + 1) % scratch_buffer_depth;
            if (has_next_tile) {
                issue_source_read(source_height_tile + 1, next_slot);
            }

            bool wrote_current_slot = false;
            for (uint32_t tile_row = 0; tile_row < tile_height; ++tile_row) {
                const uint32_t source_row = source_height_tile * tile_height + tile_row;
                if (source_row >= source_rows) {
                    break;
                }
                const int32_t physical_row_signed = physical_positions[source_row];
                if (physical_row_signed < 0 || static_cast<uint32_t>(physical_row_signed) >= total_cache_rows) {
                    continue;
                }

                const uint32_t physical_row = static_cast<uint32_t>(physical_row_signed);
                const auto [physical_page, row_in_page] = divmod(physical_row, cache_page_rows);
                const auto [cache_height_tile, dest_tile_row] = divmod(row_in_page, tile_height);
                const uint32_t cache_tile =
                    ((physical_page * num_heads + head) * cache_height_tiles + cache_height_tile) * width_tiles +
                    width_tile;

                const auto [source_face_y, source_line] = divmod(tile_row, face_height);
                const auto [dest_face_y, dest_line] = divmod(dest_tile_row, face_height);
                for (uint32_t face_x = 0; face_x < tile_width / face_width; ++face_x) {
                    const uint32_t source_offset =
                        (source_face_y * 2 + face_x) * face_bytes + source_line * face_line_bytes;
                    const uint32_t dest_offset = (dest_face_y * 2 + face_x) * face_bytes + dest_line * face_line_bytes;
                    noc.async_write<NocOptions::TXN_ID>(
                        scratch_dfb,
                        cache,
                        face_line_bytes,
                        {.offset_bytes = source_offset},
                        {.page_id = cache_tile, .offset_bytes = dest_offset},
                        {.trid = current_slot + 1});
                    wrote_current_slot = true;
                }
            }
            // Transaction IDs split completion accounting only. These unicast writes share the
            // same NoC and default write VC, so program order (and last-source-row-wins) is retained.
            writes_in_flight[current_slot] = wrote_current_slot;

            if (has_next_tile) {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = next_slot + 1});
                scratch_dfb.push_back(1);
            }
            // The DFB slot may be released once its tagged writes have departed local L1.
            if (wrote_current_slot) {
                noc.async_writes_flushed<NocOptions::TXN_ID>({.trid = current_slot + 1});
            }
            scratch_dfb.pop_front(1);
        }

        for (uint32_t slot = 0; slot < scratch_buffer_depth; ++slot) {
            wait_for_slot_writes(slot);
        }

        // Transaction IDs are sticky command-buffer state; do not leak them to the next invocation.
        noc_async_read_set_trid(0, noc.get_noc_id());
        noc_async_write_set_trid(0, noc.get_noc_id());
    };

    for (uint32_t worker = worker_start; worker < worker_count; worker += worker_stride) {
        const auto [head, width_tile] = divmod(worker, width_tiles);
        copy_rows(input1, cache1, head, width_tile);
        copy_rows(input2, cache2, head, width_tile);
    }

    positions_dfb.pop_front(1);
}
