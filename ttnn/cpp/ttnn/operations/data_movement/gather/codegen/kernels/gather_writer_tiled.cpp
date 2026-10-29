// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather writer -- TILED mode (BRISC): dual-RISC partner of
// gather_reader_tiled.cpp.
//
// Splits work by OUTPUT TILE (total_work = Ht * Wt_index). Each core owns a
// contiguous output-tile range [start, start+n) and walks it in (row h, column
// sub-range) chunks -- the SAME arithmetic as the reader, so the two RISCs stay
// tile-aligned to row boundaries through dfb::input / dfb::output.
//
// Per row chunk:
//   Phase 1: load the FULL input row h (all Wt_input tiles, DRAM ordinal
//            h*Wt_input + c) into dfb::input. This is the redundant-but-bounded
//            load that guarantees the reader's residency invariant: each core
//            loads each row it touches exactly once (guarded by the row change),
//            so a row is loaded at most once per core owning tiles in it.
//   Phase 2: flush the w_count completed output tiles the reader produced
//            (DRAM ordinal g + jw = h*Wt_index + w) to DRAM.

#include "codegen_gather_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"
#include <cstdint>

void kernel_main() {
    // Runtime args: per-core contiguous output-tile range [start, start+n)
    const uint32_t start = get_arg(args::start);
    const uint32_t n = get_arg(args::n);

    // Compile-time args. No core count: this factory hands each core a contiguous output-tile
    // range, so no row is ever derived from a core ordinal.
    constexpr uint32_t Wt_input = get_arg(args::Wt_input);
    constexpr uint32_t Wt_index = get_arg(args::Wt_index);
    // Both come from gather_output_cb_tiles()/kGatherWriteBatchTiles, the same values the factory
    // sizes dfb::output with; the Phase 2 wrap clamp below is only correct against the real depth.
    constexpr uint32_t OUT_DFB_DEPTH = get_arg(args::OUT_DFB_DEPTH);
    constexpr uint32_t WRITE_BATCH = get_arg(args::WRITE_BATCH);

    constexpr uint32_t READ_BATCH = kGatherReadBatchTiles;
    // Read-pointer position within the output DFB ring (see the Phase 2 clamp below).
    uint32_t out_dfb_pos = 0;

    constexpr uint32_t input_tile_bytes = get_tile_size(dfb::input);
    const auto input_accessor = TensorAccessor(tensor::input);

    constexpr uint32_t output_tile_bytes = get_tile_size(dfb::output);
    const auto output_accessor = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer input_buffer(dfb::input);
    DataflowBuffer output_buffer(dfb::output);

    uint32_t g = start;
    uint32_t remaining = n;
    while (remaining > 0) {
        const uint32_t h = g / Wt_index;
        const uint32_t w0 = g % Wt_index;
        uint32_t w_count = Wt_index - w0;
        if (w_count > remaining) {
            w_count = remaining;
        }

        // --- Phase 1: load FULL input row h (Wt_input tiles) into dfb::input ---
        uint32_t tiles_read = 0;
        while (tiles_read < Wt_input) {
            uint32_t batch = (Wt_input - tiles_read < READ_BATCH) ? (Wt_input - tiles_read) : READ_BATCH;
            input_buffer.reserve_back(batch);
            uint32_t l1_offset = 0;
            for (uint32_t b = 0; b < batch; b++) {
                noc.async_read(
                    input_accessor,
                    input_buffer,
                    input_tile_bytes,
                    {.page_id = h * Wt_input + tiles_read + b, .offset_bytes = 0},
                    {.offset_bytes = l1_offset});
                l1_offset += input_tile_bytes;
            }
            noc.async_read_barrier();
            input_buffer.push_back(batch);
            tiles_read += batch;
        }

        // --- Phase 2: write this chunk's w_count output tiles to DRAM ---
        // Output tile (h, w0+jw) lives at DRAM ordinal g + jw.
        uint32_t tiles_written = 0;
        while (tiles_written < w_count) {
            uint32_t batch = (w_count - tiles_written < WRITE_BATCH) ? (w_count - tiles_written) : WRITE_BATCH;
            // pop_front wraps the read pointer only when a pop lands exactly on the
            // ring end (dataflow_api.h pop contract), and a flat multi-tile read must
            // not cross it: clamp each batch to the remaining distance to the wrap.
            const uint32_t to_wrap = OUT_DFB_DEPTH - out_dfb_pos;
            if (batch > to_wrap) {
                batch = to_wrap;
            }
            output_buffer.wait_front(batch);
            uint32_t l1_offset = 0;
            for (uint32_t b = 0; b < batch; b++) {
                noc.async_write(
                    output_buffer,
                    output_accessor,
                    output_tile_bytes,
                    {.offset_bytes = l1_offset},
                    {.page_id = g + tiles_written + b, .offset_bytes = 0});
                l1_offset += output_tile_bytes;
            }
            // Popping the slot only needs the write off the local NoC, not landed at the
            // destination; completion is claimed once for the whole kernel below.
            noc.async_writes_flushed();
            output_buffer.pop_front(batch);
            tiles_written += batch;
            out_dfb_pos += batch;
            if (out_dfb_pos == OUT_DFB_DEPTH) {
                out_dfb_pos = 0;
            }
        }

        g += w_count;
        remaining -= w_count;
    }

    noc.async_write_barrier();
}
