// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar reshape, TILE layout: T DM harts of one kernel, no DataflowBuffer, no host-built mapping tensor.
//
// Both tensors are rank 3, [B, H, W], stored as 32x32 tiles of four 16x16 faces (row-major inside a face).
// A reshape keeps the flat logical element order, so output element (b, y, x) is input element number
// (b * H_out + y) * W_out + x. The kernel derives the mapping itself, one face row of an output tile (at
// most 16 elements, contiguous in tile memory) at a time: the row is a run of consecutive flat indices,
// which is cut wherever the input leaves a row or a 16-column face. Each piece is contiguous in one input
// tile, so it is fetched with one NOC read straight from that tile into its place in the output tile (the
// Quasar NOC reads and writes at byte granularity). Pieces that continue each other in both the input
// tile and the output tile (the usual case when the last dimension is unchanged: whole face columns) are
// merged into one read. No RISC ever touches the payload. Padding elements of an output tile are not
// written and keep whatever the stage held.
//
// Hart t of T owns windows (tiles_per_window consecutive output tiles) t, t + T, ... of this node's range
// and tags its transfers with transaction id t + 1, so its barrier waits on its own transfers only. Each
// hart ping-pongs between two stage halves: the tile writes of window k drain while the reads of window
// k + 1 fill the other half, and the one barrier per window covers both.
//
// Compile-time args: element_bytes, in_h, in_w, out_h, out_w (logical), in_tiles_h, in_tiles_w,
//                    out_tiles_h, out_tiles_w, num_out_tiles, tiles_per_window
// Runtime args:      window_start, num_windows

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t FACE = 16;                    // face height and width, elements
constexpr uint32_t TILE = 2 * FACE;              // tile height and width, elements
constexpr uint32_t FACE_ELEMENTS = FACE * FACE;  // elements of one face
constexpr uint32_t TILE_ELEMENTS = TILE * TILE;  // elements of one tile

// A pending NOC read: `bytes` bytes at `src_offset` of input tile `page` to `dst_offset` of the stage.
struct Run {
    uint32_t page;
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t bytes;
};

}  // namespace

void kernel_main() {
    constexpr uint32_t element_bytes = get_arg(args::element_bytes);
    constexpr uint32_t in_h = get_arg(args::in_h);
    constexpr uint32_t in_w = get_arg(args::in_w);
    constexpr uint32_t out_h = get_arg(args::out_h);
    constexpr uint32_t out_w = get_arg(args::out_w);
    constexpr uint32_t in_tiles_h = get_arg(args::in_tiles_h);
    constexpr uint32_t in_tiles_w = get_arg(args::in_tiles_w);
    constexpr uint32_t out_tiles_h = get_arg(args::out_tiles_h);
    constexpr uint32_t out_tiles_w = get_arg(args::out_tiles_w);
    constexpr uint32_t num_out_tiles = get_arg(args::num_out_tiles);
    constexpr uint32_t tiles_per_window = get_arg(args::tiles_per_window);
    constexpr uint32_t tile_bytes = TILE_ELEMENTS * element_bytes;
    constexpr uint32_t window_bytes = tiles_per_window * tile_bytes;

    const uint32_t window_start = get_arg(args::window_start);
    const uint32_t num_windows = get_arg(args::num_windows);

    const uint32_t t = get_my_thread_id();
    const uint32_t T = get_num_threads();

    Noc noc;
    const auto src = TensorAccessor(tensor::input);
    const auto dst = TensorAccessor(tensor::output);
    // One Scratchpad per node: hart t owns bytes [t * 2 * window_bytes, (t + 1) * 2 * window_bytes).
    Scratchpad<uint8_t> stage(scratch::stage);
    const uint32_t stage_base = t * 2 * window_bytes;
    const NocOptVals txn{.trid = t + 1};

    Run run{};
    auto flush = [&]() __attribute__((always_inline)) {
        if (run.bytes != 0) {
            noc.async_read<NocOptions::TXN_ID>(
                src,
                stage,
                run.bytes,
                {.page_id = run.page, .offset_bytes = run.src_offset},
                {.offset_bytes = run.dst_offset},
                txn);
            run.bytes = 0;
        }
    };

    uint32_t half = 0;
    for (uint32_t w = t; w < num_windows; w += T) {
        const uint32_t tile_first = (window_start + w) * tiles_per_window;
        const uint32_t tile_end =
            num_out_tiles - tile_first < tiles_per_window ? num_out_tiles : tile_first + tiles_per_window;
        const uint32_t base = stage_base + half * window_bytes;

        for (uint32_t tile = tile_first; tile < tile_end; ++tile) {
            const uint32_t tile_base = base + (tile - tile_first) * tile_bytes;
            const uint32_t b = tile / (out_tiles_h * out_tiles_w);
            const uint32_t in_plane = tile % (out_tiles_h * out_tiles_w);
            const uint32_t y0 = (in_plane / out_tiles_w) * TILE;
            const uint32_t x0 = (in_plane % out_tiles_w) * TILE;
            // Faces in memory order (top-left, top-right, bottom-left, bottom-right), rows top to bottom, so
            // consecutive iterations are consecutive in the output tile.
            for (uint32_t face = 0; face < 4; ++face) {
                const uint32_t fy = y0 + (face >> 1) * FACE;
                const uint32_t fx = x0 + (face & 1) * FACE;
                if (fy >= out_h || fx >= out_w) {
                    continue;  // a face of padding only
                }
                const uint32_t rows = out_h - fy < FACE ? out_h - fy : FACE;
                const uint32_t cols = out_w - fx < FACE ? out_w - fx : FACE;
                for (uint32_t r = 0; r < rows; ++r) {
                    uint32_t dst_offset = tile_base + (face * FACE_ELEMENTS + r * FACE) * element_bytes;
                    uint64_t flat = (static_cast<uint64_t>(b) * out_h + fy + r) * out_w + fx;
                    for (uint32_t left = cols; left != 0;) {
                        const uint32_t xi = flat % in_w;
                        const uint64_t row = flat / in_w;
                        const uint32_t yi = row % in_h;
                        const uint32_t bi = row / in_h;
                        // Up to the end of the input row and of the input face, whichever comes first.
                        uint32_t n = left;
                        if (n > in_w - xi) {
                            n = in_w - xi;
                        }
                        if (n > FACE - (xi % FACE)) {
                            n = FACE - (xi % FACE);
                        }
                        const uint32_t page = (bi * in_tiles_h + yi / TILE) * in_tiles_w + xi / TILE;
                        const uint32_t in_face = ((yi / FACE) & 1) * 2 + ((xi / FACE) & 1);
                        const uint32_t src_offset =
                            (in_face * FACE_ELEMENTS + (yi % FACE) * FACE + xi % FACE) * element_bytes;
                        const uint32_t bytes = n * element_bytes;
                        if (run.bytes != 0 && run.page == page && run.src_offset + run.bytes == src_offset &&
                            run.dst_offset + run.bytes == dst_offset) {
                            run.bytes += bytes;
                        } else {
                            flush();
                            run = {.page = page, .src_offset = src_offset, .dst_offset = dst_offset, .bytes = bytes};
                        }
                        flat += n;
                        dst_offset += bytes;
                        left -= n;
                    }
                }
            }
        }
        flush();
        // Also covers the previous window's writes, whose stage half the next window reads into.
        noc.async_read_barrier<NocOptions::TXN_ID>(txn);

        for (uint32_t tile = tile_first; tile < tile_end; ++tile) {
            noc.async_write<NocOptions::TXN_ID>(
                stage,
                dst,
                tile_bytes,
                {.offset_bytes = base + (tile - tile_first) * tile_bytes},
                {.page_id = tile},
                txn);
        }
        half ^= 1;
    }
    noc.async_write_barrier<NocOptions::TXN_ID>(txn);
}
