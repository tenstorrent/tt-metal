// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar slice, TILE element gather: one data-movement kernel of up to 6 threads per cluster, no
// DataflowBuffer. Used when the last two dims of a TILE slice do not start on a tile boundary or are
// strided, so an output tile is assembled in L1 from elements of several input tiles.
//
// Work model (host: slice_program_factory.cpp). item = output tile id; a cluster owns the contiguous range
// [item_start, item_start + item_count) and thread t of T builds tiles item_start + t, + T, ... of it.
//
// Output tile (plane, th, tw): plane counts the dims above the last two in mixed radix (extent[d], innermost
// last) and its first input tile is base_page + sum_d digit[d] * coef[d]. Output element (r, c) of the tile
// is input element (y, x) = (begin_h + (32 th + r) step_h, begin_w + (32 tw + c) step_w) of that plane, which
// lives in input tile (y / 32, x / 32). Rows / columns past the output's logical extent are zero.
//
// Staging: a thread owns 1 + SRC_SLOTS slots of slot_bytes in the per-cluster scratchpad. Slot 0 holds
// the output tile; the others take the input tiles it needs, SRC_SLOTS NoC reads in flight at a time. The
// element copies run under Scratchpad locks: the DM harts reach L1 through their cache, so a lock drops the
// stale lines of a freshly read tile before the loads and flushes the assembled tile before the NoC write.
//
// A tile is four 16x16 faces, row major inside a face: element (r, c) is at
//     ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + c % 16.
//
// Compile-time args: elem_bytes
// Common runtime args: base_page, tile_bytes, slot_bytes, in_wt, out_ht, out_wt, out_h, out_w, begin_h,
//                      begin_w, step_h, step_w
// Common runtime array args: extent[], coef[] (one entry per page dim above the tile plane)
// Runtime args (per cluster): item_start, item_count

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t SRC_SLOTS = 4;  // input tiles in flight per thread: a unit-step tile needs at most 2 x 2

FORCE_INLINE uint32_t row_offset(uint32_t r) { return ((r >> 4) << 9) | ((r & 15) << 4); }
FORCE_INLINE uint32_t col_offset(uint32_t c) { return ((c >> 4) << 8) | (c & 15); }

// The output rows [r0, r1) x columns [c0, c1) of the tile being built all come from one input tile.
struct Block {
    uint32_t r0, r1, c0, c1;
};

// First index past i (below end) whose source coordinate leaves the input tile of index i.
FORCE_INLINE uint32_t run_end(uint32_t i, uint32_t end, uint32_t src0, uint32_t step) {
    const uint32_t tile = (src0 + i * step) / TILE_DIM;
    do {
        ++i;
    } while (i < end && (src0 + i * step) / TILE_DIM == tile);
    return i;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    static_assert(elem_bytes == 2 || elem_bytes == 4, "slice tile gather moves 16-bit or 32-bit elements");
    using Elem = std::conditional_t<elem_bytes == 4, uint32_t, uint16_t>;

    const uint32_t item_start = get_arg(args::item_start);
    const uint32_t item_count = get_arg(args::item_count);

    const uint32_t base_page = get_arg(args::base_page);
    const uint32_t tile_bytes = get_arg(args::tile_bytes);
    const uint32_t slot_bytes = get_arg(args::slot_bytes);
    const uint32_t in_wt = get_arg(args::in_wt);
    const uint32_t out_ht = get_arg(args::out_ht);
    const uint32_t out_wt = get_arg(args::out_wt);
    const uint32_t out_h = get_arg(args::out_h);
    const uint32_t out_w = get_arg(args::out_w);
    const uint32_t begin_h = get_arg(args::begin_h);
    const uint32_t begin_w = get_arg(args::begin_w);
    const uint32_t step_h = get_arg(args::step_h);
    const uint32_t step_w = get_arg(args::step_w);

    const auto extent = get_arg(args::extent);
    const auto coef = get_arg(args::coef);

    const uint32_t thread = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    Noc noc;
    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);
    Scratchpad<uint8_t> stage(scratch::stage);
    const uint32_t out_slot = thread * (1 + SRC_SLOTS) * slot_bytes;  // then SRC_SLOTS input-tile slots
    Elem* const out_tile = reinterpret_cast<Elem*>(&stage[out_slot]);

    const uint32_t tiles_per_plane = out_ht * out_wt;
    const uint32_t item_end = item_start + item_count;
    for (uint32_t item = item_start + thread; item < item_end; item += num_threads) {
        uint32_t plane = item / tiles_per_plane;
        const uint32_t in_plane = item - plane * tiles_per_plane;
        const uint32_t th = in_plane / out_wt;
        const uint32_t tw = in_plane - th * out_wt;

        uint32_t plane_page = base_page;
        for (uint32_t d = extent.size(); d-- > 0;) {
            const uint32_t q = plane / extent[d];
            plane_page += (plane - q * extent[d]) * coef[d];
            plane = q;
        }

        // Valid rows / columns of this output tile and the input coordinates of its element (0, 0).
        const uint32_t rows = out_h - th * TILE_DIM < TILE_DIM ? out_h - th * TILE_DIM : TILE_DIM;
        const uint32_t cols = out_w - tw * TILE_DIM < TILE_DIM ? out_w - tw * TILE_DIM : TILE_DIM;
        const uint32_t y0 = begin_h + th * TILE_DIM * step_h;
        const uint32_t x0 = begin_w + tw * TILE_DIM * step_w;

        {
            const auto out_lock = stage.scoped_lock(out_slot, tile_bytes);
            if (rows < TILE_DIM || cols < TILE_DIM) {
                for (uint32_t e = 0; e < TILE_DIM * TILE_DIM; ++e) {
                    out_tile[e] = 0;
                }
            }

            Block blocks[SRC_SLOTS];
            uint32_t n = 0;
            // Copy the n staged blocks out of their input tiles (all reads have landed).
            auto drain = [&]() __attribute__((always_inline)) {
                noc.async_read_barrier();
                for (uint32_t b = 0; b < n; ++b) {
                    const auto src_lock = stage.scoped_lock(out_slot + (1 + b) * slot_bytes, tile_bytes);
                    const Elem* src_tile = reinterpret_cast<const Elem*>(&stage[out_slot + (1 + b) * slot_bytes]);
                    for (uint32_t r = blocks[b].r0; r < blocks[b].r1; ++r) {
                        const Elem* src_row = src_tile + row_offset((y0 + r * step_h) % TILE_DIM);
                        Elem* out_row = out_tile + row_offset(r);
                        for (uint32_t c = blocks[b].c0; c < blocks[b].c1; ++c) {
                            out_row[col_offset(c)] = src_row[col_offset((x0 + c * step_w) % TILE_DIM)];
                        }
                    }
                }
                n = 0;
            };

            for (uint32_t r0 = 0; r0 < rows;) {
                const uint32_t r1 = run_end(r0, rows, y0, step_h);
                const uint32_t src_th = (y0 + r0 * step_h) / TILE_DIM;
                for (uint32_t c0 = 0; c0 < cols;) {
                    const uint32_t c1 = run_end(c0, cols, x0, step_w);
                    const uint32_t src_tw = (x0 + c0 * step_w) / TILE_DIM;
                    noc.async_read(
                        input,
                        stage,
                        tile_bytes,
                        {.page_id = plane_page + src_th * in_wt + src_tw},
                        {.offset_bytes = out_slot + (1 + n) * slot_bytes});
                    blocks[n++] = {r0, r1, c0, c1};
                    if (n == SRC_SLOTS) {
                        drain();
                    }
                    c0 = c1;
                }
                r0 = r1;
            }
            if (n != 0) {
                drain();
            }
        }

        noc.async_write(stage, output, tile_bytes, {.offset_bytes = out_slot}, {.page_id = item});
        noc.async_writes_flushed();
    }
    noc.async_write_barrier();
}
