// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for indexer_score. Pops untilized 32x32 bf16 tiles and scatters
// their 32 rows as 64 B fragments into the row-major output (page = row,
// fragments are 64 B aligned). The owner of a group's last unit fills every
// row tail [group_valid_k_tiles*32, T) with -inf so skipped columns never
// read junk; compute already covers [0, group_valid_k_tiles) for every row.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

#include "indexer_score_common.hpp"

constexpr uint32_t page_bytes = get_compile_time_arg_val(num_common_ct_args);  // row-major page = T*2 bytes

constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_scratch = tt::CBIndex::c_17;    // writer-only -inf scratch tile
constexpr uint32_t cb_out_strip = tt::CBIndex::c_18;  // full-width fast-untilize strip output

constexpr uint32_t frag_bytes = tt::constants::TILE_WIDTH * sizeof(uint16_t);  // one bf16 tile row
constexpr uint32_t scratch_bytes = tt::constants::TILE_HW * sizeof(uint16_t);

/** Stamp the -inf scratch tile (plain L1 stores) and return its L1 address for reuse as a fill source. */
inline uint32_t fill_inf_scratch_and_get_addr() {
    fill_neginf_tile<scratch_bytes>(cb_scratch, /*tile_id=*/0);
    return CircularBuffer(cb_scratch).get_write_ptr();
}

/** Scatter one untilized tile's rows into output rows of q-tile-row q_row, column tile k_tile. */
template <typename OutAcc>
inline void write_tile(Noc noc, const OutAcc& out_acc, uint32_t q_row, uint32_t k_tile) {
    CircularBuffer cb(cb_out);
    cb.wait_front(1);
    uint32_t src = cb.get_read_ptr();
    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
        noc.async_write(
            CoreLocalMem<uint32_t>(src),
            out_acc,
            frag_bytes,
            {},
            {.page_id = q_row * tt::constants::TILE_HEIGHT + r, .offset_bytes = k_tile * frag_bytes});
        src += frag_bytes;
    }
    noc.async_write_barrier();
    cb.pop_front(1);
}

/** Scatter one full-width fast-untilize strip into output rows of q-tile-row q_row, columns
 *  starting at column tile k_tile_start. The strip is one wide row-major block in cb_out_strip:
 *  32 rows, each strip_w*32 contiguous bf16 (row pitch strip_w*32 elements). So each of the 32
 *  rows is a single strip_w*64-byte run written contiguously to the output row at column offset
 *  k_tile_start*64 -- one async_write per row instead of strip_w per-tile 64 B fragments. */
template <typename OutAcc>
inline void write_strip(Noc noc, const OutAcc& out_acc, uint32_t q_row, uint32_t k_tile_start, uint32_t strip_w) {
    CircularBuffer cb(cb_out_strip);
    cb.wait_front(strip_w);
    uint32_t src = cb.get_read_ptr();
    const uint32_t row_bytes = strip_w * frag_bytes;
    for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
        noc.async_write(
            CoreLocalMem<uint32_t>(src),
            out_acc,
            row_bytes,
            {},
            {.page_id = q_row * tt::constants::TILE_HEIGHT + rr, .offset_bytes = k_tile_start * frag_bytes});
        src += row_bytes;
    }
    noc.async_write_barrier();
    cb.pop_front(strip_w);
}

/** Fill the group's row tails [tail_start_tiles*32, T) with -inf from the L1 scratch tile.
 *  compute emits the full [0, group_valid) rectangle for every row in the group (a future cell
 *  of an upper row is a full -inf masked tile), so the tail the writer owns starts at the
 *  group's valid width, the same column for every row. Starting here (rather than each row's
 *  own valid length) keeps the writer's fill disjoint from compute's output, so when a group's
 *  units split across cores no two cores write the same output bytes. */
template <typename OutAcc>
inline void fill_group_tails(
    Noc noc, const OutAcc& out_acc, uint32_t scratch, uint32_t q_tile_start, uint32_t tail_start_tiles) {
    if (tail_start_tiles >= k_len_tiles) {
        return;  // group is fully causal-valid, no tail
    }
    const uint32_t tail_bytes_total = (k_len_tiles - tail_start_tiles) * frag_bytes;
    const uint32_t base_off = tail_start_tiles * frag_bytes;
    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
        for (uint32_t rr = 0; rr < tt::constants::TILE_HEIGHT; ++rr) {
            uint32_t off = base_off;
            uint32_t left = tail_bytes_total;
            while (left > 0) {
                const uint32_t n = left < scratch_bytes ? left : scratch_bytes;
                noc.async_write(
                    CoreLocalMem<uint32_t>(scratch),
                    out_acc,
                    n,
                    {},
                    {.page_id = (q_tile_start + r) * tt::constants::TILE_HEIGHT + rr, .offset_bytes = off});
                off += n;
                left -= n;
            }
        }
    }
    noc.async_write_barrier();
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t flat_start = get_arg_val<uint32_t>(1);
    const uint32_t flat_count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<num_common_ct_args + 1>();
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    Noc noc;

    const uint32_t scratch = fill_inf_scratch_and_get_addr();

    WorkUnitSpan span;
    span.start(flat_start);

    for (uint32_t i = 0; i < flat_count; ++i) {
        const uint32_t k_tiles_in_unit = span.k_tiles();
        const uint32_t k_tile0 = span.k_tile_start();
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            const uint32_t q_row_abs = span.q_tile_start() + r;
            // Mirror compute: a full-width unmasked row arrives as one strip in cb_out_strip;
            // every other row arrives as per-tile (r, c) tiles in cb_out.
            if constexpr (use_fast_strip) {
                if (row_valid_prefix(q_row_abs, k_tile0, k_tiles_in_unit) == k_tiles_per_unit) {
                    write_strip(noc, out_acc, q_row_abs, k_tile0, k_tiles_per_unit);
                    continue;
                }
            }
            for (uint32_t c = 0; c < k_tiles_in_unit; ++c) {
                write_tile(noc, out_acc, q_row_abs, k_tile0 + c);
            }
        }
        if (span.last_in_group()) {
            fill_group_tails(noc, out_acc, scratch, span.q_tile_start(), span.valid_k_tiles());
        }
        span.advance();
    }
}
