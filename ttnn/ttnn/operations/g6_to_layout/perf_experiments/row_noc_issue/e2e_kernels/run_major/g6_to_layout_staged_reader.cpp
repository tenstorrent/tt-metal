// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// R1 staged_reorder reader (NCRISC / NoC0): load_block + reorder_block.
//
// The core's blocks are read in batches of read_batch_tiles tiles into cb_tile_window (a
// reader-private ring of tile_window_depth batches) and reordered from there into the block's
// cb_row_block page with self-aimed 32-B NoC reads, one per live face row. While batch b is
// reordered, batches b+1 .. b+depth-1 are already on the wire under their own trids.
// Only live face-row prefixes are read; unread slot bytes map to stage bytes the writer never
// writes, so input padding never reaches the output.
// Zones (profiler builds only): per block, the batch issued ahead of the reserve, the reserve wait
// and the load + reorder work each have a zone; the per-batch and per-tile trid waits inside a block
// are sum zones (TT_METAL_PROFILER_SUM), so their count does not scale the marker buffer.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "row_issue.hpp"
namespace row_issue {
constexpr uint32_t G6_ROW_ISSUE = row_issue::ISSUE_STATE_BULK;
}

using namespace g6_to_layout;
namespace dkl = dataflow_kernel_lib;

namespace {

constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
constexpr uint32_t cb_row_block = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
constexpr uint32_t read_batch_tiles = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
constexpr uint32_t tile_window_depth = get_compile_time_arg_val(Geometry::next_ct_arg + 3);
constexpr uint32_t input_args_offset = Geometry::next_ct_arg + 4;

constexpr uint32_t batch_bytes = read_batch_tiles * TILE_BYTES;
// Trids 1..depth tag the DRAM batches; two more alternate per tile over the reorder copies so
// that no trid ever has more than one tile's 64 copies outstanding (the per-trid counter holds 255).
constexpr uint32_t reorder_trid_base = tile_window_depth + 1;
static_assert(tile_window_depth >= 1 && reorder_trid_base + 1 <= NOC_MAX_TRANSACTION_ID);
static_assert(2 * read_batch_tiles <= NOC_MAX_TRANSACTION_ID_COUNT);

// A run of up to read_batch_tiles consecutive tiles of one block, in row-major block order.
struct Batch {
    Block block;
    uint32_t i0;
    uint32_t j0;
    uint32_t count;
    bool first_of_block;
    bool last_of_block;
};

struct BatchCursor {
    BlockWalk walk;
    Block block;
    uint32_t i;
    uint32_t j;
    uint32_t tiles_left;

    FORCE_INLINE bool next(Batch& batch) {
        batch.first_of_block = tiles_left == 0;
        if (tiles_left == 0) {
            if (!walk.next(block)) {
                return false;
            }
            i = 0;
            j = 0;
            tiles_left = block.rows * block.cols;
        }
        batch.block = block;
        batch.i0 = i;
        batch.j0 = j;
        batch.count = min_u32(read_batch_tiles, tiles_left);
        tiles_left -= batch.count;
        batch.last_of_block = tiles_left == 0;
        j += batch.count;
        while (j >= block.cols) {
            j -= block.cols;
            ++i;
        }
        return true;
    }
};

// Live face-row prefixes of one tile into a 2048-B slot at the same face offsets; adjacent
// whole faces are merged into one read.
FORCE_INLINE void read_live_tile(uint64_t src, uint32_t dst, uint32_t live_rows, uint32_t live_cols) {
    const uint32_t top_rows = min_u32(live_rows, FACE_HW);
    const uint32_t bottom_rows = live_rows - top_rows;
    if (live_cols > FACE_HW) {
        if (bottom_rows == FACE_HW) {
            noc_async_read(src, dst, TILE_BYTES);
        } else if (bottom_rows > 0) {
            noc_async_read(src, dst, 2 * FACE_BYTES + bottom_rows * FACE_ROW_BYTES);
            noc_async_read(src + 3 * FACE_BYTES, dst + 3 * FACE_BYTES, bottom_rows * FACE_ROW_BYTES);
        } else if (top_rows == FACE_HW) {
            noc_async_read(src, dst, 2 * FACE_BYTES);
        } else {
            noc_async_read(src, dst, top_rows * FACE_ROW_BYTES);
            noc_async_read(src + FACE_BYTES, dst + FACE_BYTES, top_rows * FACE_ROW_BYTES);
        }
    } else {
        noc_async_read(src, dst, top_rows * FACE_ROW_BYTES);
        if (bottom_rows > 0) {
            noc_async_read(src + 2 * FACE_BYTES, dst + 2 * FACE_BYTES, bottom_rows * FACE_ROW_BYTES);
        }
    }
}

template <typename Accessor>
FORCE_INLINE void load_batch(const Batch& batch, uint32_t slot_base, const Accessor& input) {
    uint32_t i = batch.i0;
    uint32_t j = batch.j0;
    uint32_t dst = slot_base;
    for (uint32_t k = 0; k < batch.count; ++k) {
        const uint32_t g = batch.block.tile_row + i;
        const uint32_t c = batch.block.col + j;
        read_live_tile(input.get_noc_addr(g * Geometry::tensor_cols_tiles + c), dst, live_rows_of(g), live_cols_of(c));
        dst += TILE_BYTES;
        if (++j == batch.block.cols) {
            j = 0;
            ++i;
        }
    }
}

// Face row fr of face f of block tile (i, j) goes to stage row i * tile_row_stage_rows +
// (f / 2) * 16 + fr at byte j * 64 + (f % 2) * 32. Only rows < live_rows and the right half
// only when live_cols > 16 are copied.
FORCE_INLINE void reorder_batch(
    Noc noc, const Batch& batch, uint32_t slot_base, uint32_t stage_base, uint32_t& parity) {
    // Per batch, not per kernel: the DRAM reads of load_batch reprogram the same read cmd_buf.
    dkl::set_read_state<FACE_ROW_BYTES>(noc, slot_base);
    uint32_t i = batch.i0;
    uint32_t j = batch.j0;
    uint32_t tile = slot_base;
    for (uint32_t k = 0; k < batch.count; ++k) {
        const uint32_t g = batch.block.tile_row + i;
        const uint32_t live_rows = live_rows_of(g);
        const bool right_half = live_cols_of(batch.block.col + j) > FACE_HW;
        const uint32_t trid = reorder_trid_base + parity;
        dkl::set_read_trid(noc, trid);
        {
            DeviceZoneScopedSumN2("wait_reorder_copies");
            dkl::async_read_barrier_with_trid(noc, trid);
        }

        const uint32_t row_base = stage_base + i * Geometry::stage_tile_row_bytes + j * TILE_ROW_BYTES;
        const bool run_major =
            Geometry::full_row_blocks && batch.block.col == 0 && batch.block.cols == Geometry::tensor_cols_tiles;
        uint32_t dst = row_base;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t half_base = half * FACE_HW;
            if (live_rows <= half_base) {
                break;
            }
            const uint32_t rows = min_u32(live_rows - half_base, FACE_HW);
            uint32_t src = tile + half * 2 * FACE_BYTES;
            for (uint32_t r = 0; r < rows; ++r) {
                if (run_major) {
                    dst = row_base + row_issue::run_pos(half_base + r, live_rows) * Geometry::stage_row_bytes;
                }
                dkl::read_with_state<FACE_ROW_BYTES>(noc, dst, src);
                if (right_half) {
                    dkl::read_with_state<FACE_ROW_BYTES>(noc, dst + FACE_ROW_BYTES, src + FACE_BYTES);
                }
                src += FACE_ROW_BYTES;
                if (!run_major) {
                    dst += Geometry::stage_row_bytes;
                }
            }
        }
        parity ^= 1;
        tile += TILE_BYTES;
        if (++j == batch.block.cols) {
            j = 0;
            ++i;
        }
    }
    DeviceZoneScopedSumN2("wait_reorder_copies");
    dkl::async_read_barrier_with_trid(noc, reorder_trid_base);
    dkl::async_read_barrier_with_trid(noc, reorder_trid_base + 1);
}

}  // namespace

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<input_args_offset>();
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr);
    Noc noc;
    const uint32_t window_base = get_write_ptr(cb_tile_window);

    DeviceZoneScopedN("g6_staged_reader");
    BatchCursor issue_cursor{{tile_begin, tile_begin + num_tiles}, {}, 0, 0, 0};
    Batch in_flight[tile_window_depth];
    uint32_t issued = 0;
    uint32_t parity = 0;

    // Batch n lands in slot n % depth; it is issued depth - 1 batches ahead of its reorder, into
    // the slot whose previous batch was reordered (and its copies retired) one iteration earlier.
    auto issue_next = [&]() {
        Batch batch;
        if (!issue_cursor.next(batch)) {
            return;
        }
        const uint32_t slot = issued % tile_window_depth;
        dkl::set_read_trid(noc, 1 + slot);
        load_batch(batch, window_base + slot * batch_bytes, input);
        in_flight[slot] = batch;
        ++issued;
    };
    for (uint32_t d = 0; d + 1 < tile_window_depth; ++d) {
        issue_next();
    }
    // Batches never straddle blocks, so the batch at the top of each iteration opens a block.
    uint32_t done = 0;
    for (;;) {
        {
            DeviceZoneScopedN("issue_dram_batch");
            issue_next();
        }
        if (done == issued) {
            break;
        }
        {
            DeviceZoneScopedN("reserve_row_block");
            cb_reserve_back(cb_row_block, 1);
        }
        {
            DeviceZoneScopedN("load_reorder_block");
            const uint32_t stage_base = get_write_ptr(cb_row_block);
            for (;;) {
                const uint32_t slot = done % tile_window_depth;
                const Batch& batch = in_flight[slot];
                {
                    DeviceZoneScopedSumN1("wait_dram_batch");
                    dkl::async_read_barrier_with_trid(noc, 1 + slot);
                }
                reorder_batch(noc, batch, window_base + slot * batch_bytes, stage_base, parity);
                ++done;
                if (batch.last_of_block) {
                    break;
                }
                issue_next();
            }
        }
        cb_push_back(cb_row_block, 1);
    }
    dkl::set_read_trid(noc, 0);
    {
        DeviceZoneScopedN("final_read_barrier");
        noc_async_read_barrier();
    }
}
