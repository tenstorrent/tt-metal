// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader (NCRISC / NoC0) of the one-tile-column regimes, W <= 16: load_left_faces_block.
//
// Faces 1 and 3 hold only padding and are never read. Each tile of a block lands in a 1024-B slot
// of the block's cb_left_faces page: the live-row prefix of face 0 at +0, of face 2 at +512.
// Block n's reads carry trid n % read_depth_max + 1. read_depth blocks are on the wire while this RISC
// builds its share of the oldest landed block's slot image (same_bank_slot_writes: rows
// [0, reader_rows) of g6_to_layout_slot_image.hpp, pushed on cb_slot_image). If the next block has
// still not landed after the build, more blocks are issued, up to read_depth_max: a core whose reads
// arrive late keeps more of them queued, a core whose reads keep up does not add to the queues.
// Only the oldest block's trid is waited on, so reads stay in flight across every wait.
// Zones (profiler builds only): two run totals (TT_METAL_PROFILER_SUM), so a run of ~100 blocks
// stays inside the RISC's marker buffer: reader_load (reserve, read issue, trid wait) and
// reader_build (image reserve and this RISC's share of the build).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_slot_image.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

using namespace g6_to_layout;
namespace dkl = dataflow_kernel_lib;

// Tiles are issued from tile `first` of the block on, wrapping; each lands in its own slot, so only
// the issue order depends on `first`.
template <typename Accessor>
FORCE_INLINE void load_left_faces(const Accessor& input, const Block& block, uint32_t slot0, uint32_t first) {
    for (uint32_t n = 0, i = first; n < block.rows; ++n, i = i + 1 == block.rows ? 0 : i + 1) {
        const uint32_t g = block.tile_row + i;
        const uint32_t slot = slot0 + i * LEFT_SLOT_BYTES;
        const uint32_t live_rows = live_rows_of(g);
        const uint32_t top_rows = min_u32(live_rows, FACE_HW);
        const uint64_t src = input.get_noc_addr(g);
        noc_async_read(src, slot, top_rows * FACE_ROW_BYTES);
        if (live_rows > FACE_HW) {
            noc_async_read(src + 2 * FACE_BYTES, slot + FACE_BYTES, (live_rows - FACE_HW) * FACE_ROW_BYTES);
        }
    }
}

// Non-blocking twin of async_read_barrier_with_trid: true once every read tagged `trid` has landed.
// The command-buffer poll orders the counter read after the NIU has taken the last issued read.
FORCE_INLINE bool reads_landed(Noc noc, uint32_t trid) {
    const uint32_t noc_id = noc.get_noc_id();
    while (!noc_cmd_buf_ready(noc_id, read_cmd_buf));
    return ncrisc_noc_read_with_transaction_id_flushed(noc_id, trid);
}

void kernel_main() {
    constexpr uint32_t cb_left_faces = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool slot_writes = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr uint32_t cb_slot_image = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    constexpr uint32_t reader_slot_eighths = get_compile_time_arg_val(Geometry::next_ct_arg + 3);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(Geometry::next_ct_arg + 4);
    constexpr uint32_t read_depth = get_compile_time_arg_val(Geometry::next_ct_arg + 5);
    constexpr uint32_t read_depth_max = get_compile_time_arg_val(Geometry::next_ct_arg + 6);
    constexpr uint32_t faces_depth = get_compile_time_arg_val(Geometry::next_ct_arg + 7);
    constexpr bool read_stagger = get_compile_time_arg_val(Geometry::next_ct_arg + 8) != 0;
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 9>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t faces_page_bytes = Geometry::block_rows_tiles * LEFT_SLOT_BYTES;
    static_assert(Geometry::tensor_cols_tiles == 1 && Geometry::width <= FACE_HW);
    // Issuing a block reserves up to read_depth_max pages while the page of the block being built is
    // still held: the writer pops it only after this RISC's share of its image, built after the issue.
    static_assert(read_depth >= 1 && read_depth <= read_depth_max && read_depth_max <= NOC_MAX_TRANSACTION_ID);
    static_assert(read_depth_max + 1 <= faces_depth);
    static_assert(2 * Geometry::block_rows_tiles <= NOC_MAX_TRANSACTION_ID_COUNT);

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr);
    // Only the output's bank interleaving is needed here (the run layout), never its address.
    const uint32_t banks = TensorAccessor(output_args, 0).contiguous_page_stride();
    // bank_stagger: each block's first read goes to bank (x + y) % in_banks of this core's logical
    // coordinates, so neighbouring cores open their blocks on different banks.
    const uint32_t in_banks = input.contiguous_page_stride();
    const uint32_t stagger_bank = (get_absolute_logical_x() + get_absolute_logical_y()) % in_banks;
    auto first_tile = [&](const Block& block) -> uint32_t {
        if constexpr (!read_stagger) {
            return 0;
        }
        return (stagger_bank + in_banks - block.tile_row % in_banks) % in_banks % block.rows;
    };

    DeviceZoneScopedN("g6_face_reader");
    Noc noc;
    // Block n lands in page n % faces_depth: the CB write pointer only advances on a push, so it
    // lags the blocks issued ahead of it, and their pages are addressed from the ring base.
    const uint32_t faces_base = get_write_ptr(cb_left_faces);
    BlockWalk walk{tile_begin, tile_begin + num_tiles};
    Block in_flight[read_depth_max];
    uint32_t issued = 0;
    uint32_t pushed = 0;

    // Issues the next block's reads while fewer than `depth` blocks are in flight.
    auto issue_below = [&](uint32_t depth) {
        Block next;
        if (issued - pushed >= depth || !walk.next(next)) {
            return false;
        }
        cb_reserve_back(cb_left_faces, issued - pushed + 1);
        dkl::set_read_trid(noc, issued % read_depth_max + 1);
        load_left_faces(input, next, faces_base + issued % faces_depth * faces_page_bytes, first_tile(next));
        in_flight[issued % read_depth_max] = next;
        ++issued;
        return true;
    };
    auto land_oldest = [&]() {
        const uint32_t trid = pushed % read_depth_max + 1;
        if constexpr (read_depth_max > read_depth) {
            while (!reads_landed(noc, trid) && issue_below(read_depth_max)) {
            }
        }
        dkl::async_read_barrier_with_trid(noc, trid);
        cb_push_back(cb_left_faces, 1);
        ++pushed;
    };

    {
        DeviceZoneScopedSumN1("reader_load");
        while (issue_below(read_depth)) {
        }
        if (issued > 0) {
            land_oldest();
        }
    }
    while (pushed > 0) {
        const uint32_t k = pushed - 1;
        const Block block = in_flight[k % read_depth_max];
        const uint32_t faces = faces_base + k % faces_depth * faces_page_bytes;
        {
            DeviceZoneScopedSumN1("reader_load");
            while (issue_below(read_depth)) {
            }
        }
        if constexpr (slot_writes) {
            DeviceZoneScopedSumN2("reader_build");
            const SlotRuns<slot_bytes> runs(block, banks);
            cb_reserve_back(cb_slot_image, 1);
            // RISC copies: this RISC's NoC reads of the next blocks are in flight.
            build_slot_rows<slot_bytes, false>(
                block, runs, faces, get_write_ptr(cb_slot_image), 0, runs.reader_rows(reader_slot_eighths));
            cb_push_back(cb_slot_image, 1);
        }
        if (pushed == issued) {
            break;
        }
        {
            DeviceZoneScopedSumN1("reader_load");
            land_oldest();
        }
    }
    dkl::set_read_trid(noc, 0);
}
