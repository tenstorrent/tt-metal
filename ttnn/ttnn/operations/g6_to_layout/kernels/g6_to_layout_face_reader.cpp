// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader (NCRISC / NoC0) of the one-tile-column regimes, W <= 16: load_left_faces_block.
//
// Faces 1 and 3 hold only padding and are never read. Each tile of a block lands in a 1024-B slot
// of the block's cb_left_faces page: the live-row prefix of face 0 at +0, of face 2 at +512. Block
// k + 1's reads are issued before block k's page is handed on, so one block of reads stays in
// flight while this RISC builds its share of block k's slot image (same_bank_slot_writes: rows
// [0, reader_rows) of g6_to_layout_slot_image.hpp, pushed on cb_slot_image).
// Zones (profiler builds only): two run totals (TT_METAL_PROFILER_SUM), so a run of ~100 blocks
// stays inside the RISC's marker buffer: reader_load (reserve, read issue, read barrier) and
// reader_build (image reserve and this RISC's share of the build).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_slot_image.hpp"

using namespace g6_to_layout;

template <typename Accessor>
FORCE_INLINE void load_left_faces(const Accessor& input, const Block& block, uint32_t slot) {
    for (uint32_t i = 0; i < block.rows; ++i) {
        const uint32_t g = block.tile_row + i;
        const uint32_t live_rows = live_rows_of(g);
        const uint32_t top_rows = min_u32(live_rows, FACE_HW);
        const uint64_t src = input.get_noc_addr(g);
        noc_async_read(src, slot, top_rows * FACE_ROW_BYTES);
        if (live_rows > FACE_HW) {
            noc_async_read(src + 2 * FACE_BYTES, slot + FACE_BYTES, (live_rows - FACE_HW) * FACE_ROW_BYTES);
        }
        slot += LEFT_SLOT_BYTES;
    }
}

void kernel_main() {
    constexpr uint32_t cb_left_faces = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool slot_writes = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr uint32_t cb_slot_image = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    constexpr uint32_t reader_slot_eighths = get_compile_time_arg_val(Geometry::next_ct_arg + 3);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(Geometry::next_ct_arg + 4);
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 5>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    static_assert(Geometry::tensor_cols_tiles == 1 && Geometry::width <= FACE_HW);

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr);
    // Only the output's bank interleaving is needed here (the run layout), never its address.
    const uint32_t banks = TensorAccessor(output_args, 0).contiguous_page_stride();

    DeviceZoneScopedN("g6_face_reader");
    BlockWalk walk{tile_begin, tile_begin + num_tiles};
    Block block;
    if (!walk.next(block)) {
        return;
    }
    cb_reserve_back(cb_left_faces, 1);
    uint32_t faces = get_write_ptr(cb_left_faces);
    load_left_faces(input, block, faces);
    noc_async_read_barrier();
    cb_push_back(cb_left_faces, 1);

    bool more = true;
    while (more) {
        Block next;
        more = walk.next(next);
        uint32_t next_faces = 0;
        if (more) {
            DeviceZoneScopedSumN1("reader_load");
            cb_reserve_back(cb_left_faces, 1);
            next_faces = get_write_ptr(cb_left_faces);
            load_left_faces(input, next, next_faces);
        }
        if constexpr (slot_writes) {
            DeviceZoneScopedSumN2("reader_build");
            const SlotRuns<slot_bytes> runs(block, banks);
            cb_reserve_back(cb_slot_image, 1);
            // RISC copies: this RISC's NoC reads of the next block are in flight.
            build_slot_rows<slot_bytes, false>(
                block, runs, faces, get_write_ptr(cb_slot_image), 0, runs.reader_rows(reader_slot_eighths));
            cb_push_back(cb_slot_image, 1);
        }
        if (more) {
            {
                DeviceZoneScopedSumN1("reader_load");
                noc_async_read_barrier();
            }
            cb_push_back(cb_left_faces, 1);
        }
        block = next;
        faces = next_faces;
    }
}
