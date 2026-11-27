// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer (BRISC / NoC1) of the one-tile-column regimes, W <= 16, where every output row is a prefix
// of one face row of its tile's cb_left_faces slot.
//
// face_direct (store_face_rows_block): one write of 2 * W bytes per row, straight from its face row.
//
// same_bank_slot_writes (2 * W < the output's aligned page): the block's rows are gathered into a
// slot image laid out bank run by bank run (g6_to_layout_slot_image.hpp), and write_bank_runs writes
// each run in one NoC write. The reader builds rows [0, reader_rows) and pushes the image page on
// cb_slot_image; this RISC builds the rest. Block k's image is popped once its writes have left L1:
// after the flush before block k + 1's writes, so a block's runs drain while the next is built.
// Zones (profiler builds only): face_direct has per-block zones (its blocks are few) and sums its
// flushes; same_bank_slot_writes runs ~100 blocks per core, so it keeps two run totals instead
// (TT_METAL_PROFILER_SUM): writer_waits (faces, image and flush waits) and writer_build (its share
// of the build). The rest of the kernel zone is write issue.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_slot_image.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_left_faces = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool slot_writes = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr uint32_t cb_slot_image = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    constexpr uint32_t slot_image_depth = get_compile_time_arg_val(Geometry::next_ct_arg + 3);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(Geometry::next_ct_arg + 4);
    constexpr uint32_t reader_slot_eighths = get_compile_time_arg_val(Geometry::next_ct_arg + 5);
    constexpr bool writer_noc_copies = get_compile_time_arg_val(Geometry::next_ct_arg + 6) != 0;
    constexpr auto output_args = TensorAccessorArgs<Geometry::next_ct_arg + 7>();
    constexpr uint32_t row_bytes = Geometry::width * DATUM_BYTES;
    static_assert(Geometry::tensor_cols_tiles == 1 && Geometry::width <= FACE_HW);
    static_assert(!slot_writes || (row_bytes < slot_bytes && slot_image_depth >= 1));

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr);

    DeviceZoneScopedN("g6_face_writer");
    BlockWalk walk{tile_begin, tile_begin + num_tiles};
    Block block;

    if constexpr (!slot_writes) {
        while (walk.next(block)) {
            {
                DeviceZoneScopedN("wait_left_faces");
                cb_wait_front(cb_left_faces, 1);
            }
            {
                DeviceZoneScopedN("store_face_rows_block");
                uint32_t slot = get_read_ptr(cb_left_faces);
                for (uint32_t i = 0; i < block.rows; ++i) {
                    const uint32_t g = block.tile_row + i;
                    const uint32_t live_rows = live_rows_of(g);
                    const uint32_t first_page = first_row_page_of(g);
                    for (uint32_t r = 0; r < live_rows; ++r) {
                        const uint32_t src = slot + (r / FACE_HW) * FACE_BYTES + (r % FACE_HW) * FACE_ROW_BYTES;
                        noc_async_write(src, output.get_noc_addr(first_page + r), row_bytes);
                    }
                    slot += LEFT_SLOT_BYTES;
                }
                {
                    DeviceZoneScopedSumN1("left_faces_writes_flushed");
                    noc_async_writes_flushed();
                }
            }
            cb_pop_front(cb_left_faces, 1);
        }
    } else {
        constexpr uint32_t image_bytes = Geometry::block_rows_tiles * Geometry::tile_row_stage_rows * slot_bytes;
        const uint32_t banks = output.contiguous_page_stride();
        ASSERT(output.get_aligned_page_size() == slot_bytes);
        const uint32_t image_base = get_read_ptr(cb_slot_image);
        uint32_t image_idx = 0;
        uint32_t images_held = 0;  // earlier blocks' images whose runs may still be leaving L1

        while (walk.next(block)) {
            const SlotRuns<slot_bytes> runs(block, banks);
            ASSERT(output.num_contiguous_pages(runs.first_page, runs.first_page + runs.rows) == runs.slots(0));
            const uint32_t image = image_base + image_idx * image_bytes;
            image_idx = image_idx + 1 == slot_image_depth ? 0 : image_idx + 1;
            {
                DeviceZoneScopedSumN1("writer_waits");
                cb_wait_front(cb_left_faces, 1);
                cb_wait_front(cb_slot_image, images_held + 1);
            }
            {
                DeviceZoneScopedSumN2("writer_build");
                build_slot_rows<slot_bytes, writer_noc_copies>(
                    block, runs, get_read_ptr(cb_left_faces), image, runs.reader_rows(reader_slot_eighths), runs.rows);
            }
            cb_pop_front(cb_left_faces, 1);
            // write_bank_runs
            if (images_held > 0) {
                {
                    DeviceZoneScopedSumN1("writer_waits");
                    noc_async_writes_flushed();
                }
                cb_pop_front(cb_slot_image, images_held);
                images_held = 0;
            }
            uint32_t src = image;
            for (uint32_t j = 0; j < runs.count(); ++j) {
                const uint32_t run_bytes = runs.slots(j) * slot_bytes;
                noc_async_write(src, output.get_noc_addr(runs.first_page + j), run_bytes);
                src += run_bytes;
            }
            images_held = 1;
            if constexpr (slot_image_depth == 1) {
                // The reader's next image is this page: free it now.
                {
                    DeviceZoneScopedSumN1("writer_waits");
                    noc_async_writes_flushed();
                }
                cb_pop_front(cb_slot_image, 1);
                images_held = 0;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_slot_image, images_held);
    }
    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
