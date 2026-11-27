// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// R2 face_direct writer (BRISC / NoC1): store_face_rows_block.
//
// Every output row is a prefix of one face row, so it is written straight from its face row in
// the cb_left_faces slot: one write of 2 * W bytes per row, no stage copy.
// Zones (profiler builds only): per block, the wait and the block's work each have a zone; the
// flush inside the work is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_left_faces = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr auto output_args = TensorAccessorArgs<Geometry::next_ct_arg + 1>();
    constexpr uint32_t row_bytes = Geometry::width * DATUM_BYTES;
    static_assert(Geometry::tensor_cols_tiles == 1 && Geometry::width <= FACE_HW);

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr);

    DeviceZoneScopedN("g6_face_writer");
    BlockWalk walk{tile_begin, tile_begin + num_tiles};
    Block block;
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
                const uint32_t first_page = first_out_page_of(g);
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
    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
