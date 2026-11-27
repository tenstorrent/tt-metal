// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// R2 face_direct reader (NCRISC / NoC0): load_left_faces_block.
//
// R2 runs when the tensor is one tile column with W <= 16: faces 1 and 3 hold only padding and
// are never read. Each tile of a block lands in a 1024-B slot of the block's cb_left_faces
// page: the live-row prefix of face 0 at +0, of face 2 at +512. One read barrier per block.
// Zones (profiler builds only): per block, the reserve wait and the block's work each have a zone;
// the read barrier inside the work is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_left_faces = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 1>();
    static_assert(Geometry::tensor_cols_tiles == 1 && Geometry::width <= FACE_HW);

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr);

    DeviceZoneScopedN("g6_face_reader");
    BlockWalk walk{tile_begin, tile_begin + num_tiles};
    Block block;
    while (walk.next(block)) {
        {
            DeviceZoneScopedN("reserve_left_faces");
            cb_reserve_back(cb_left_faces, 1);
        }
        {
            DeviceZoneScopedN("load_left_faces_block");
            uint32_t slot = get_write_ptr(cb_left_faces);
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
            {
                DeviceZoneScopedSumN1("left_faces_read_barrier");
                noc_async_read_barrier();
            }
        }
        cb_push_back(cb_left_faces, 1);
    }
}
