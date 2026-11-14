// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

// One fn dot core's operands, then, level by level, the partials its tree children write into cb_recv
// (semaphore l counts the level-l children that have delivered).
//
// The K-slice of A is read straight out of the collapse core whose ROW_MAJOR [H, D/8] shard holds it: a shard is
// that core's K-slices laid end to end. The [Kc, N] slab of B is fn_weight's L1 shard on this core (cb_in1 aliases
// it) or, from DRAM, its tiles [b_first_tile, b_first_tile + in1_num_tiles).
void kernel_main() {
    constexpr uint32_t in0_cb_index = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_index = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t recv_cb_index = get_named_compile_time_arg_val("cb_recv");

    constexpr uint32_t in0_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_levels = get_compile_time_arg_val(3);
    constexpr uint32_t a_slice_bytes = get_compile_time_arg_val(4);
    constexpr bool b_in_dram = get_compile_time_arg_val(5) != 0;
    constexpr auto b_args = TensorAccessorArgs<6>();

    const uint32_t a_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t a_noc_y = get_arg_val<uint32_t>(1);
    const uint32_t a_addr = get_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_arg_val<uint32_t>(3);
    const uint32_t b_first_tile = get_arg_val<uint32_t>(4);
    constexpr uint32_t children_arg = 5;

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    in0_cb.reserve_back(in0_num_tiles);
    in1_cb.reserve_back(in1_num_tiles);
    noc_async_read(get_noc_addr(a_noc_x, a_noc_y, a_addr), in0_cb.get_write_ptr(), a_slice_bytes);
    if constexpr (b_in_dram) {
        const auto b = TensorAccessor(b_args, b_addr);
        const uint32_t tile_bytes = in1_cb.get_tile_size();
        for (uint32_t t = 0; t < in1_num_tiles; ++t) {
            noc.async_read(b, in1_cb, tile_bytes, {.page_id = b_first_tile + t}, {.offset_bytes = t * tile_bytes});
        }
    }
    noc.async_read_barrier();
    in0_cb.push_back(in0_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    CircularBuffer recv_cb(recv_cb_index);
    for (uint32_t level = 0; level < num_levels; ++level) {
        // A core collects children on levels 0 .. L-1 for some L and on none above.
        const uint32_t num_children = get_arg_val<uint32_t>(children_arg + level);
        if (num_children == 0) {
            break;
        }
        const uint32_t level_num_tiles = num_children * block_num_tiles;
        recv_cb.reserve_back(level_num_tiles);
        Semaphore<> arrived(level);
        arrived.wait(num_children);
        arrived.set(0);
        recv_cb.push_back(level_num_tiles);
    }
}
