// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(2);
constexpr uint32_t input_slice_0_cb_id = get_compile_time_arg_val(3);
constexpr uint32_t input_slice_1_cb_id = get_compile_time_arg_val(4);
constexpr uint32_t input_slice_2_cb_id = get_compile_time_arg_val(5);
constexpr uint32_t input_slice_3_cb_id = get_compile_time_arg_val(6);
constexpr uint32_t input_slice_4_cb_id = get_compile_time_arg_val(7);
constexpr uint32_t input_slice_5_cb_id = get_compile_time_arg_val(8);
constexpr uint32_t input_slice_6_cb_id = get_compile_time_arg_val(9);
constexpr uint32_t input_slice_7_cb_id = get_compile_time_arg_val(10);
constexpr uint32_t intermediate_slice_0_cb_id = get_compile_time_arg_val(11);
constexpr uint32_t intermediate_slice_1_cb_id = get_compile_time_arg_val(12);
constexpr uint32_t intermediate_slice_2_cb_id = get_compile_time_arg_val(13);
constexpr uint32_t intermediate_slice_3_cb_id = get_compile_time_arg_val(14);
constexpr uint32_t intermediate_slice_4_cb_id = get_compile_time_arg_val(15);
constexpr uint32_t intermediate_slice_5_cb_id = get_compile_time_arg_val(16);
constexpr uint32_t intermediate_slice_6_cb_id = get_compile_time_arg_val(17);
constexpr uint32_t intermediate_slice_7_cb_id = get_compile_time_arg_val(18);
constexpr uint32_t compute_cb_id = get_compile_time_arg_val(19);

// NOTE: hardcoded for ring size of 8
constexpr uint32_t input_slice_cb_ids[8] = {
    input_slice_0_cb_id,
    input_slice_1_cb_id,
    input_slice_2_cb_id,
    input_slice_3_cb_id,
    input_slice_4_cb_id,
    input_slice_5_cb_id,
    input_slice_6_cb_id,
    input_slice_7_cb_id};

// NOTE: hardcoded for ring size of 8
constexpr uint32_t intermediate_slice_cb_ids[8] = {
    intermediate_slice_0_cb_id,
    intermediate_slice_1_cb_id,
    intermediate_slice_2_cb_id,
    intermediate_slice_3_cb_id,
    intermediate_slice_4_cb_id,
    intermediate_slice_5_cb_id,
    intermediate_slice_6_cb_id,
    intermediate_slice_7_cb_id};

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        // don't reduce on the first slice
        if (i == 0) {
            // next slice idx
            if (direction) {
                slice_idx--;
            } else {
                slice_idx++;
            }
            continue;
        }

        uint32_t actual_slice_idx;
        if (direction) {
            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
        } else {
            actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
        }

        uint32_t input_slice_cb_id = input_slice_cb_ids[actual_slice_idx];
        uint32_t intermediate_slice_cb_id = intermediate_slice_cb_ids[actual_slice_idx];

        binary_op_init_common(input_slice_cb_id, intermediate_slice_cb_id, compute_cb_id);
        add_tiles_init(input_slice_cb_id, intermediate_slice_cb_id, false);

        uint32_t tiles_read = start_tiles_read;
        uint32_t tiles_to_read = start_tiles_to_read;
        while (tiles_read < tiles_to_read) {
            cb_wait_front(input_slice_cb_id, tile_granularity);
            cb_wait_front(intermediate_slice_cb_id, tile_granularity);
            cb_reserve_back(compute_cb_id, tile_granularity);
            acquire_dst();
            for (uint32_t tile_id = 0; tile_id < tile_granularity; ++tile_id) {
                add_tiles(input_slice_cb_id, intermediate_slice_cb_id, tile_id, tile_id, tile_id);
                pack_tile(tile_id, compute_cb_id);
            }
            release_dst();
            cb_pop_front(input_slice_cb_id, tile_granularity);
            cb_pop_front(intermediate_slice_cb_id, tile_granularity);
            cb_push_back(compute_cb_id, tile_granularity);
            tiles_read += tile_granularity;
        }

        // next slice idx
        if (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }
}
