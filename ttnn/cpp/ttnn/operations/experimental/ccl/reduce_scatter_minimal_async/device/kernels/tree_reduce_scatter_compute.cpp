// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

///////////////////////////////////////////////////////////////////////////////
// Tree Reduce-Scatter Compute Kernel
//
// This kernel performs the reduction (addition) for the tree-based algorithm:
// - Waits for reader to provide local/accumulated data in cb_input
// - Waits for reader to provide received data in cb_intermediate
// - Adds the two and pushes result to cb_output for writer
//
// Multi-worker support:
// - Each worker has a direction (forward=1, backward=0)
// - Only processes slices where receive direction matches worker direction
// - Supports tile range splitting for multiple workers within same direction
///////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include "api/compute/eltwise_binary.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t num_steps = get_compile_time_arg_val(2);
constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(4);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(5);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(6);

// Tensor shape info
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(7);
constexpr uint32_t slice_C = get_compile_time_arg_val(8);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(9);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(10);
constexpr uint32_t dim = get_compile_time_arg_val(11);

// Worker direction: 0=backward, 1=forward
constexpr uint32_t direction = get_compile_time_arg_val(12);

///////////////////////////////////////////////////
// TREE ALGORITHM HELPER FUNCTIONS
///////////////////////////////////////////////////

// Compute offset of this device relative to slice owner
// Returns value normalized to range appropriate for tree algorithm
FORCE_INLINE int32_t compute_offset(uint32_t device_id, uint32_t slice_id, uint32_t ring_sz) {
    int32_t raw_offset = static_cast<int32_t>(device_id) - static_cast<int32_t>(slice_id);
    // Normalize to [-ring_sz/2, ring_sz/2]
    if (raw_offset > static_cast<int32_t>(ring_sz / 2)) {
        raw_offset -= ring_sz;
    } else if (raw_offset < -static_cast<int32_t>((ring_sz - 1) / 2)) {
        raw_offset += ring_sz;
    }
    return raw_offset;
}

// Left tree root: 0 for even ring_size, -1 for odd ring_size
FORCE_INLINE int32_t get_left_root(uint32_t ring_sz) { return (ring_sz % 2 == 0) ? 0 : -1; }

// Check if offset is a LEFT side receiver at given step
FORCE_INLINE bool is_left_receiver(int32_t offset, uint32_t step, uint32_t ring_sz) {
    int32_t left_root = get_left_root(ring_sz);
    if (offset > left_root) {
        return false;
    }

    int32_t adj = offset - left_root;  // adj <= 0
    uint32_t stride = 1u << step;
    uint32_t group = stride << 1;
    uint32_t abs_adj = static_cast<uint32_t>(-adj);

    // Receiver: even multiple of stride (including 0)
    return (abs_adj % group == 0);
}

// Check if offset is a RIGHT side receiver at given step
FORCE_INLINE bool is_right_receiver(int32_t offset, uint32_t step) {
    if (offset < 1) {
        return false;
    }

    uint32_t adj = static_cast<uint32_t>(offset - 1);  // adj >= 0
    uint32_t stride = 1u << step;
    uint32_t group = stride << 1;

    // Receiver: even multiple of stride
    return (adj % group == 0);
}

// Get receive direction: 0=none, 1=backward (from device-1), 2=forward (from device+1)
FORCE_INLINE uint32_t get_receive_direction(int32_t offset, uint32_t step, uint32_t num_steps_total, uint32_t ring_sz) {
    bool is_final = (step == num_steps_total - 1);

    if (is_final && offset == 0) {
        // Final step: owner receives from right side (+1)
        return 2;  // forward (from +1)
    }

    if (is_left_receiver(offset, step, ring_sz)) {
        return 1;  // backward (left side receivers get data from device-1)
    }

    if (is_right_receiver(offset, step)) {
        return 2;  // forward (right side receivers get data from device+1)
    }

    return 0;
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    uint32_t start_tile_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t end_tile_offset = get_arg_val<uint32_t>(arg_idx++);

    ///////////////////////////////////////////////////
    // INITIALIZATION
    ///////////////////////////////////////////////////

    binary_op_init_common(cb_input_id, cb_intermediate_id, cb_output_id);
    add_tiles_init(cb_input_id, cb_intermediate_id, false);

    // Derived constants
    constexpr uint32_t slice_num_pages = slice_C * slice_Ht * slice_Wt;
    constexpr uint32_t tiles_per_channel = (dim == 1) ? slice_num_pages : (slice_Ht * slice_Wt);
    constexpr uint32_t num_channel_iters = (dim == 1) ? 1 : slice_C;

    // Direction mask: forward=2, backward=1
    constexpr uint32_t my_direction_mask = direction ? 2 : 1;

    ///////////////////////////////////////////////////
    // MAIN LOOP
    ///////////////////////////////////////////////////

    for (uint32_t batch = 0; batch < input_tensor_B; ++batch) {
        for (uint32_t step = 0; step < num_steps; ++step) {
            for (uint32_t slice = 0; slice < ring_size; ++slice) {
                int32_t offset = compute_offset(my_chip_id, slice, ring_size);
                uint32_t recv_dir = get_receive_direction(offset, step, num_steps, ring_size);

                // Only process if this slice's receive direction matches our worker direction
                if ((recv_dir & my_direction_mask) == 0) {
                    continue;
                }

                // Process tiles for this slice within our assigned range
                for (uint32_t c = 0; c < num_channel_iters; ++c) {
                    // Compute tile range for this channel
                    uint32_t tile_start, tile_end;
                    if constexpr (dim == 1) {
                        tile_start = start_tile_offset;
                        tile_end = end_tile_offset;
                    } else {
                        uint32_t global_tile_start = c * tiles_per_channel;
                        uint32_t global_tile_end = (c + 1) * tiles_per_channel;

                        // Check if our range overlaps with this channel
                        if (end_tile_offset <= global_tile_start || start_tile_offset >= global_tile_end) {
                            continue;  // No overlap
                        }

                        // Compute local tile range within this channel
                        tile_start =
                            (start_tile_offset > global_tile_start) ? (start_tile_offset - global_tile_start) : 0;
                        tile_end = (end_tile_offset < global_tile_end) ? (end_tile_offset - global_tile_start)
                                                                       : tiles_per_channel;
                    }

                    for (uint32_t tile_offset = tile_start; tile_offset < tile_end; tile_offset += tile_granularity) {
                        uint32_t tiles_this_chunk = std::min(tile_granularity, tile_end - tile_offset);

                        // Wait for reader to provide both inputs
                        cb_wait_front(cb_input_id, tiles_this_chunk);
                        cb_wait_front(cb_intermediate_id, tiles_this_chunk);

                        // Reserve output space
                        cb_reserve_back(cb_output_id, tiles_this_chunk);

                        // Perform reduction: add tiles from input and intermediate
                        acquire_dst();
                        for (uint32_t t = 0; t < tiles_this_chunk; ++t) {
                            add_tiles(cb_input_id, cb_intermediate_id, t, t, t);
                            pack_tile(t, cb_output_id);
                        }
                        release_dst();

                        // Signal completion
                        cb_pop_front(cb_input_id, tiles_this_chunk);
                        cb_pop_front(cb_intermediate_id, tiles_this_chunk);
                        cb_push_back(cb_output_id, tiles_this_chunk);
                    }
                }
            }
        }
    }
}
