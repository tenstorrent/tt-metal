// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

///////////////////////////////////////////////////////////////////////////////
// Tree Reduce-Scatter Reader Kernel
//
// This kernel reads data for the tree-based reduce-scatter algorithm:
// - Reads received data from intermediate buffer (sent by neighbor)
// - Reads local/accumulated data from input tensor (step 0) or intermediate (step 1+)
// - Pushes both to CBs for compute kernel to reduce
//
// Multi-worker support:
// - Each worker has a direction (forward=1, backward=0)
// - Forward workers handle slices received from forward direction (device+1)
// - Backward workers handle slices received from backward direction (device-1)
// - Multiple workers within same direction split tiles among themselves
///////////////////////////////////////////////////////////////////////////////

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t num_steps = get_compile_time_arg_val(2);
constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);         // local/accumulated data
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(4);  // received data
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);

// Tensor shape info (for 4D tensor: [B, C, H, W] in tiles)
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(7);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(8);  // full tensor width in tiles
constexpr uint32_t slice_C = get_compile_time_arg_val(9);          // slice channels
constexpr uint32_t slice_Ht = get_compile_time_arg_val(10);        // slice height in tiles
constexpr uint32_t slice_Wt = get_compile_time_arg_val(11);        // slice width in tiles
constexpr uint32_t dim = get_compile_time_arg_val(12);             // scatter dimension (1=C, 2=H, 3=W)

// Worker direction: 0=backward, 1=forward
constexpr uint32_t direction = get_compile_time_arg_val(13);

// TensorAccessor compile-time args start here
constexpr uint32_t tensor_args_start = 14;

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

// Compute the starting tile ID for a slice based on scatter dimension
FORCE_INLINE uint32_t get_slice_start_tile_id(
    uint32_t slice_idx,
    uint32_t batch_idx,
    uint32_t channel_idx,
    uint32_t batch_num_pages,
    uint32_t channel_num_pages) {
    if constexpr (dim == 3) {
        // Scatter on W: slice_idx indexes into W dimension
        return batch_idx * batch_num_pages + channel_idx * channel_num_pages + slice_idx * slice_Wt;
    } else if constexpr (dim == 2) {
        // Scatter on H: slice_idx indexes into H dimension
        return batch_idx * batch_num_pages + channel_idx * channel_num_pages + slice_idx * slice_Ht * slice_Wt;
    } else if constexpr (dim == 1) {
        // Scatter on C: slice_idx indexes into C dimension
        return batch_idx * batch_num_pages + slice_idx * slice_C * slice_Ht * slice_Wt;
    } else {
        ASSERT(false);
        return 0;
    }
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t sem_addr = get_arg_val<uint32_t>(arg_idx++);  // semaphore for this direction
    uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tile_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t end_tile_offset = get_arg_val<uint32_t>(arg_idx++);

    ///////////////////////////////////////////////////
    // TENSOR ACCESSOR SETUP
    ///////////////////////////////////////////////////

    constexpr auto input_tensor_args = TensorAccessorArgs<tensor_args_start>();
    constexpr uint32_t input_ct_offset = input_tensor_args.num_compile_time_args();
    auto input_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, page_size);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<tensor_args_start + input_ct_offset>();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address, page_size);

    ///////////////////////////////////////////////////
    // DERIVED CONSTANTS
    ///////////////////////////////////////////////////

    // Pages per batch and per channel (for indexing)
    constexpr uint32_t slice_num_pages = slice_C * slice_Ht * slice_Wt;
    constexpr uint32_t batch_num_pages = ring_size * slice_num_pages;  // all slices in a batch
    constexpr uint32_t channel_num_pages = slice_Ht * slice_Wt;
    constexpr uint32_t tiles_per_channel = (dim == 1) ? slice_num_pages : channel_num_pages;
    constexpr uint32_t num_channel_iters = (dim == 1) ? 1 : slice_C;

    // Intermediate buffer layout: [buffer_0][buffer_1] where each buffer holds ring_size slices
    // buffer_idx 0 or 1, then slice, then tiles within slice
    auto get_intermediate_tile_id = [](uint32_t buffer_idx, uint32_t slice_idx, uint32_t tile_offset) -> uint32_t {
        return buffer_idx * ring_size * slice_num_pages + slice_idx * slice_num_pages + tile_offset;
    };

    // Direction mask: forward=2, backward=1
    constexpr uint32_t my_direction_mask = direction ? 2 : 1;

    ///////////////////////////////////////////////////
    // SEMAPHORE SETUP
    ///////////////////////////////////////////////////

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint32_t sem_count = 0;
    uint32_t chunk_count = 0;

    ///////////////////////////////////////////////////
    // MAIN LOOP
    ///////////////////////////////////////////////////

    for (uint32_t batch = 0; batch < input_tensor_B; ++batch) {
        for (uint32_t step = 0; step < num_steps; ++step) {
            // Double-buffer indices
            uint32_t recv_buffer_idx = step % 2;
            uint32_t accum_buffer_idx = (step > 0) ? ((step - 1) % 2) : 0;

            for (uint32_t slice = 0; slice < ring_size; ++slice) {
                int32_t offset = compute_offset(my_chip_id, slice, ring_size);
                uint32_t recv_dir = get_receive_direction(offset, step, num_steps, ring_size);

                // Only process if this slice's receive direction matches our worker direction
                if ((recv_dir & my_direction_mask) == 0) {
                    continue;
                }

                // Process tiles for this slice within our assigned range
                for (uint32_t c = 0; c < num_channel_iters; ++c) {
                    // Compute base tile IDs
                    uint32_t input_tile_base =
                        get_slice_start_tile_id(slice, batch, c, batch_num_pages, channel_num_pages);
                    uint32_t channel_tile_base = c * tiles_per_channel;

                    // Process only tiles within our assigned range [start_tile_offset, end_tile_offset)
                    uint32_t tile_start = (c == 0) ? start_tile_offset : 0;
                    uint32_t tile_end = (c == num_channel_iters - 1) ? end_tile_offset : tiles_per_channel;

                    // Adjust for multi-channel case: work splitting is across total tiles in slice
                    if constexpr (dim != 1) {
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

                        // Wait for semaphore based on chunks_per_sync
                        chunk_count++;
                        if (chunk_count % chunks_per_sync == 0 || tile_offset == tile_start) {
                            sem_count++;
                            noc_semaphore_wait_min(sem_ptr, sem_count);
                        }

                        // 1. Read RECEIVED data from intermediate buffer into cb_intermediate
                        cb_reserve_back(cb_intermediate_id, tiles_this_chunk);
                        uint32_t recv_l1_addr = get_write_ptr(cb_intermediate_id);
                        for (uint32_t t = 0; t < tiles_this_chunk; ++t) {
                            uint32_t intermediate_tile_id =
                                get_intermediate_tile_id(recv_buffer_idx, slice, channel_tile_base + tile_offset + t);
                            uint64_t noc_addr = get_noc_addr(intermediate_tile_id, intermediate_addrgen);
                            noc_async_read(noc_addr, recv_l1_addr, page_size);
                            recv_l1_addr += page_size;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_intermediate_id, tiles_this_chunk);

                        // 2. Read LOCAL/ACCUMULATED data into cb_input
                        cb_reserve_back(cb_input_id, tiles_this_chunk);
                        uint32_t local_l1_addr = get_write_ptr(cb_input_id);
                        for (uint32_t t = 0; t < tiles_this_chunk; ++t) {
                            uint64_t noc_addr;
                            if (step == 0) {
                                // Step 0: read from input tensor
                                uint32_t input_tile_id = input_tile_base + tile_offset + t;
                                noc_addr = get_noc_addr(input_tile_id, input_addrgen);
                            } else {
                                // Step 1+: read from accumulated intermediate
                                uint32_t accum_tile_id = get_intermediate_tile_id(
                                    accum_buffer_idx, slice, channel_tile_base + tile_offset + t);
                                noc_addr = get_noc_addr(accum_tile_id, intermediate_addrgen);
                            }
                            noc_async_read(noc_addr, local_l1_addr, page_size);
                            local_l1_addr += page_size;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_input_id, tiles_this_chunk);
                    }
                }
            }
        }
    }
}
