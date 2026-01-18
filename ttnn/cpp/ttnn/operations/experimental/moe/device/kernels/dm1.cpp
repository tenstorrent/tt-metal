// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_w1_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2c_mm0 = tt::CBIndex::c_2;
    constexpr auto cb_c2c_mm1 = tt::CBIndex::c_3;
    constexpr auto cb_c2w_elt = tt::CBIndex::c_4;
    constexpr auto cb_r2c_in2 = tt::CBIndex::c_5;
    constexpr auto cb_c2w_mm2 = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w1 = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_tile_size = get_tile_size(cb_r2c_w0);
    constexpr uint32_t w1_tile_size = get_tile_size(cb_r2c_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t out_tile_size = get_tile_size(cb_c2w_elt);

    // Tensor accessors
    const auto in_accessor = TensorAccessor(in_args, in_addr, in_tile_size);
    const auto w0_w1_accessor = TensorAccessor(w0_w1_args, w0_w1_addr, w0_tile_size);
    const auto w2_accessor = TensorAccessor(w2_args, w2_addr, w2_tile_size);
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_tile_size);

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = 224;
    constexpr uint32_t num_w2_tiles_h = 64;

    const uint32_t num_w0_w1_tiles_w = (core_id < 8) ? 5 : 6;
    const uint32_t num_w2_tiles_w = (core_id < 8) ? 18 : 20;

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    // Ring synchronization setup
    constexpr uint32_t NUM_CORES = 12;  // Total cores in the ring
    const uint32_t semaphore_addr = get_semaphore(ring_semaphore_id);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    const uint64_t neighbor_semaphore_noc_addr = get_noc_addr(neighbor_physical_x, neighbor_physical_y, semaphore_addr);

    // Ring data transfer setup
    constexpr uint32_t TILES_PER_CORE = 6;  // Always send 6 tiles, even if only 5 have valid data
    constexpr uint32_t in2_tile_size = get_tile_size(cb_r2c_in2);
    constexpr uint32_t tiles_transfer_size = TILES_PER_CORE * in2_tile_size;
    const uint32_t local_cb_in2_addr = get_write_ptr(cb_r2c_in2);  // Local CB base address
    const uint64_t neighbor_cb_in2_base_addr =
        get_noc_addr(neighbor_physical_x, neighbor_physical_y, local_cb_in2_addr);

    // All cores must do the same number of ring sync iterations to avoid deadlock
    // Cores 0-7 have 9 iterations of real work, cores 8-11 have 10
    constexpr uint32_t MAX_MM2_ITERS = 20 / 2;
    const uint32_t my_mm2_iters = num_mm2_tiles / 2;  // 9 for cores 0-7, 10 for cores 8-11

    uint32_t semaphore_value = 0;

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Write from cb_c2w_elt
        for (uint32_t i = 0; i < num_elt_tiles; ++i) {
            cb_wait_front(cb_c2w_elt, 1);
            cb_pop_front(cb_c2w_elt, 1);
        }

        // Read to cb_r2c_in2
        for (uint32_t i = 0; i < num_w2_tiles_h; ++i) {
            // cb_reserve_back(cb_r2c_in2, 1);
            // cb_push_back(cb_r2c_in2, 1);
        }

        // Write from cb_c2w_mm2
        // All cores iterate MAX_MM2_ITERS times for ring sync, but only do CB work if i < my_mm2_iters
        for (uint32_t i = 0; i < MAX_MM2_ITERS; ++i) {
            // Only do CB work if this core has work for this iteration
            if (i < my_mm2_iters) {
                cb_wait_front(cb_c2w_mm2, 2);
                cb_pop_front(cb_c2w_mm2, 2);
            }

            // Ring synchronization: all cores participate regardless of whether they had CB work
            // With 12 cores in a ring, we perform 12 steps so the signal propagates around the entire ring
            for (uint32_t step = 0; step < NUM_CORES; ++step) {
                // Double buffer offset: alternate between buffer 0 and buffer 1 based on step parity
                const uint32_t buffer_offset = (step & 1) * tiles_transfer_size;

                // Write 6 tiles from local cb_r2c_in2 to neighbor's cb_r2c_in2
                // Split into 8KB + 4KB packets (NOC_MAX_BURST_SIZE=8KB)
                // Using noc_async_write_one_packet with posted=true for better performance
                const uint32_t local_src_addr = local_cb_in2_addr + buffer_offset;
                const uint64_t neighbor_dst_addr = neighbor_cb_in2_base_addr + buffer_offset;
                constexpr uint32_t first_packet_size = NOC_MAX_BURST_SIZE;                        // 8KB (4 tiles)
                constexpr uint32_t second_packet_size = tiles_transfer_size - first_packet_size;  // 4KB (2 tiles)

                // First packet: 8KB
                noc_async_write_one_packet<true, /*posted=*/true>(local_src_addr, neighbor_dst_addr, first_packet_size);

                // Second packet: 4KB
                noc_async_write_one_packet<true, /*posted=*/true>(
                    local_src_addr + first_packet_size, neighbor_dst_addr + first_packet_size, second_packet_size);

                // Signal neighbor that data is ready (increment their semaphore)
                // Use posted=true for fire-and-forget (no acknowledgment needed)
                noc_semaphore_inc</*posted=*/true>(neighbor_semaphore_noc_addr, 1);

                // Ensure semaphore increment is issued to the NOC
                noc_async_posted_atomic_barrier();

                // Wait for my predecessor to have signaled me (wait for semaphore to reach step + 1)
                noc_semaphore_wait_min(my_semaphore_ptr, ++semaphore_value);
            }
        }
    }
}
