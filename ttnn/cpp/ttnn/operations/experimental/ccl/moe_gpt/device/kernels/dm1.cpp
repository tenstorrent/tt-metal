// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_gpt_ring_common.h"

#include "api/debug/dprint_pages.h"
void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t enable_dram_output = get_named_compile_time_arg_val("enable_dram_output");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_w1_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;

    // CB Aliases
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;
    constexpr auto cb_c2s_out = tt::CBIndex::c_1;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2);

    // Constants for MoEGPT
    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_gpt_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_gpt_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];  // 7 or 8
    const uint32_t num_w2_tiles_w = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];                    // 7 or 8

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    // The number of times to repeat the all2all
    constexpr uint32_t num_a2a_iters = moe_gpt_ring::NUM_A2A_ITERS_A;  // 2

    // The number of steps to take in the all2all is the number of cores
    constexpr uint32_t num_a2a_steps_per_iter = moe_gpt_ring::NUM_CORES;  // 12

    // The number of tiles to send in each step (max of 7/8 = 8 for GPT-OSS)
    constexpr uint32_t tiles_per_step = moe_gpt_ring::IN2_TILES_PER_STEP_A;  // 8

    //-------------------------------------------------------------------------
    // Ring NoC setup
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(ring_semaphore_id);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    const uint64_t neighbor_semaphore_noc_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, semaphore_addr);

    // Size of each transfer in bytes: 8 tiles * 2048 = 16384 bytes
    constexpr uint32_t a2a_xfer_bytes_per_step = tiles_per_step * in2_tile_size;

    // NOC_MAX_BURST_SIZE = 8192 bytes on Wormhole. Split into packets of 4 tiles each
    // (4 * 2048 = 8192 bytes = NOC_MAX_BURST_SIZE). 8 tiles / 4 tiles per packet = 2 packets.
    constexpr uint32_t a2a_tiles_per_packet = 4;
    constexpr uint32_t a2a_packet_size = a2a_tiles_per_packet * in2_tile_size;   // 8192
    constexpr uint32_t a2a_num_packets = tiles_per_step / a2a_tiles_per_packet;  // 2

    // Source and destination addresses for the all2all
    const uint32_t local_base_addr = get_write_ptr(cb_s2c_in2);
    const uint64_t neighbor_base_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, local_base_addr);

    // 6 buffers for A2A intermediate data. 6 divides 12 (steps per iter),
    // so buffer indices cycle back to 0 after a full ring rotation.
    // 5 steps of slack between write and overwrite is sufficient for compute.
    constexpr uint32_t NUM_A2A_BUFFERS = 12;
    uint32_t LOCAL_BUFFER_OFFSET[NUM_A2A_BUFFERS];
    for (uint32_t i = 0; i < NUM_A2A_BUFFERS; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_base_addr + i * a2a_xfer_bytes_per_step;
    }
    // Reset semaphore to 0 for clean state on each invocation.
    // When the program is cached and reused, the semaphore retains its value
    // from the previous run. Resetting here is safe because:
    // 1. The command queue guarantees all NOC activity from the previous
    //    invocation has completed before this kernel starts.
    // 2. All cores are waiting for cb_c2w_rdy (compute's SwiGLU) before
    //    any ring activity begins, so no predecessor can write to our
    //    semaphore before we reset it.
    noc_semaphore_set(my_semaphore_ptr, 0);
    uint32_t semaphore_value = 0;

    // Set state for the data writes
    noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_base_addr, a2a_packet_size, /*noc=*/1, vchannel);

    // Set state for the semaphore write
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/false>(
        neighbor_semaphore_noc_addr, /*val=*/0, /*be=*/0xF, /*cmd_buf=*/write_at_cmd_buf, /*noc=*/1, vchannel);

    // tt::data_movement::common::print_bf16_pages(get_read_ptr(cb_s2c_in), 1024, 90);

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Wait for compute core to tell us that all mm01 data is ready
        cb_wait_front(cb_c2w_rdy, 1);
        cb_pop_front(cb_c2w_rdy, 1);

        // Take the data in cb_s2c_in2 and send it to the next core in the ring
        // GPT-OSS: 2 A2A iterations x 12 steps = 24 rotations per expert
        for (uint32_t i = 0; i < num_a2a_iters; ++i) {
            for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                // Wait for current data to be ready in cb_s2c_in2
                while ((*my_semaphore_ptr) < semaphore_value) {
                };

                // Signal to compute core that data is ready
                cb_reserve_back(cb_w2c_rdy, 1);
                cb_push_back(cb_w2c_rdy, 1);

                // Write 8 tiles from local cb_s2c_in2 to neighbor's cb_s2c_in2
                // Buffer index cycles with modulo, resetting at each iteration.
                const uint32_t src_buf = step % NUM_A2A_BUFFERS;
                const uint32_t dst_buf = (step + 1) % NUM_A2A_BUFFERS;
                const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[src_buf];
                const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[dst_buf];

                // Send as 2 packets of 4 tiles each (8192 bytes per packet)
                for (uint32_t pkt = 0; pkt < a2a_num_packets; ++pkt) {
                    uint32_t pkt_offset = pkt * a2a_packet_size;
                    noc_async_write_one_packet_with_state</*posted=*/true>(
                        local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
                }

                // Ensure all data packets are in the NOC before signaling readiness.
                // Data writes (write_at_cmd_buf) and semaphore writes (write_at_cmd_buf)
                // share the same command buffer, but without this flush the semaphore
                // increment can overtake in-flight data packets at the destination.
                noc_async_posted_writes_flushed();

                // Signal neighbor that data is ready (increment their semaphore value)
                noc_inline_dw_write_with_state<
                    /*update_addr_lo=*/false,
                    /*update_counter=*/true,
                    /*posted=*/true,
                    /*update_addr_hi=*/false,
                    /*update_val=*/true>(++semaphore_value);

                // Ensure writes have left the core before continuing
                noc_async_posted_writes_flushed();
            }
        }

        // Cross-expert boundary barrier: after the ring loop, wait for the
        // predecessor's FINAL write to our buf 0 before letting compute
        // overwrite buf 0 with the next expert's SwiGLU.
        //
        // The predecessor's last A2A step (step 11, iter 1) writes to our
        // dst_buf = (11+1)%12 = 0. The ring loop only waited for semaphore
        // value (semaphore_value - 1) which confirms step 10's data, not
        // step 11's. Here we wait for semaphore_value (set by ++semaphore_value
        // on the last iteration), confirming the predecessor's step 11 write
        // has landed at our L1.
        if (expert_id < num_experts - 1) {
            while ((*my_semaphore_ptr) < semaphore_value) {
            };
            // Signal compute that buf 0 is safe to overwrite
            cb_reserve_back(cb_w2c_rdy, 1);
            cb_push_back(cb_w2c_rdy, 1);
        }
    }

    //-------------------------------------------------------------------------
    // DRAM output write phase
    //-------------------------------------------------------------------------
    // After all experts are done, copy output tiles from L1 sharded buffer to
    // a contiguous interleaved DRAM tensor with shape (E, 1, M, K) in TILE_LAYOUT.
    // Each core owns tiles_per_core tiles (7 or 8) of the K dimension per expert.
    if constexpr (enable_dram_output) {
        // Read DRAM output runtime args
        const auto dram_output_addr = get_arg_val<uint32_t>(argidx++);
        const auto k_start_tile = get_arg_val<uint32_t>(argidx++);

        // K dimension in tiles (K=2880, 2880/32=90)
        constexpr uint32_t K_tiles = moe_gpt_ring::NUM_W0_W1_TILES_H;  // 90

        // tiles_per_core for this core (7 or 8)
        const uint32_t tiles_per_core = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];

        // Create address generator for the interleaved DRAM output
        // Uses InterleavedAddrGen which computes bank-interleaved NOC addresses
        const InterleavedAddrGen<true> dram_output_addrgen = {
            .bank_base_address = dram_output_addr,
            .page_size = in_tile_size,
        };

        // L1 source: cb_c2s_out base address (same buffer as input, output written in-place)
        const uint32_t cb_base = get_write_ptr(cb_c2s_out);

        // For each expert, write this core's tiles to the DRAM output tensor
        for (uint32_t e = 0; e < num_experts; ++e) {
            for (uint32_t t = 0; t < tiles_per_core; ++t) {
                // L1 source: expert e's section starts at tile (e * K_tiles),
                // and this core's valid output is at tiles 0..tiles_per_core-1
                uint32_t l1_addr = cb_base + (e * K_tiles + t) * in_tile_size;

                // DRAM destination: linear tile index in (E, 1, M, K) tensor
                // Expert e occupies tiles [e*K_tiles .. (e+1)*K_tiles-1]
                uint32_t dram_tile_id = e * K_tiles + k_start_tile + t;

                noc_async_write_tile(dram_tile_id, dram_output_addrgen, l1_addr);
            }
        }
        noc_async_write_barrier();
    }
}
