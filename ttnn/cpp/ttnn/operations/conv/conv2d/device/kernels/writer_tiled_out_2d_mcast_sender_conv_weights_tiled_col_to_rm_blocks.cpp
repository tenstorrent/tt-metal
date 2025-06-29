// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include "dataflow_api.h"

#define ENABLE_DEBUG 1

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t bias_in_dram = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(8);
    constexpr uint32_t weight_block_height_ntiles = get_compile_time_arg_val(9);
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(10);
    constexpr uint32_t weight_stride_h = get_compile_time_arg_val(11);
    // constexpr uint32_t weight_next_block_stride_h = get_compile_time_arg_val(12);
    // constexpr uint32_t weight_next_block_stride_w = get_compile_time_arg_val(13);

    // Bias arg. Unused if bias fusion is not enabled.
    // constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);

    // constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);
    // constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(16);

    uint32_t i = 0;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i++);
    // Bias arg. Unused if bias fusion is not enabled.
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);
    const uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i++);
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t noop = get_arg_val<uint32_t>(i++);
    if (noop) {
        return;
    }

    // mcast args
    const uint32_t weights_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_dests = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_cores = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    *(weights_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* weights_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_sender_semaphore_addr);

    const uint64_t weights_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        weights_mcast_dest_noc_start_x,
        weights_mcast_dest_noc_start_y,
        weights_mcast_dest_noc_end_x,
        weights_mcast_dest_noc_end_y,
        weights_mcast_receiver_semaphore_addr);
#endif

// read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS
    constexpr uint32_t bias_pagesize = get_tile_size(bias_cb_id);
    constexpr DataFormat bias_df = get_dataformat(bias_cb_id);
    const InterleavedAddrGenFast<bias_in_dram> s_bias = {
        .bank_base_address = bias_addr, .page_size = bias_pagesize, .data_format = bias_df};

    bool load_bias = true;
#endif

    DPRINT << "weight_block_num_tiles: " << weight_block_num_tiles << ENDL();
    // DPRINT << "out_num_blocks_w: " << out_num_blocks_w << ENDL();
    // DPRINT << "out_num_blocks_h: " << out_num_blocks_h << ENDL();
    DPRINT << "weight_stride_h: " << weight_stride_h << ENDL();

    DPRINT << "num_blocks_weight_h: " << num_blocks_weight_h << ENDL();
    DPRINT << "weight_block_height_ntiles: " << weight_block_height_ntiles << ENDL();
    DPRINT << "weight_block_width_ntiles: " << weight_block_width_ntiles << ENDL();
    DPRINT << "weight_block_height_num_outer: " << weight_block_height_num_outer << ENDL();
    // DPRINT << "tiles to read in one block: "
    //        << num_blocks_weight_h * weight_block_height_ntiles * weight_block_width_ntiles << ENDL();

    constexpr uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    constexpr DataFormat weight_df = get_dataformat(cb_id_weight);
    const InterleavedAddrGenFast<true> s_weight = {
        .bank_base_address = weight_addr_dram_base, .page_size = weight_tile_nbytes, .data_format = weight_df};
    constexpr uint32_t weights_block_size_bytes = weight_tile_nbytes * weight_block_num_tiles;

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t weight_current_block_start_tile_id = out_start_tile_id_w;
    for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
         weight_tile_h_outer_i++) {
        for (uint32_t height_block_index = 0; height_block_index < num_blocks_weight_h; height_block_index++) {
            cb_reserve_back(cb_id_weight, weight_block_num_tiles);
            uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);

            const uint32_t weights_start_address = weight_write_l1_addr;

            for (uint32_t block_weight_h = 0; block_weight_h < weight_block_height_ntiles; block_weight_h++) {
                // DPRINT << "weight_tile_id: " << weight_current_block_start_tile_id << ENDL();
                noc_async_read_tile(weight_current_block_start_tile_id, s_weight, weight_write_l1_addr);
                weight_write_l1_addr += weight_tile_nbytes;
                weight_current_block_start_tile_id += weight_stride_h;
            }

            noc_async_read_barrier();
            // tt::data_movement::common::print_bf16_pages(weights_start_address, 1024, weight_block_num_tiles);

            // mcast args
#ifndef SKIP_MCAST
            // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
            // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to
            // zero for the next block
            noc_semaphore_wait(weights_mcast_sender_semaphore_addr_ptr, weights_mcast_num_dests);
            noc_semaphore_set(weights_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t weights_multicast_data_addr = get_noc_multicast_addr(
                weights_mcast_dest_noc_start_x,
                weights_mcast_dest_noc_start_y,
                weights_mcast_dest_noc_end_x,
                weights_mcast_dest_noc_end_y,
                weights_start_address);
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                weights_start_address,
                weights_multicast_data_addr,
                weights_block_size_bytes,
                weights_mcast_num_cores,
                true);

            // We should also multicast the flag to destinations
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_semaphore_set_multicast(
                weights_mcast_receiver_semaphore_addr,
                weights_mcast_receiver_semaphore_noc_addr,
                weights_mcast_num_cores);
#endif
            cb_push_back(cb_id_weight, weight_block_num_tiles);
        }  // for weight_block_height_num_outer
    }

    noc_async_write_barrier();
}
