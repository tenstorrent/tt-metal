// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#define ENABLE_DEBUG 0

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
    constexpr uint32_t weight_next_block_stride_h = get_compile_time_arg_val(12);
    constexpr uint32_t weight_next_block_stride_w = get_compile_time_arg_val(13);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);

    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(16);

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

    constexpr uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    constexpr DataFormat weight_df = get_dataformat(cb_id_weight);
    const InterleavedAddrGenFast<true> s_weight = {
        .bank_base_address = weight_addr_dram_base, .page_size = weight_tile_nbytes, .data_format = weight_df};

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t weight_start_tile_id = out_start_tile_id_w;
    uint32_t weight_inner_block_stride_h =
        weight_next_block_stride_h / weight_block_height_num_outer;  // TODO: Pass as args
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            // READ WEIGHTS + MCAST SEND WEIGHTS
            // read weight blocks inner dim
            // read weight slice - 1 block of weights in width dim and full weight matrix height
            // read slice only once for all activation blocks
            uint32_t weight_current_block_start_tile_id = weight_start_tile_id;
            for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                 weight_tile_h_outer_i++) {
                cb_reserve_back(cb_id_weight, weight_block_num_tiles);
                uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);

                // mcast args
                uint32_t weights_start_address = weight_write_l1_addr;
                uint32_t weights_block_size_bytes = 0;
                // loop over weight block tiles along h
                // num_blocks_weight_h * weight_block_height_ntiles
                // weight_stride_h
                for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h * weight_block_height_ntiles;
                     block_weight_h++) {
                    // mcast args
                    // uint32_t weights_start_address = weight_write_l1_addr;
                    // uint32_t weights_block_size_bytes = 0;

                    uint32_t weight_tile_id = weight_current_block_start_tile_id;
                    // loop over weight block tiles along w
                    for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                        noc_async_read_tile(weight_tile_id, s_weight, weight_write_l1_addr);
                        weight_write_l1_addr += weight_tile_nbytes;
                        weights_block_size_bytes += weight_tile_nbytes;
                        weight_tile_id += 1;
                    }  // for weight_block_w
                    weight_current_block_start_tile_id += weight_stride_h;
                }

                noc_async_read_barrier();

#ifndef SKIP_MCAST
                // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
                // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to zero
                // for the next block
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

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc
                // even though cmd bufs are different Also, this only works because we are setting VCs statically (using
                // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may not
                // be sent in order they are issued
                noc_async_writes_flushed();
#endif
                // We should also multicast the flag to destinations
                // num_dests must not include source, since we are NOT really doing a local copy!
                noc_semaphore_set_multicast(
                    weights_mcast_receiver_semaphore_addr,
                    weights_mcast_receiver_semaphore_noc_addr,
                    weights_mcast_num_cores);
#endif
                cb_push_back(cb_id_weight, weight_block_num_tiles);
            }  // for weight_block_height_num_outer

#ifdef FUSE_BIAS
            if (load_bias) {
                cb_reserve_back(bias_cb_id, bias_ntiles);
                uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);

                // mcast args
                uint32_t bias_start_address = bias_l1_addr;
                uint32_t bias_block_size_bytes = 0;
                for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles; ++bias_tile) {
                    noc_async_read_tile(bias_tile, s_bias, bias_l1_addr);
                    bias_l1_addr += bias_pagesize;
                    bias_block_size_bytes += bias_pagesize;
                }
                noc_async_read_barrier();

// MCAST BIAS (shares some mcast args with weights)
#ifndef SKIP_MCAST
                // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
                // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to zero
                // for the next block
                noc_semaphore_wait(weights_mcast_sender_semaphore_addr_ptr, weights_mcast_num_dests);
                noc_semaphore_set(weights_mcast_sender_semaphore_addr_ptr, 0);

                // Now we have the block in the CB address, we can mcast to dests!
                uint64_t bias_multicast_data_addr = get_noc_multicast_addr(
                    weights_mcast_dest_noc_start_x,
                    weights_mcast_dest_noc_start_y,
                    weights_mcast_dest_noc_end_x,
                    weights_mcast_dest_noc_end_y,
                    bias_start_address);
                // num_dests must not include source, since we are NOT really doing a local copy!
                noc_async_write_multicast(
                    bias_start_address, bias_multicast_data_addr, bias_block_size_bytes, weights_mcast_num_cores, true);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc
                // even though cmd bufs are different Also, this only works because we are setting VCs statically (using
                // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may not
                // be sent in order they are issued
                noc_async_writes_flushed();
#endif
                // We should also multicast the flag to destinations
                // num_dests must not include source, since we are NOT really doing a local copy!
                noc_semaphore_set_multicast(
                    weights_mcast_receiver_semaphore_addr,
                    weights_mcast_receiver_semaphore_noc_addr,
                    weights_mcast_num_cores);
#endif

                cb_push_back(bias_cb_id, bias_ntiles);
                load_bias = false;
            }
#endif

        }  // out_num_blocks_h

        // Increment weight start tile id for next block in width dim
        weight_start_tile_id += weight_next_block_stride_w;
    }  // out_num_blocks_w

    noc_async_write_barrier();
}
