// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp
// (height-sharded conv2d weights mcast sender; also reads the activation second-reader half on
// split-reader paths).
//
// Despite the filename this kernel never writes the OUT buffer: it reads + multicasts weights (and
// bias) and, when split reader is on, reads the second-reader half of the activation. In the Metal
// 2.0 factory OUT is bound to this kernel as a DEGENERATE CONSUMER (resolution #1); there is no
// out-CB code in the body.
//
// Algorithm body identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs -> dfb:: tokens (weights / bias / act_second_reader / act_sharded / reader_indices)
//   - weight/bias TensorAccessorArgs + base-address RTAs -> tensor::weights / tensor::bias bindings
//   - weights-mcast semaphore RTAs -> Semaphore(sem::weights_mcast_sender / weights_mcast_receiver)
//   - remaining positional CTAs -> get_arg(args::name); remaining RTAs -> get_arg(args::name)
//   - DataflowBuffer -> DataflowBuffer (objects passed to conv_reader_common.hpp helpers stay
//     experimental::CB); get_tile_size(cb) -> cb.get_entry_size()
//   - dfb::bias / tensor::bias gated behind FUSE_BIAS; dfb::act_second_reader gated behind SPLIT_READER

#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"

void kernel_main() {
    constexpr uint32_t num_blocks_weight_h = get_arg(args::num_blocks_weight_h);
    constexpr uint32_t weight_block_num_tiles = get_arg(args::weight_block_num_tiles);

    constexpr uint32_t weight_block_height_num_outer = get_arg(args::weight_block_height_num_outer);
    constexpr uint32_t weight_block_height_ntiles = get_arg(args::weight_block_height_ntiles);
    constexpr uint32_t weight_block_width_ntiles = get_arg(args::weight_block_width_ntiles);
    constexpr uint32_t weight_stride_h = get_arg(args::weight_stride_h);
    constexpr uint32_t weight_next_block_stride_h = get_arg(args::weight_next_block_stride_h);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_arg(args::bias_ntiles);

    constexpr uint32_t out_num_blocks_h = get_arg(args::out_num_blocks_h);

    constexpr bool fuse_bias = get_arg(args::fuse_bias);

    constexpr bool split_reader_enabled = get_arg(args::split_reader_enabled);
    constexpr bool activation_reuse_enabled = get_arg(args::activation_reuse_enabled);

    // Split reader args
    constexpr uint32_t act_block_num_tiles = get_arg(args::act_block_num_tiles);
    constexpr uint32_t conv_act_c_read_bytes = get_arg(args::conv_act_c_read_bytes);
    constexpr uint32_t weight_size_w = get_arg(args::weight_size_w);
    constexpr uint32_t conv_act_size_w_padded = get_arg(args::conv_act_size_w_padded);
    constexpr uint32_t act_block_w_extra_align_bytes = get_arg(args::act_block_w_extra_align_bytes);
    constexpr bool needs_act_block_zero_out = get_arg(args::needs_act_block_zero_out) == 1;
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t weights_size_h = get_arg(args::weights_size_h);

    // Activation reuse args
    constexpr uint32_t act_reuse_cb_tiles = get_arg(args::act_reuse_cb_tiles);
    constexpr uint32_t act_block_w_tiles = get_arg(args::act_block_w_tiles);
    constexpr bool readers_process_full_image_widths = get_arg(args::readers_process_full_image_widths) == 1;
    constexpr uint32_t image_width_tiles = get_arg(args::image_width_tiles);
    constexpr uint32_t output_image_width = get_arg(args::output_image_width);
    constexpr uint32_t window_reuse_offset = get_arg(args::window_reuse_offset);
    constexpr bool need_to_push_remaining_tiles = get_arg(args::need_to_push_remaining_tiles) == 1;
    constexpr bool single_core_processes_multiple_batches = get_arg(args::single_core_processes_multiple_batches) == 1;

    [[maybe_unused]] const uint32_t out_start_tile_id_w = get_arg(args::out_start_tile_id_w);
#ifdef FUSE_BIAS
    const uint32_t bias_tile_offset = get_arg(args::bias_tile_offset);
#endif

    // Experimental API objects
    Noc noc;

#ifdef SPLIT_READER
    DataflowBuffer cb_act_second_obj(dfb::act_second_reader);
    if constexpr (split_reader_enabled) {
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<dfb::act_second_reader>(noc, cb_act_second_obj);
        }
    }
#endif

    // mcast args
    const McastRect mcast_rect = {
        get_arg(args::mcast_dest_noc_start_x),
        get_arg(args::mcast_dest_noc_start_y),
        get_arg(args::mcast_dest_noc_end_x),
        get_arg(args::mcast_dest_noc_end_y)};
    const uint32_t weights_mcast_num_dests = get_arg(args::weights_mcast_num_dests);
    const uint32_t weights_mcast_num_cores = get_arg(args::weights_mcast_num_cores);
    Semaphore<> weights_mcast_sender_sem(sem::weights_mcast_sender);
    Semaphore<> weights_mcast_receiver_sem(sem::weights_mcast_receiver);
    MulticastEndpoint mcast_ep;
    DataflowBuffer cb_weight_obj(dfb::weights);
#ifdef SPLIT_READER
    DataflowBuffer cb_reader_indices_obj(dfb::reader_indices);
    DataflowBuffer cb_sharded_act_obj(dfb::act_sharded);
#endif
#ifdef FUSE_BIAS
    DataflowBuffer cb_bias_obj(dfb::bias);
#endif
    // Pre-built mcast destination; .addr is updated per mcast call
    McastDst mcast_dst = {
        .noc_x_start = mcast_rect.noc_x_start,
        .noc_y_start = mcast_rect.noc_y_start,
        .noc_x_end = mcast_rect.noc_x_end,
        .noc_y_end = mcast_rect.noc_y_end,
        .addr = 0};

#ifdef SPLIT_READER
    const uint32_t remaining_tiles_to_push =
        split_reader_enabled && activation_reuse_enabled ? get_arg(args::remaining_tiles_to_push) : 0;

    // Split reader configuration
    if constexpr (split_reader_enabled) {
#ifdef CONFIG_TENSOR_IN_DRAM
        cb_reader_indices_obj.wait_front(1);
#endif
    }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        split_reader_enabled ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_indices_obj.get_write_ptr())
                             : nullptr;
    uint32_t reader_idx = 0;
    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    uint32_t act_l1_read_addr = split_reader_enabled ? cb_sharded_act_obj.get_read_ptr() : 0;
    // coalesce reads along weight_size_w
    uint32_t start_reader_idx = split_reader_enabled ? (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1 : 0;
    uint32_t cb_start_addr = split_reader_enabled ? cb_act_second_obj.get_write_ptr() : 0;
    uint32_t reader_offset = act_l1_read_addr;
#endif

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    weights_mcast_receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
#endif

    // read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS
    const uint32_t bias_pagesize =
        fuse_bias ? cb_bias_obj.get_entry_size() : 0;  // dummy but valid value in case bias is not enabled
    const auto s_bias = TensorAccessor(tensor::bias);
    bool load_bias = true;
#endif

    const uint32_t weight_tile_nbytes = cb_weight_obj.get_entry_size();
    const auto s_weight = TensorAccessor(tensor::weights);

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    constexpr uint32_t weight_inner_block_stride_h =
        weight_next_block_stride_h / weight_block_height_num_outer;  // TODO: Pass as args

    [[maybe_unused]] uint32_t l1_write_addr_act = 0;
    for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        // READ WEIGHTS + MCAST SEND WEIGHTS
        // read weight blocks inner dim
        // read weight slice - 1 block of weights in width dim and full weight matrix height
        // read slice only once for all activation blocks
        uint32_t weight_h_offset = 0;

        uint32_t weight_current_block_start_tile_id = 0;

#ifdef SPLIT_READER
        if constexpr (split_reader_enabled) {
            if constexpr (activation_reuse_enabled) {
                l1_write_addr_act = cb_start_addr;
                cb_act_second_obj.evil_set_write_ptr(l1_write_addr_act);
            }
            reader_offset = act_l1_read_addr;
        }
#endif

        for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
#ifdef SPLIT_READER
            if constexpr (split_reader_enabled) {
                // Do the second half of the reads for act
                experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);
                reader_idx = start_reader_idx;

                if constexpr (!activation_reuse_enabled) {
                    cb_act_second_obj.reserve_back(act_block_num_tiles);
                    l1_write_addr_act = cb_act_second_obj.get_write_ptr();
                    read_sticks<
                        dilation_w,
                        coalesced_read_bytes,
                        conv_act_c_read_bytes,
                        act_block_w_extra_align_bytes,
                        stride_w_bytes,
                        weight_size_w,
                        stride_w>(noc, packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);
                    noc.async_read_barrier();
                    cb_act_second_obj.push_back(act_block_num_tiles);

                    reader_offset += window_outer_offset;
                } else {
                    read_sticks_activation_reuse<
                        coalesced_read_bytes,
                        conv_act_c_read_bytes,
                        act_block_w_extra_align_bytes,
                        window_outer_offset,
                        weight_size_w,
                        stride_w,
                        weights_size_h,
                        dfb::act_second_reader,
                        act_reuse_cb_tiles,
                        act_block_w_tiles,
                        readers_process_full_image_widths,
                        image_width_tiles,
                        output_image_width,
                        window_reuse_offset,
                        single_core_processes_multiple_batches>(
                        noc,
                        cb_act_second_obj,
                        packed_reader_indices_ptr,
                        act_l1_read_addr,
                        l1_write_addr_act,
                        reader_idx,
                        cb_start_addr);

                    if constexpr (need_to_push_remaining_tiles) {
                        if (block_weight_h == num_blocks_weight_h - 1) {
                            // Last core sometimes has less work to do, but we still need to push the same number of
                            // tiles to avoid blocking compute kernels
                            push_remaining_tiles<dfb::act_second_reader, act_block_w_tiles, image_width_tiles>(
                                cb_act_second_obj, remaining_tiles_to_push, cb_start_addr);
                        }
                    }
                }
            }
#endif

            // Do weights read + mcast
            cb_weight_obj.reserve_back(weight_block_num_tiles);
            if (bh == 0) {
                uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id + weight_h_offset;

                uint32_t weight_write_offset = 0;
                uint32_t weights_block_size_bytes = 0;

                // loop over weight block tiles along h
                for (uint32_t weight_tile_h_i = 0; weight_tile_h_i < weight_block_height_ntiles; ++weight_tile_h_i) {
                    uint32_t weight_tile_id = weight_row_start_tile_id;
                    // loop over weight block tiles along w
                    for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                        // DPRINT("weight_tile_id={}\n", weight_tile_id);
                        noc.async_read(
                            s_weight,
                            cb_weight_obj,
                            weight_tile_nbytes,
                            {.page_id = weight_tile_id},
                            {.offset_bytes = weight_write_offset});
                        weight_write_offset += weight_tile_nbytes;
                        weights_block_size_bytes += weight_tile_nbytes;
                        weight_tile_id += 1;
                    }  // for weight_block_w
                    weight_row_start_tile_id += weight_stride_h;
                }  // for weight_block_h
                noc.async_read_barrier();

#ifndef SKIP_MCAST
                // wait until all weights mcast destinations have atomically incremented the weights
                // semaphore_addr (i.e. its value should be weights_mcast_num_dests), then reset the
                // semaphore_addr value back to zero for the next block
                weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                weights_mcast_sender_sem.set(0);

                // Now we have the block in the CB address, we can mcast to dests!
                // num_dests must not include source, since we are NOT really doing a local copy!
                mcast_dst.addr = cb_weight_obj.get_write_ptr();
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(cb_weight_obj.get_write_ptr()),
                    mcast_ep,
                    weights_block_size_bytes,
                    weights_mcast_num_cores,
                    {},
                    mcast_dst,
                    true);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                // statically (using NOC_CMD_STATIC_VC).
                // We should also multicast the flag to destinations
                // num_dests must not include source, since we are NOT really doing a local copy!
                weights_mcast_receiver_sem.set_multicast(
                    noc,
                    mcast_rect.noc_x_start,
                    mcast_rect.noc_y_start,
                    mcast_rect.noc_x_end,
                    mcast_rect.noc_y_end,
                    weights_mcast_num_cores,
                    false);
#endif

                weight_current_block_start_tile_id += weight_next_block_stride_h;
            }

            cb_weight_obj.push_back(weight_block_num_tiles);
        }  // for num_blocks_weight_h
        weight_h_offset += weight_inner_block_stride_h;

#ifdef FUSE_BIAS
        if constexpr (fuse_bias) {
            if (load_bias) {
                cb_bias_obj.reserve_back(bias_ntiles);

                uint32_t bias_write_offset = 0;
                uint32_t bias_block_size_bytes = 0;
                for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles; ++bias_tile) {
                    noc.async_read(
                        s_bias,
                        cb_bias_obj,
                        bias_pagesize,
                        {.page_id = bias_tile},
                        {.offset_bytes = bias_write_offset});
                    bias_write_offset += bias_pagesize;
                    bias_block_size_bytes += bias_pagesize;
                }
                noc.async_read_barrier();

// MCAST BIAS (shares some mcast args with weights)
#ifndef SKIP_MCAST
                // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
                // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to zero
                // for the next block
                weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                weights_mcast_sender_sem.set(0);

                // Now we have the block in the CB address, we can mcast to dests!
                // num_dests must not include source, since we are NOT really doing a local copy!
                mcast_dst.addr = cb_bias_obj.get_write_ptr();
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(cb_bias_obj.get_write_ptr()),
                    mcast_ep,
                    bias_block_size_bytes,
                    weights_mcast_num_cores,
                    {},
                    mcast_dst,
                    true);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc
                // even though cmd bufs are different Also, this only works because we are setting VCs statically (using
                // NOC_CMD_STATIC_VC).
                // We should also multicast the flag to destinations
                // num_dests must not include source, since we are NOT really doing a local copy!
                weights_mcast_receiver_sem.set_multicast(
                    noc,
                    mcast_rect.noc_x_start,
                    mcast_rect.noc_y_start,
                    mcast_rect.noc_x_end,
                    mcast_rect.noc_y_end,
                    weights_mcast_num_cores,
                    false);
#endif

                cb_bias_obj.push_back(bias_ntiles);
                load_bias = false;
            }
        }
#endif
#ifdef SPLIT_READER
        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
        }
#endif
    }  // out_num_blocks_h
    // Drain outstanding NOC writes AND atomics before returning (Metal 2.0 FW epilogue does not).
    noc.async_full_barrier();
}
