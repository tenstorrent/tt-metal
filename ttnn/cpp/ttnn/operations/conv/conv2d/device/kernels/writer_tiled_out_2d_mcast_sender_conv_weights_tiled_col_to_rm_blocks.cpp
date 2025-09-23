// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "height_sharded_reader_common.hpp"

#define ENABLE_DEBUG 1

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

constexpr uint32_t weight_size_h = get_compile_time_arg_val(27);  // Input filter window height
constexpr uint32_t weight_size_w = get_compile_time_arg_val(20);  // Input filter window width

template <int window_height, int window_width>
FORCE_INLINE void read_dilated_channels(
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_bytes,
    const uint32_t stride_h_bytes,
    const uint32_t stride_w_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_bytes);
#pragma GCC unroll weight_size_h
    for (uint32_t outer = 0; outer < window_height; outer++) {
        uint32_t act_l1_read_addr_row_offset = act_l1_read_addr_plus_offset;
#pragma GCC unroll weight_size_w
        for (uint32_t inner = 0; inner < window_width; inner++) {
            // Read the partial depth.
            noc_async_read_one_packet_with_state<true>(act_l1_read_addr_row_offset, l1_write_addr_act);
            // Increment by full depth to go to the next pixel
            l1_write_addr_act += conv_act_c_bytes;
            act_l1_read_addr_row_offset += stride_w_bytes;
        }
        // Go to the next row
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

FORCE_INLINE
void read_channels(
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_read_bytes,
    const uint32_t coalesced_read_bytes,
    const uint32_t stride_h_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_read_bytes);
#pragma GCC unroll weight_size_h
    for (uint32_t inner = 0; inner < weight_size_h; inner++) {
        noc_async_read_one_packet_with_state<true>(act_l1_read_addr_plus_offset, l1_write_addr_act);
        l1_write_addr_act += coalesced_read_bytes;
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);

    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(8);
    constexpr uint32_t weight_block_height_ntiles = get_compile_time_arg_val(9);
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(10);
    constexpr uint32_t weight_stride_h = get_compile_time_arg_val(11);
    constexpr uint32_t weight_next_block_stride_w = get_compile_time_arg_val(13);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);

    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(16);
    constexpr uint32_t weight_block_height_num_outer_in = get_compile_time_arg_val(17);

#ifdef SPLIT_READER
    // Use existing args that factory already passes but are currently ignored
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(5);
    constexpr uint32_t window_outer = get_compile_time_arg_val(6);  // num_blocks_act_w
    constexpr bool sliced_inner_dim = num_blocks_weight_h > 1;      // Derived like block sharded reader

    // Additional args - will need factory integration for block sharded + split reader
    constexpr uint32_t act_block_num_tiles_split_last = get_compile_time_arg_val(18);  // This is what factory passes
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(20);
    constexpr uint32_t padded_conv_act_size_w = get_compile_time_arg_val(21);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(22);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(24);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(25);
    constexpr uint32_t stride_w = get_compile_time_arg_val(26);
#endif
    // Without split reader, weight tensor args start at 35
    constexpr auto s_weight_args = TensorAccessorArgs<35>();
    constexpr auto s_bias_args = TensorAccessorArgs<s_weight_args.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i++);
    // Bias arg. Unused if bias fusion is not enabled.
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);
    const uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i++);
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i++);

    // mcast args
    const uint32_t weights_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_dests = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_cores = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;

#ifdef SPLIT_READER
    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_second_reader>();
    }
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // Initial setup for second reader (starting from second reader's data)
    uint32_t start_reader_idx = (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1;
    uint32_t reader_idx = start_reader_idx;

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    constexpr uint32_t window_outer_offset = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;

    // TODO add config tensor if in DRAM

    const uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
    // DPRINT << "noc_async_read_one_packet_set_state: " << get_noc_addr(act_l1_read_addr) << " " <<
    // coalesced_read_bytes
    //        << ENDL();

#endif

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
    const auto s_bias = TensorAccessor(s_bias_args, bias_addr, bias_pagesize);

    bool load_bias = true;
#endif

    constexpr uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    const auto s_weight = TensorAccessor(s_weight_args, weight_addr_dram_base, weight_tile_nbytes);
    constexpr uint32_t weights_block_size_bytes = weight_tile_nbytes * weight_block_num_tiles;

    // Pre-compute constants used in tile_id calculation (preserving exact original logic)
    constexpr uint32_t tiles_per_full_block =
        num_blocks_weight_h * weight_block_height_ntiles * weight_block_height_num_outer_in * weight_block_width_ntiles;
    constexpr uint32_t height_stride_factor = weight_block_height_ntiles * weight_stride_h;

    uint32_t weight_start_tile_id = out_start_tile_id_w;
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            // Read activation data using block sharded pattern (for second reader)
#ifdef SPLIT_READER
            uint32_t reader_offset = act_l1_read_addr;
#endif
            for (uint32_t height_block_index = 0; height_block_index < num_blocks_weight_h; height_block_index++) {
#ifdef SPLIT_READER
                noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                reader_idx = start_reader_idx;
                cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
                if (is_sender_core) {
                    uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
                    if constexpr (sliced_inner_dim) {
                        read_sticks<
                            dilation_w,
                            coalesced_read_bytes,
                            conv_act_c_read_bytes,
                            act_block_w_extra_align_bytes,
                            stride_w_bytes,
                            weight_size_w,
                            stride_w>(packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);
                    } else {
                        uint16_t num_elems = packed_reader_indices_ptr[reader_idx] & 0xffff;
                        while (num_elems--) {
                            reader_idx++;
                            uint16_t start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
                            uint16_t end_ind = packed_reader_indices_ptr[reader_idx] >> 16;
                            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                                if constexpr (dilation_w == 1) {
                                    read_channels(
                                        l1_write_addr_act,
                                        act_l1_read_addr,
                                        ind,
                                        conv_act_c_read_bytes,
                                        coalesced_read_bytes,
                                        stride_h_bytes);
                                    if constexpr (act_block_w_extra_align_bytes) {
                                        l1_write_addr_act += act_block_w_extra_align_bytes;
                                    }
                                } else {
                                    read_dilated_channels<weight_size_h, weight_size_w>(
                                        l1_write_addr_act,
                                        act_l1_read_addr,
                                        ind,
                                        conv_act_c_read_bytes,
                                        stride_h_bytes,
                                        stride_w_bytes);
                                }
                            }
                        }
                        reader_idx++;
                    }
                    noc_async_read_barrier();
                }
                DPRINT << "ACT SECOND READER PUSH BACK: " << act_block_num_tiles_split_last << ENDL();
                cb_push_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
                reader_offset += window_outer_offset;
#endif
                // Compute height block offset once per outer loop iteration
                const uint32_t height_block_offset = height_block_index * height_stride_factor;
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
                    cb_reserve_back(cb_id_weight, weight_block_num_tiles);
                    uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);

                    const uint32_t outer_block_offset = weight_tile_h_outer_i * tiles_per_full_block;
                    uint32_t tile_id = weight_start_tile_id + height_block_offset + outer_block_offset;
                    // mcast args
                    uint32_t weights_start_address = weight_write_l1_addr;
                    for (uint32_t block_weight_h = 0; block_weight_h < weight_block_height_ntiles; block_weight_h++) {
                        uint32_t weight_tile_id = tile_id;

                        for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles;
                             ++weight_tile_w_i) {
                            noc_async_read_tile(weight_tile_id++, s_weight, weight_write_l1_addr);
                            weight_write_l1_addr += weight_tile_nbytes;
                        }
                        tile_id += weight_stride_h;
                    }
                    noc_async_read_barrier();

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

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and same
                    // vc even though cmd bufs are different Also, this only works because we are setting VCs statically
                    // (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                    // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may
                    // not be sent in order they are issued
                    noc_async_writes_flushed();
#endif
                    // We should also multicast the flag to destinations
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_semaphore_set_multicast(
                        weights_mcast_receiver_semaphore_addr,
                        weights_mcast_receiver_semaphore_noc_addr,
                        weights_mcast_num_cores);
#endif
                    DPRINT << "WEIGHTS PUSH BACK: " << weight_block_num_tiles << ENDL();
                    cb_push_back(cb_id_weight, weight_block_num_tiles);
                }  // for weight_block_height_num_outer
            }
#ifdef SPLIT_READER
            // Update reader index for next iteration (split reader increment)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
#endif
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
