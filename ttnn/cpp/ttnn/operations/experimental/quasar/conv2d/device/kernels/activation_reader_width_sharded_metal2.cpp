// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of activation_reader_width_sharded.cpp (width-sharded conv2d).
// Algorithm body identical to the legacy kernel; host-binding surface migrated:
//   - CB-index CTAs -> dfb:: tokens (act / act_row_major / act_tilized / act_sharded / reader_indices)
//   - semaphore-id CTAs -> sem::act_mcast_sender / sem::act_mcast_receiver
//   - remaining positional CTAs -> get_arg(args::name)
//   - RTAs (this_core_x/y, num_cores_x) -> get_arg(args::name); the X/Y mcast NoC lookup tables are
//     positional runtime VARARGS (variable count = full grid) accessed via get_vararg(i)
//   - experimental::CB -> DataflowBuffer
//   - DRAM config-tensor read uses tensor::reader_indices (CONFIG_TENSOR_IN_DRAM path)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Multicast rectangle: NOC coordinate range for multicast destinations.
struct McastRect {
    uint32_t noc_x_start, noc_y_start, noc_x_end, noc_y_end;
};
#ifndef COMPILE_FOR_TRISC
using McastDst = noc_traits_t<MulticastEndpoint>::dst_args_mcast_type;
#endif

// Only a part of the total channel depth (width) is used in one block.
template <int window_height, int window_width>
FORCE_INLINE void read_channels(
    Noc noc,
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_bytes,
    const uint32_t conv_act_c_read_bytes,
    const uint32_t stride_h_bytes,
    const uint32_t stride_w_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_bytes);
#pragma GCC unroll window_height
    for (uint32_t outer = 0; outer < window_height; outer++) {
        uint32_t act_l1_read_addr_row_offset = act_l1_read_addr_plus_offset;
#pragma GCC unroll window_width
        for (uint32_t inner = 0; inner < window_width; inner++) {
            experimental::read_with_state(noc, l1_write_addr_act, act_l1_read_addr_row_offset);
            l1_write_addr_act += conv_act_c_read_bytes;
            act_l1_read_addr_row_offset += stride_w_bytes;
        }
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

void kernel_main() {
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr uint32_t conv_act_size_w = get_arg(args::input_size_w);
    constexpr uint32_t conv_act_c_read_bytes = get_arg(args::conv_act_c_read_bytes);
    constexpr uint32_t weight_size_h = get_arg(args::weight_size_h);
    constexpr uint32_t weight_size_w = get_arg(args::weight_size_w);
    constexpr uint32_t act_block_h_datums = get_arg(args::act_block_h_datums);
    constexpr uint32_t act_block_num_tiles = get_arg(args::act_block_num_tiles);
    constexpr uint32_t num_input_cores = get_arg(args::num_input_cores);
    constexpr uint32_t act_num_blocks_h = get_arg(args::act_num_blocks_h);
    constexpr uint32_t act_num_blocks_w = get_arg(args::act_num_blocks_w);
    Semaphore act_mcast_sender_sem(sem::act_mcast_sender);
    Semaphore act_mcast_receiver_sem(sem::act_mcast_receiver);
    constexpr McastRect mcast_rect = {
        get_arg(args::act_mcast_start_x),
        get_arg(args::act_mcast_start_y),
        get_arg(args::act_mcast_end_x),
        get_arg(args::act_mcast_end_y)};
    constexpr uint32_t act_mcast_sender_size_bytes = get_arg(args::act_mcast_sender_size_bytes);
    constexpr uint32_t num_output_cores = get_arg(args::num_output_cores);
    constexpr uint32_t num_reader_cores = get_arg(args::num_reader_cores);

    constexpr uint32_t num_mcast_cores = num_input_cores > num_output_cores ? num_input_cores : num_output_cores;

    uint32_t this_core_x = get_arg(args::this_core_x);
    uint32_t this_core_y = get_arg(args::this_core_y);
    // Num of cols of compute cores. (Total Cores, not active cores.)
    uint32_t num_cores_x = get_arg(args::num_cores_x);

    // X and Y lookup tables for translating logical to physical cores, supplied as positional
    // runtime varargs: first num_cores_x entries are the X lookup, the next entries are the Y lookup.
    // get_vararg(i) returns the i-th vararg.

    // Equivalent to Core Index.
    uint32_t this_core_id = this_core_x + (num_cores_x * this_core_y);

    if (this_core_id >= num_mcast_cores) {
        return;
    }

    DataflowBuffer reader_indices_cb(dfb::reader_indices);
    DataflowBuffer act_rm_cb(dfb::act_row_major);
    DataflowBuffer act_cb(dfb::act);
    DataflowBuffer tilized_in0_cb(dfb::act_tilized);
    DataflowBuffer sharded_act_cb(dfb::act_sharded);
    Noc noc;

#ifdef CONFIG_TENSOR_IN_DRAM
    // Read this core's slice of the reader-indices config tensor from DRAM into reader_indices_cb.
    {
        const auto config_accessor = TensorAccessor(tensor::reader_indices);
        constexpr uint32_t config_page_size = get_arg(args::config_page_size);
        noc.async_read(config_accessor, reader_indices_cb, config_page_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        reader_indices_cb.push_back(1);
    }
#endif

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_cb.get_write_ptr());

    MulticastEndpoint mcast_ep;
    McastDst mcast_dst = {
        .noc_x_start = mcast_rect.noc_x_start,
        .noc_y_start = mcast_rect.noc_y_start,
        .noc_x_end = mcast_rect.noc_x_end,
        .noc_y_end = mcast_rect.noc_y_end,
        .addr = 0};

    act_mcast_receiver_sem.set(VALID);

    constexpr uint32_t conv_act_c_bytes = conv_act_c_read_bytes * act_num_blocks_w;
    constexpr uint32_t stride_w_bytes = conv_act_c_bytes * dilation_w;
    constexpr uint32_t stride_h_bytes = (conv_act_size_w)*conv_act_c_bytes * dilation_h;

    uint32_t act_l1_read_addr = sharded_act_cb.get_read_ptr();
    experimental::set_read_state<conv_act_c_read_bytes>(noc, act_l1_read_addr);
    uint32_t reader_idx = 0;
    uint32_t l1_write_addr_act = 0;

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t ntile_height = act_block_h_datums / TILE_HEIGHT;
    constexpr uint32_t ntile_width = act_block_num_tiles / ntile_height;

    for (uint32_t block_h_index = 0; block_h_index < act_num_blocks_h; block_h_index++) {
        act_l1_read_addr = sharded_act_cb.get_read_ptr();
        uint32_t old_reader_idx = reader_idx;
        for (uint32_t block_w_index = 0; block_w_index < act_num_blocks_w; block_w_index++) {
            reader_idx = old_reader_idx;
            if (this_core_id < num_input_cores) {
                uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
                uint16_t num_elems = two_reader_indices & 0xffff;

                uint16_t remaining_indexes = TILE_HEIGHT;
                while (num_elems--) {
                    reader_idx++;
                    two_reader_indices = packed_reader_indices_ptr[reader_idx];
                    uint16_t start_ind = two_reader_indices & 0xffff;
                    uint16_t end_ind = two_reader_indices >> 16;
                    for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                        if (remaining_indexes == TILE_HEIGHT) {
                            l1_write_addr_act = act_rm_cb.get_write_ptr();
                            act_rm_cb.reserve_back(ntile_width);
                        }
                        read_channels<weight_size_h, weight_size_w>(
                            noc,
                            l1_write_addr_act,
                            act_l1_read_addr,
                            ind,
                            conv_act_c_bytes,
                            conv_act_c_read_bytes,
                            stride_h_bytes,
                            stride_w_bytes);

                        if (--remaining_indexes == 0) {
                            noc.async_read_barrier();
                            act_rm_cb.push_back(ntile_width);
                            l1_write_addr_act = act_rm_cb.get_write_ptr();
                            remaining_indexes = TILE_HEIGHT;
                        }
                    }
                }
                if (remaining_indexes && remaining_indexes != TILE_HEIGHT) {
                    noc.async_read_barrier();
                    act_rm_cb.push_back(ntile_width);
                }
                reader_idx++;

                act_l1_read_addr += conv_act_c_read_bytes;
            } else {
                for (uint32_t tile_h_index = 0; tile_h_index < ntile_height; tile_h_index++) {
                    act_rm_cb.reserve_back(ntile_width);
                    act_rm_cb.push_back(ntile_width);
                }
            }

#ifndef SKIP_MCAST
            for (uint32_t act_w_outer_i = 0; act_w_outer_i < num_input_cores; act_w_outer_i++) {
                act_cb.reserve_back(act_block_num_tiles);
                if (act_w_outer_i == this_core_id) {
                    act_mcast_sender_sem.wait_min(num_mcast_cores - 1);
                    act_mcast_sender_sem.set(0);

                    act_mcast_receiver_sem.set(INVALID);

                    tilized_in0_cb.wait_front(act_block_num_tiles);

                    auto tilized_src = CoreLocalMem<uint32_t>(tilized_in0_cb.get_read_ptr());

                    mcast_dst.addr = act_cb.get_write_ptr();
                    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                        tilized_src,
                        mcast_ep,
                        act_mcast_sender_size_bytes,
                        num_reader_cores,
                        {.offset_bytes = 0},
                        mcast_dst,
                        true);

                    act_mcast_receiver_sem.set(VALID);
                    act_mcast_receiver_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                        noc,
                        mcast_rect.noc_x_start,
                        mcast_rect.noc_y_start,
                        mcast_rect.noc_x_end,
                        mcast_rect.noc_y_end,
                        num_reader_cores);
                    noc.async_write_barrier();
                } else {
                    act_mcast_receiver_sem.set(INVALID);

                    uint32_t sender_logical_x = act_w_outer_i % num_cores_x;
                    uint32_t sender_logical_y = act_w_outer_i / num_cores_x;

                    // X lookup table is varargs [0, num_cores_x); Y lookup follows it.
                    uint32_t sender_x = get_vararg(sender_logical_x);
                    uint32_t sender_y = get_vararg(num_cores_x + sender_logical_y);

                    act_mcast_sender_sem.up(noc, sender_x, sender_y, 1);

                    act_mcast_receiver_sem.wait(VALID);
                }

                act_cb.push_back(act_block_num_tiles);
            }  // num_input_cores
            tilized_in0_cb.pop_front(act_block_num_tiles);
#endif
        }
    }
    noc.async_read_barrier();
    noc.async_write_barrier();
}
