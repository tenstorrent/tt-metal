// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // clang-format off
    // Definitions
    //   block_h: This the length of the row we wish to processes in terms of tiles
    //
    //   out_block_...: This is the length of our Circular Buffer, sometimes the length of out tensors(block_h) are larger than L1 space, so we
    //   have to process chunks of this data at a time
    //   this chunk is called an out_block
    //
    //   num_out_blocks: This is the number of chunks specified by the use, such that a CBs (length defined by out_block) fit in L1
    //   (Users should minimize the number of num_out_blocks for better perf)
    //
    //   ...normal:  If num_out_blocks evenly divides block_h, then all chunks are the size normal
    //
    //   ...last: If num_out_blocks does not divides block_h, the leftovers are put into a chunk of length last
    //
    //   sender: This refers to a core that does aggregation calculations
    //   for the group of cores
    //
    //   receiver: This the cores that receive the aggregated results from sender, they only do
    //   local computations that they send to the sender for final aggregation
    //
    // GROUPNORM RECIEVER DESCIPTION
    // This is a high level desciption of the stages of this kernel, tags will be added to show where in the code each
    // stage starts and ends
    //
    // Batch Loop:
    //   Group Loop:
    //     This is the process which repeats for every group
    //     First Read of data:
    //       If Reciever:
    //           Send partial reduction of Average to Sender Core
    //       If Sender:
    //           Pack Partials:
    //               Accumulate partial reductions into single tile
    //               Calculates the Global average sum
    //           Send Global:
    //               Send Global Average to all Receiver cores
    //     Second Read of data:
    //       If Reciever:
    //           Send partial reduction of Varience to Sender Core
    //       If Sender:
    //           Pack Partials:
    //               Accumulate partial reductions into single tile
    //               Calculates the Global Varience sum
    //           Send Global:
    //               Send Global Varience to all Receiver cores
    //          Third Read of data:
    //
    //      // clang-format on
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool out_is_dram = get_compile_time_arg_val(1) == 1;

    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));

    constexpr uint32_t num_batch_group = get_compile_time_arg_val(4);
    constexpr uint32_t num_batches = get_compile_time_arg_val(5);

    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_compile_time_arg_val(6);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(7);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(9);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(10);

    constexpr uint32_t block_h = get_compile_time_arg_val(11);
    constexpr uint32_t block_w = get_compile_time_arg_val(12);
    constexpr uint32_t block_hw = get_compile_time_arg_val(13);

    constexpr uint32_t num_cols_per_group = get_compile_time_arg_val(14);
    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(15);

    constexpr uint32_t block_w_last = get_compile_time_arg_val(16);
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_compile_time_arg_val(17);
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_compile_time_arg_val(18);
    constexpr uint32_t group_row_offset = get_compile_time_arg_val(19);
    constexpr uint32_t num_out_blocks = get_compile_time_arg_val(20);

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t tile_w_minux_group_size = TILE_WIDTH - num_cols_per_group;
    uint32_t row_offset = num_cols_per_group;
    uint32_t index_g_offset = 0;
    uint32_t index_b_offset = 0;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t out_start_id = get_arg_val<uint32_t>(3);
    uint32_t num_channels_tiles = get_arg_val<uint32_t>(4);
    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;  // E[x] partial reduce
    constexpr uint32_t cb_ex2_partial = tt::CBIndex::c_21;  // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;          // E[x] partial reduce
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_13;        // E[x]^2 partial reduce
    constexpr uint32_t cb_ex2_global = tt::CBIndex::c_14;  // E[x]^2 global reduce
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;         // input cb
    constexpr uint32_t cb_repack = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;
    constexpr uint32_t cb_reread_out = tt::CBIndex::c_23;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);  // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial);          // data format
    const DataFormat out_data_format = get_dataformat(cb_out0);

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, reduce_receiver_semaphore_addr);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = get_read_ptr(cb_in0);
    uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_reserve_back(cb_repack, per_core_N);
        uint32_t l1_write_addr_repack = get_write_ptr(cb_repack);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc_async_read_barrier();
        cb_push_back(cb_repack, per_core_N);
    }
#endif

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    uint32_t out_block_hw_last = out_block_hw_normal;
    const uint32_t num_reads_of_input = 3;
    if constexpr(block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = block_h % num_out_blocks;
        out_block_hw_last = out_block_h_last * block_w;
    }

    index_b_offset = 0;

    // Start Batch Loop:
    for (uint32_t b = 0; b < num_batches; ++b) {
        index_g_offset = 0;
        row_offset = num_cols_per_group;

        // Start Group Loop:
        for (uint32_t i = 0; i < num_groups; ++i) {
            //The following loop is for the 3 passes of input tensor required for GroupNorm
            //First Pass: Calculates average value
            //Second Pass: Calculates the Variance value
            //Third Pass: Calculates final value
            //Definition: num_read_of_input = 3
            for (uint32_t cur_read_iteration = 0; cur_read_iteration < num_reads_of_input; ++cur_read_iteration) {
                uint32_t out_block_start_id_offset = 0;
                for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                    uint32_t out_block_h_actual, out_block_hw_actual;
                    if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                        out_block_h_actual = out_block_h_last;
                        out_block_hw_actual = out_block_hw_last;
                    } else {
                        out_block_h_actual = out_block_h_normal;
                        out_block_hw_actual = out_block_hw_normal;
                    }
#if !defined(READER_REPACK) or !defined(TILIZE_IN)
                    const uint32_t src0_tile_bytes = get_tile_size(cb_in0);
                    const DataFormat src0_data_format = get_dataformat(cb_in0);
                    const InterleavedAddrGenFast<src0_is_dram> src_a = {
                        .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};
                    uint32_t l1_write_addr;
                    l1_write_addr = get_write_ptr(cb_in0);
                    cb_reserve_back(cb_in0, out_block_hw_normal);
                    for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                        for (uint32_t nt = 0; nt < block_w; nt++) {
                            noc_async_read_tile(
                                start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt + index_b_offset +
                                    index_g_offset,
                                src_a,
                                l1_write_addr);
                            l1_write_addr += src0_tile_bytes;
                            noc_async_read_barrier();
                        }
                    }
                    cb_push_back(cb_in0, out_block_hw_normal);

#endif
                    if (cur_read_iteration == 0 || cur_read_iteration == 1) {
                        //Section for wating for local reduce to be pushed to a cb_ex_partial
                        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
                        if (cur_read_iteration == 0) {
                            //Wait for local avg calculation
                            cb_wait_front(cb_ex_partial, 1);
                        } else {
                            //Wait for local variance calculation
                            cb_wait_front(cb_ex2_partial, 1);
                        }
                        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);

                        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
                        if (cur_read_iteration == 0) {
                            cb_pop_front(cb_ex_partial, 1);
                        } else {
                            cb_pop_front(cb_ex2_partial, 1);
                        }
                    } else if (cur_read_iteration == 2) {
                        const InterleavedAddrGenFast<out_is_dram> dst_a = {
                            .bank_base_address = out_addr,
                            .page_size = single_tile_size_bytes,
                            .data_format = out_data_format};

                        // add or copy with previous output results
                        uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

                        const uint32_t dst_tile_bytes = get_tile_size(cb_reread_out);
                        uint32_t l1_write_addr;
                        l1_write_addr = get_write_ptr(cb_reread_out);
                        cb_reserve_back(cb_reread_out, out_block_hw_normal);

                        for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                            for (uint32_t nt = 0; nt < block_w_curr; nt++) {
                                noc_async_read_tile(
                                    out_start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt +
                                        index_b_offset + index_g_offset,
                                    dst_a,
                                    l1_write_addr);
                                l1_write_addr += dst_tile_bytes;
                                noc_async_read_barrier();
                            }
                        }
                        cb_push_back(cb_reread_out, out_block_hw_normal);
                    }
                    out_block_start_id_offset += out_block_h_actual * num_channels_tiles;
                }

                if (cur_read_iteration == 0 || cur_read_iteration == 1) {
                    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
                    uint32_t cb_mcast_receive;
                    if (cur_read_iteration == 0) {
                        cb_mcast_receive = cb_ex_global;
                    } else if (cur_read_iteration == 1) {
                        cb_mcast_receive = cb_ex2_global;
                    }
                    cb_reserve_back(cb_mcast_receive, 1);
                    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
                    cb_push_back(cb_mcast_receive, 1);
                }
            }

            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }
        }  // End Group Loop:
        index_b_offset += num_tiles_per_batch;
    }  // End Batch Loop:

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = get_write_ptr(cb_out0);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_wait_front(cb_repack_out, per_core_N);
        uint32_t in0_l1_read_addr = get_read_ptr(cb_repack_out);
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc_async_read_barrier();
        cb_pop_front(cb_repack_out, per_core_N);
    }
#endif
}
