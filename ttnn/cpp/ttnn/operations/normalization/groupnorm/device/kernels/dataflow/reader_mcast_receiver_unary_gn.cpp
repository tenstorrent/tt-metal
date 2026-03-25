// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

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
    // GROUPNORM RECEIVER DESCRIPTION
    // This is a high level description of the stages of this kernel, tags will be added to show where in the code each
    // stage starts and ends
    //
    // Batch Loop:
    //   Group Loop:
    //     This is the process which repeats for every group
    //     First Read of data:
    //       If Receiver:
    //           Send partial reduction of Average to Sender Core
    //       If Sender:
    //           Pack Partials:
    //               Accumulate partial reductions into single tile
    //               Calculates the Global average sum
    //           Send Global:
    //               Send Global Average to all Receiver cores
    //     Second Read of data:
    //       If Receiver:
    //           Send partial reduction of Variance to Sender Core
    //       If Sender:
    //           Pack Partials:
    //               Accumulate partial reductions into single tile
    //               Calculates the Global Variance sum
    //           Send Global:
    //               Send Global Variance to all Receiver cores
    //          Third Read of data:
    //
    //      // clang-format on

    constexpr uint32_t reduce_receiver_semaphore_id = get_named_compile_time_arg_val("reduce_receiver_semaphore_id");
    constexpr uint32_t reduce_sender_semaphore_id = get_named_compile_time_arg_val("reduce_sender_semaphore_id");

    constexpr uint32_t num_batch_group = get_named_compile_time_arg_val("num_batch_group");
    constexpr uint32_t num_batches = get_named_compile_time_arg_val("num_batches");

    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    const uint32_t per_core_N_bytes = get_named_compile_time_arg_val("per_core_N_bytes");
    const uint32_t per_core_N_bytes_with_stride = get_named_compile_time_arg_val("per_core_N_bytes_with_stride");
    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("TILE_HEIGHT");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");
    constexpr uint32_t block_hw = get_named_compile_time_arg_val("block_hw");

    constexpr uint32_t num_cols_per_group = get_named_compile_time_arg_val("num_cols_per_group");
    constexpr uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr uint32_t block_w_last = get_named_compile_time_arg_val("block_w_last");
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_named_compile_time_arg_val("GROUP_SIZE_IS_POWER_OF_2");
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_named_compile_time_arg_val("GROUP_SIZE_SMALLER_THAN_TILE_W");
    constexpr uint32_t group_row_offset = get_named_compile_time_arg_val("group_row_offset");
    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");

    // 19 and 20 are used in welford version but unused in this version
    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;
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

    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;  // E[x] partial reduce
    constexpr uint32_t cb_ex2_partial_id = tt::CBIndex::c_21;  // E[x] partial reduce
    constexpr uint32_t cb_ex_id = tt::CBIndex::c_9;// E[x] partial reduce
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;// E[x] global reduce
    constexpr uint32_t cb_ex2_id = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex2_global_id = tt::CBIndex::c_14;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_x_id = tt::CBIndex::c_24;
    constexpr uint32_t cb_reread_out_id = tt::CBIndex::c_23;

    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    experimental::Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    experimental::CircularBuffer cb_ex_partial(cb_ex_partial_id);
    experimental::CircularBuffer cb_ex2_partial(cb_ex2_partial_id);
    experimental::CircularBuffer cb_ex_global(cb_ex_global_id);
    experimental::CircularBuffer cb_ex2_global(cb_ex2_global_id);
    experimental::CircularBuffer cb_in0(cb_in0_id);
    experimental::CircularBuffer cb_repack(cb_repack_id);
    experimental::CircularBuffer cb_repack_out(cb_repack_out_id);
    experimental::CircularBuffer cb_out0(cb_out0_id);
    experimental::CircularBuffer cb_reread_out(cb_reread_out_id);

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial_id);
    const DataFormat out_data_format = get_dataformat(cb_out0_id);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = cb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    experimental::UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = cb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(self_ep, experimental::CoreLocalMem<uint32_t>(l1_write_addr_repack), per_core_N_bytes, {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0}, {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        cb_repack.push_back(per_core_N);
    }
#endif

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    uint32_t out_block_hw_last = out_block_hw_normal;
    const uint32_t num_reads_of_input = 3;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        uint32_t residual = block_h - (num_out_blocks * out_block_h_normal);
        num_out_blocks_padded += (residual / out_block_h_normal + 1);
        out_block_h_last = residual % out_block_h_normal;
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
                    const uint32_t src0_tile_bytes = get_tile_size(cb_in0_id);
                    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);
                    uint32_t l1_write_addr;
                    l1_write_addr = cb_in0.get_write_ptr();
                    cb_in0.reserve_back(out_block_hw_normal);
                    for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                        for (uint32_t nt = 0; nt < block_w; nt++) {
                            noc.async_read(
                                src_a,
                                experimental::CoreLocalMem<uint32_t>(l1_write_addr),
                                src0_tile_bytes,
                                {.page_id = start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt + index_b_offset +
                                    index_g_offset},
                                {});
                            l1_write_addr += src0_tile_bytes;
                            noc.async_read_barrier();
                        }
                    }
                    cb_in0.push_back(out_block_hw_normal);

#endif
                    if (cur_read_iteration == 0 || cur_read_iteration == 1) {
                        //Section for waiting for local reduce to be pushed to a cb_ex_partial
                        reduce_sender_sem.set(INVALID);
                        if (cur_read_iteration == 0) {
                            //Wait for local avg calculation
                            cb_ex_partial.wait_front(1);
                        } else {
                            //Wait for local variance calculation
                            cb_ex2_partial.wait_front(1);
                        }
                        reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1);

                        reduce_sender_sem.wait(VALID);
                        if (cur_read_iteration == 0) {
                            cb_ex_partial.pop_front(1);
                        } else {
                            cb_ex2_partial.pop_front(1);
                        }
                    } else if (cur_read_iteration == 2) {
                        // add or copy with previous output results
                        const auto dst_a = TensorAccessor(out_args, out_addr, single_tile_size_bytes);

                        uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

                        const uint32_t dst_tile_bytes = get_tile_size(cb_reread_out_id);
                        uint32_t l1_write_addr;
                        l1_write_addr = cb_reread_out.get_write_ptr();
                        cb_reread_out.reserve_back(out_block_hw_normal);

                        for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                            for (uint32_t nt = 0; nt < block_w_curr; nt++) {
                                noc.async_read(
                                    dst_a,
                                    experimental::CoreLocalMem<uint32_t>(l1_write_addr),
                                    single_tile_size_bytes,
                                    {.page_id = out_start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt +
                                        index_b_offset + index_g_offset},
                                    {});
                                l1_write_addr += dst_tile_bytes;
                                noc.async_read_barrier();
                            }
                        }
                        cb_reread_out.push_back(out_block_hw_normal);
                    }
                    out_block_start_id_offset += out_block_h_actual * num_channels_tiles;
                }

                if (cur_read_iteration == 0 || cur_read_iteration == 1) {
                    reduce_sender_sem.set(INVALID);
                    if (cur_read_iteration == 0) {
                        cb_ex_global.reserve_back(1);
                        reduce_sender_sem.wait(VALID);
                        cb_ex_global.push_back(1);
                    } else if (cur_read_iteration == 1) {
                        cb_ex2_global.reserve_back(1);
                        reduce_sender_sem.wait(VALID);
                        cb_ex2_global.push_back(1);
                    }
                }
            }

            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == tile_width) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }
        }
        index_b_offset += num_tiles_per_batch;
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = cb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = cb_repack_out.get_read_ptr();
        uint32_t src_addr_in0 = in0_l1_read_addr;
        experimental::UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(self_ep, experimental::CoreLocalMem<uint32_t>(l1_write_addr_repack), per_core_N_bytes, {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0}, {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        cb_repack_out.pop_front(per_core_N);
    }
#endif
}
