// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"
#include "tt-metalium/constants.hpp"
#include "noc_parameters.h"

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
    // GROUPNORM SENDER DESCIPTION
    // This is a high level desciption of the stages of this kernel, tags will be added to show where in the code each
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
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));

    constexpr uint32_t num_mcast_cores = get_compile_time_arg_val(2);
    constexpr uint32_t num_batch_group = get_compile_time_arg_val(3);
    constexpr uint32_t num_batches = get_compile_time_arg_val(4);
    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_compile_time_arg_val(5);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(6);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(7);
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(8);
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
    // These are numbers in absolute terms, on a per batch, per group, per core basis without tiling
    constexpr uint32_t num_channels_per_group = get_compile_time_arg_val(21);
    constexpr uint32_t num_rows_per_group = get_compile_time_arg_val(22);

    constexpr auto src0_args = TensorAccessorArgs<23>();
    constexpr auto out_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tt::constants::TILE_WIDTH - num_cols_per_group;
    uint32_t row_offset = num_cols_per_group;
    uint32_t index_g_offset = 0;
    uint32_t index_b_offset = 0;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t out_start_id = get_arg_val<uint32_t>(3);
    uint32_t num_channels_tiles = get_arg_val<uint32_t>(4);

    const bool has_mcast_first_group = get_arg_val<uint32_t>(5);
    const bool has_mcast_last_group = get_arg_val<uint32_t>(6);

    // mid mcast group
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(7);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(8);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(9);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(10);
    const uint32_t num_mcast_cores_mid_group = get_arg_val<uint32_t>(11);

    // first mcast group
    uint32_t mcast_first_group_dest_noc_start_x;
    uint32_t mcast_first_group_dest_noc_start_y;
    uint32_t mcast_first_group_dest_noc_end_x;
    uint32_t mcast_first_group_dest_noc_end_y;
    // last mcast group
    uint32_t mcast_last_group_dest_noc_start_x;
    uint32_t mcast_last_group_dest_noc_start_y;
    uint32_t mcast_last_group_dest_noc_end_x;
    uint32_t mcast_last_group_dest_noc_end_y;
    // volatile tt_l1_ptr uint32_t * noc_coord;
    tt_l1_ptr uint32_t* noc_coord_x;
    tt_l1_ptr uint32_t* noc_coord_y;

    // number of cores in mcast groups
    uint32_t num_mcast_cores_first_group;
    uint32_t num_mcast_cores_last_group;

    // noc addrs for first and last groups
    uint64_t reduce_sender_first_group_semaphore_noc_addr;
    uint64_t multicast_first_group_data_noc;
    uint64_t reduce_sender_last_group_semaphore_noc_addr;
    uint64_t multicast_last_group_data_noc;

    if (has_mcast_first_group and has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(16);

        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(17);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(18);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(19);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(20);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(21);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(22));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(22 + num_mcast_cores));

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else {
        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_mcast_cores));
    }

    const uint64_t reduce_sender_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        reduce_sender_semaphore_addr);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);

    if (has_mcast_first_group) {
        reduce_sender_first_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_first_group_data_noc = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            0);
    }
    if (has_mcast_last_group) {
        reduce_sender_last_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_last_group_data_noc = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            0);
    }

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    *reduce_sender_semaphore_addr_ptr = VALID;
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;  // sharded cb
    constexpr uint32_t cb_repack = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_reread_out = tt::CBIndex::c_23;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);

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

    uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    uint32_t out_block_hw_last = out_block_hw_normal;
    const uint32_t num_reads_of_input = 2;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = block_h % num_out_blocks;
        out_block_hw_last = out_block_h_last * block_w;
    }
    uint32_t cb_ex_external_tiles_required = num_out_blocks_padded * num_mcast_cores * 16 / single_tile_size_bytes;
    if ((num_out_blocks_padded * num_mcast_cores * 16) % single_tile_size_bytes) {
        cb_ex_external_tiles_required++;
    }

    index_b_offset = 0;
    for (uint32_t b = 0; b < num_batches; ++b) {
        index_g_offset = 0;
        row_offset = num_cols_per_group;

        for (uint32_t m = 0; m < num_groups; ++m) {
            // The following loop is for the 2 passes of input tensor required for GroupNorm
            // First Pass: Calculates average and variance value
            // Second Pass: Calculates final value
            // Definition: num_read_of_input = 2
            for (uint32_t cur_read_iteration = 0; cur_read_iteration < num_reads_of_input; ++cur_read_iteration) {
                uint32_t out_block_start_id_offset = 0;
                uint32_t cb_ex_external_bytes_written = 0;

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
                    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);
                    uint32_t l1_write_addr;
                    l1_write_addr = get_write_ptr(cb_in0);
                    cb_reserve_back(cb_in0, out_block_hw_normal);
                    for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                        for (uint32_t nt = 0; nt < block_w; nt++) {
                            noc_async_read_tile(
                                start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt +
                                    index_b_offset + index_g_offset,
                                src_a,
                                l1_write_addr);
                            l1_write_addr += src0_tile_bytes;
                            noc_async_read_barrier();
                        }
                    }
                    cb_push_back(cb_in0, out_block_hw_normal);
#endif
                    if (cur_read_iteration == 1) {
                        const auto dst_a = TensorAccessor(out_args, out_addr, single_tile_size_bytes);

                        // add or copy with previous output results
                        uint32_t block_w_curr =
                            index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

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

                if (cur_read_iteration == 0) {
                    cb_reserve_back(cb_ex_global, 2);
                    cb_wait_front(cb_ex_partial, 2);

                    // Read mean and variance arrays from cb_ex_partial, then combine using Welford
                    auto local_read_ptr = get_read_ptr(cb_ex_partial);
                    auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_read_ptr);
                    auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_read_ptr + single_tile_size_bytes);

                    auto local_result = combine_welford_stats<tt::constants::TILE_WIDTH, num_channels_per_group * num_rows_per_group / tt::constants::TILE_WIDTH, 2>(p_local_means, p_local_vars);

                    // Write this to cb_ex_global
                    auto global_means_ptr = get_write_ptr(cb_ex_global);
                    volatile uint16_t* p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
                    auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;
                    volatile uint16_t* p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
                    p_global_means[0] = local_result.mean;
                    p_global_vars[0] = local_result.variance;

                    cb_pop_front(cb_ex_partial, 2);

                    if constexpr (num_mcast_cores > 1) {
                        // Wait until all other cores have signaled that their partial data is ready
                        noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_mcast_cores - 1);
                        noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);

                        for (uint32_t i = 1; i < num_mcast_cores; ++i) {
                            uint64_t noc_means_addr =
                                get_noc_addr(noc_coord_x[i], noc_coord_y[i], global_means_ptr);
                            uint64_t noc_vars_addr =
                                get_noc_addr(noc_coord_x[i], noc_coord_y[i], global_vars_ptr);
                            noc_async_read_one_packet(noc_means_addr, global_means_ptr + i * NOC_DRAM_READ_ALIGNMENT_BYTES, NOC_DRAM_READ_ALIGNMENT_BYTES);
                            noc_async_read_one_packet(noc_vars_addr, global_vars_ptr + i * NOC_DRAM_READ_ALIGNMENT_BYTES, NOC_DRAM_READ_ALIGNMENT_BYTES);
                        }
                        noc_async_read_barrier();
                    }

                    // Read mean and variance arrays from cb_ex_global, then combine using Welford
                    auto global_result = combine_welford_stats<num_mcast_cores, num_channels_per_group * num_rows_per_group, 16>(p_global_means, p_global_vars);

                    // Write this to cb_ex_global
                    p_global_means[0] = global_result.mean;
                    p_global_vars[0] = global_result.variance;

                    if constexpr (num_mcast_cores > 1) {
                        // mcast to other cores
                        noc_async_write_multicast(
                            global_means_ptr,
                            multicast_data_noc | global_means_ptr,
                            2 * single_tile_size_bytes,
                            num_mcast_cores_mid_group,
                            true);
                        noc_semaphore_set_multicast(
                            reduce_sender_semaphore_addr,
                            reduce_sender_semaphore_noc_addr,
                            num_mcast_cores_mid_group,
                            false);

                        if (has_mcast_first_group) {
                            noc_async_write_multicast(
                                global_means_ptr,
                                multicast_first_group_data_noc | global_means_ptr,
                                2 * single_tile_size_bytes,
                                num_mcast_cores_first_group,
                                true);
                            noc_semaphore_set_multicast(
                                reduce_sender_semaphore_addr,
                                reduce_sender_first_group_semaphore_noc_addr,
                                num_mcast_cores_first_group,
                                false);
                        }

                        if (has_mcast_last_group) {
                            noc_async_write_multicast(
                                global_means_ptr,
                                multicast_last_group_data_noc | global_means_ptr,
                                2 * single_tile_size_bytes,
                                num_mcast_cores_last_group,
                                true);
                            noc_semaphore_set_multicast(
                                reduce_sender_semaphore_addr,
                                reduce_sender_last_group_semaphore_noc_addr,
                                num_mcast_cores_last_group,
                                false);
                        }
                        noc_async_write_barrier();
                    }

                    cb_push_back(cb_ex_global, 2);
                }
            }
            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == tt::constants::TILE_WIDTH) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == tt::constants::TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > tt::constants::TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > tt::constants::TILE_WIDTH) {
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
    uint32_t l1_write_addr_repack = get_write_ptr(cb_out0);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_wait_front(cb_repack_out, per_core_N);
        uint32_t in0_l1_read_addr = get_read_ptr(cb_repack_out);
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc_async_read_barrier();
        cb_pop_front(cb_repack_out, per_core_N);
    }
#endif
}
