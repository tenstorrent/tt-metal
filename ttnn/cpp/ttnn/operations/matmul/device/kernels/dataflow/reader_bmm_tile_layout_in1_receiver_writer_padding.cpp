// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/ring_buffer.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

// Latency measurement: read low 32 bits of wall clock (no-op cost: ~1 cycle)
inline uint32_t read_wall_clock() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    // READER
    uint32_t rt_args_idx = 0;
    // in1 mcast args
    const uint32_t in1_mcast_sender_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_sender_noc_y = get_arg_val<uint32_t>(rt_args_idx++);

    // WRITER
    // out tensor args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(rt_args_idx++);

    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(rt_args_idx++);

#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_h_dim = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t last_num_blocks_w_dim = get_arg_val<uint32_t>(rt_args_idx++);
#endif

    // COMPILE TIME ARGS
    // READER
    // in1 block args
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(0);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(3);
    // in1 mcast args
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(4));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    // batch args
    constexpr uint32_t batch = get_compile_time_arg_val(6);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(7);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(9);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_tensor_next_w_dim_block_stride = get_compile_time_arg_val(11);
    constexpr uint32_t out_tensor_next_h_dim_block_stride = get_compile_time_arg_val(12);

    // out subblock args
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(14);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(15);

    // batch args
    constexpr uint32_t MtNt = get_compile_time_arg_val(16);  // if 0
    // Don't need batch; same as batch from READER args

#ifdef FUSE_BIAS
    // in3 block args
    constexpr uint32_t in3_block_w = get_compile_time_arg_val(17);

    constexpr uint32_t cb_id_in3 = 3;
#endif
    constexpr bool fuse_op_reduce_scatter = (bool)get_compile_time_arg_val(18);

    constexpr auto out_args = TensorAccessorArgs<19>();
    OpSignaler op_signaler;
    if constexpr (fuse_op_reduce_scatter) {
        op_signaler = OpSignaler(rt_args_idx);
    }
    // WRITER

    constexpr uint32_t cb_id_in1 = 1;

    // WRITER
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;

    // WRITER
    // single-tile
    const uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(cb_id_out0);

    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);

    // WRITER
    const auto s = TensorAccessor(out_args, out_tensor_addr, output_single_tile_size_bytes);

    const uint64_t in1_mcast_sender_semaphore_noc_addr =
        get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_mcast_sender_semaphore_addr);

    // Latency tracking: max cycles spent in each semaphore wait site
    uint32_t max_wait_in1_sem = 0;      // Site 0x10: wait for in1 mcast data
    uint32_t max_wait_bias_sem = 0;     // Site 0x20: wait for bias mcast data
    uint32_t max_wait_out_sharded = 0;  // Site 0x30: wait for all output tiles (OUT_SHARDED)
    uint32_t max_wait_out_cb = 0;       // Site 0x40: wait for output CB tiles (non-sharded writer)

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    // Operand 1
                    cb_reserve_back(cb_id_in1, in1_block_num_tiles);

                    // Set in1 semaphore value to INVALID
                    noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

                    // Atomic increment source core counter
                    noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

                    // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    {
                        uint32_t t0 = read_wall_clock();
                        noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);
                        uint32_t dt = read_wall_clock() - t0;
                        if (dt > max_wait_in1_sem) {
                            max_wait_in1_sem = dt;
                        }
                    }

                    cb_push_back(cb_id_in1, in1_block_num_tiles);
                }

#ifdef FUSE_BIAS
                // Only read bias on first batch, or we have multiple output blocks
                if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                    // Operand 2
                    cb_reserve_back(cb_id_in3, in3_block_w);

                    // Set in1 semaphore value to INVALID
                    noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

                    // Atomic increment source core counter
                    noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

                    // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    {
                        uint32_t t0 = read_wall_clock();
                        noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);
                        uint32_t dt = read_wall_clock() - t0;
                        if (dt > max_wait_bias_sem) {
                            max_wait_bias_sem = dt;
                        }
                    }

                    cb_push_back(cb_id_in3, in3_block_w);
                }
#endif

#ifndef OUT_SHARDED
                // WRITER
                uint32_t num_blocks_h_dim_ = bh >= last_num_blocks_h_dim - 1 ? last_num_blocks_h_dim : num_blocks_h_dim;
                uint32_t num_blocks_w_dim_ = bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
                uint32_t out_num_nonzero_subblocks_h_ = out_num_nonzero_subblocks_h;
                uint32_t out_num_nonzero_subblocks_w_ = out_num_nonzero_subblocks_w;
                if (bh == num_blocks_h_dim_ - 1) {
                    out_num_nonzero_subblocks_h_ = out_last_num_nonzero_subblocks_h;
                }
                if (bw == num_blocks_w_dim_ - 1) {
                    out_num_nonzero_subblocks_w_ = out_last_num_nonzero_subblocks_w;
                }
                uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;
                for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {
                    uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                    for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {
                        uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                        uint32_t out_subblock_h_ = out_subblock_h;
                        uint32_t out_subblock_w_ = out_subblock_w;
                        uint32_t subblock_tiles_addr_skip = 0;
                        if (bh == num_blocks_h_dim_ - 1 && sbh == out_num_nonzero_subblocks_h_ - 1) {
                            out_subblock_h_ = out_last_subblock_h;
                        }
                        if (bw == num_blocks_w_dim_ - 1 && sbw == out_num_nonzero_subblocks_w_ - 1) {
                            out_subblock_w_ = out_last_subblock_w;
                            subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                        }

                        {
                            uint32_t t0 = read_wall_clock();
                            cb_wait_front(cb_id_out0, out_subblock_tile_count);
                            uint32_t dt = read_wall_clock() - t0;
                            if (dt > max_wait_out_cb) {
                                max_wait_out_cb = dt;
                            }
                        }
                        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                        for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                            uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                            for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                if (bh < num_blocks_h_dim_ && bw < num_blocks_w_dim_) {
                                    noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                                }

                                l1_read_addr += output_single_tile_size_bytes;

                                out_tensor_tile_id += out_tensor_stride_w;
                            }
                            // Skip padded tiles in subblock along row
                            l1_read_addr += subblock_tiles_addr_skip;
                            out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                        }

                        noc_async_write_barrier();

                        cb_pop_front(cb_id_out0, out_subblock_tile_count);
                        out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                    }
                    // Pop fully padded subblocks along the row
                    if (bw == num_blocks_w_dim_ - 1) {
                        cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
                        cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
                    }
                    out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                }
                // Pop row(s) of fully padded subblocks
                if (bh == num_blocks_h_dim_ - 1) {
                    cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
                    cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
                }
#endif
                out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;
            }
            out_tensor_current_h_dim_block_tile_id += out_tensor_next_h_dim_block_stride;
        }
        out_tensor_start_tile_id += MtNt;

        if (fuse_op_reduce_scatter) {
            // Signal reduce_scatter to go
            op_signaler.synchronize_workers_and_signal_op(0);
        }
    }

#if OUT_SHARDED
    {
        uint32_t t0 = read_wall_clock();
        cb_wait_front(
            cb_id_out0,
            batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
        uint32_t dt = read_wall_clock() - t0;
        if (dt > max_wait_out_sharded) {
            max_wait_out_sharded = dt;
        }
    }
#endif

    // Push max wait latencies to ring buffer.
    // Format: 0xSSNNNNNN where SS=site ID, NNNNNN=max cycles (24-bit, up to 16M cycles ~16ms)
    // Site 0x10: in1 mcast semaphore wait
    // Site 0x20: bias mcast semaphore wait
    // Site 0x30: output CB wait (OUT_SHARDED final)
    // Site 0x40: output CB wait (non-sharded per-subblock)
    WATCHER_RING_BUFFER_PUSH(0x10000000 | (max_wait_in1_sem & 0x00FFFFFF));
    WATCHER_RING_BUFFER_PUSH(0x20000000 | (max_wait_bias_sem & 0x00FFFFFF));
    WATCHER_RING_BUFFER_PUSH(0x30000000 | (max_wait_out_sharded & 0x00FFFFFF));
    WATCHER_RING_BUFFER_PUSH(0x40000000 | (max_wait_out_cb & 0x00FFFFFF));
}
