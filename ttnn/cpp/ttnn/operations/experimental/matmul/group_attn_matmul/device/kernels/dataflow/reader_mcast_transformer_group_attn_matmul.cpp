// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    uint32_t i = 0;

    uint32_t has_work_for_mcast_kv_heads = get_arg_val<uint32_t>(i++);
    if (has_work_for_mcast_kv_heads == 0) {
        return;
    }
    uint32_t has_work_for_q_heads = get_arg_val<uint32_t>(i++);
    const bool has_work_for_q_heads_bool = has_work_for_q_heads == 1;

    uint32_t src1_addr = get_arg_val<uint32_t>(i++);
    uint32_t Mt = get_arg_val<uint32_t>(i++);
    uint32_t Nt = get_arg_val<uint32_t>(i++);
    uint32_t num_kv_heads = get_arg_val<uint32_t>(i++);  // in1[1] (ie. in1 C)
    uint32_t in1_CKtNt = get_arg_val<uint32_t>(i++);
    uint32_t in1_CKtNt_mul_32 = get_arg_val<uint32_t>(i++);
    uint32_t blocks = get_arg_val<uint32_t>(i++);
    uint32_t in1_start_id = get_arg_val<uint32_t>(i++);

    // matmul params
    uint32_t in0_block_w = get_arg_val<uint32_t>(i++);
    uint32_t out_block_w = get_arg_val<uint32_t>(i++);
    uint32_t in1_num_subblocks = get_arg_val<uint32_t>(i++);
    uint32_t in1_num_blocks = get_arg_val<uint32_t>(i++);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(i++);

    // constants
    uint32_t Nt_bytes = get_arg_val<uint32_t>(i++);
    uint32_t in1_block_w_tile_bytes = get_arg_val<uint32_t>(i++);
    uint32_t out_last_subblock_w = get_arg_val<uint32_t>(i++);
    uint32_t in1_last_block_w_tile_read_bytes = get_arg_val<uint32_t>(i++);
    uint32_t in1_last_block_addr_skip = get_arg_val<uint32_t>(i++);

    constexpr bool transpose_hw_bool = get_compile_time_arg_val(0) == 1;
    constexpr bool row_major = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(2);
    constexpr auto in1_args = TensorAccessorArgs<3>();
    using In1McastArgs = McastArgs<in1_args.next_compile_time_args_offset(), 20>;
    constexpr In1McastArgs in1_mcast_args;

    constexpr uint32_t cb_id_in1 = 1;  // mcast receive all kv_heads; compute chooses which kv_heads to use for matmul
    constexpr uint32_t cb_id_in2 = 2;  // all interleaved or sharded KV heads for one user batch

    Noc noc;
    CircularBuffer cb_in1_obj(cb_id_in1);
    CircularBuffer cb_in2_obj(cb_id_in2);
    std::optional<In1McastArgs::SenderPipe> in1_sender_pipe;
    std::optional<In1McastArgs::ReceiverPipe> in1_receiver_pipe;
    if (in1_mcast_args.can_send()) {
        in1_sender_pipe.emplace(in1_mcast_args.sender(noc));
    }
    if (in1_mcast_args.can_receive()) {
        in1_receiver_pipe.emplace(in1_mcast_args.receiver(noc));
    }

    constexpr uint32_t num_rows_in_one_tile = 32;
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    constexpr uint32_t in0_num_blocks_w = 1;  // TODO: Must be 1; generalize to support inner dim blocking

#ifndef IN1_SHARDED
    const auto s1 = TensorAccessor(in1_args, src1_addr);
#endif

    uint32_t local_noc_x = my_x[noc.get_noc_id()];
    uint32_t local_noc_y = my_y[noc.get_noc_id()];
    UnicastEndpoint local_src;

    bool mcast_in1_to_local_cb = false;
    uint32_t in1_sharded_cb_addr = cb_in2_obj.get_read_ptr();
#ifdef IN1_SHARDED
    // Only used for sharded
    // Don't need to track batch because user batch must be 32 (ie. Mt must be 1)
    uint32_t in1_sharded_l1_addr_Nt = in1_sharded_cb_addr;
    uint32_t in1_block_w_tile_read_bytes = in1_block_w_tile_bytes;
    if (in1_num_blocks == 1) {
        mcast_in1_to_local_cb =
            true;  // For sharded in1, if no blocking along Nt, directly mcast instead of doing local copies
    }
#else
    // Only used for interleaved
    uint32_t in1_batch;
    uint32_t in1_tensor_id_along_Nt;
    uint32_t in1_tensor_id;
    uint32_t in1_block_addr_skip = 0;  // Skip padded subblocks to prevent reading from undefined memory
    uint32_t out_subblock_w_ = out_subblock_w;
#endif

    for (uint32_t b = 0; b < blocks; b++) {  // TODO: Must be 1
#ifndef IN1_SHARDED
        in1_batch = in1_start_id;
#endif

        for (uint32_t m = 0; m < Mt; m++) {  // TODO: Must be 1; generalize to support batch > 32 (ie. Mt > 1)
#ifndef IN1_SHARDED
            in1_tensor_id_along_Nt = in1_batch;
#endif
            for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
                const bool last_out = in1_block == in1_num_blocks - 1;
                if (last_out) {
#ifdef IN1_SHARDED
                    in1_block_w_tile_read_bytes = in1_last_block_w_tile_read_bytes;
#else
                    out_subblock_w_ = out_last_subblock_w;
                    in1_block_addr_skip = in1_last_block_addr_skip;
#endif
                }

#ifndef IN1_SHARDED
                uint32_t in1_tensor_id = in1_tensor_id_along_Nt;  // Tracks id along Kt, kv_heads, and batch
#endif
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                    for (uint32_t in0_block = 0; in0_block < in0_num_blocks_w;
                         in0_block++) {  // TODO: Must be 1; generalize to support inner dim blocking
                        cb_in1_obj.reserve_back(in1_block_num_tiles);
                        uint32_t l1_write_addr_in1 = cb_in1_obj.get_write_ptr();

                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks;
                             in1_subblock++) {  // TODO: Must be 1; for generic padding, need full + partial subblocks
                            // Read in1 block
                            if (in1_mcast_args.should_send(tile_row_id)) {
// MCAST SENDER: send all kv_heads in one user batch
#ifdef IN1_SHARDED
                                if (!mcast_in1_to_local_cb) {
                                    // TODO: Try to optimize away the local copy and self-mcast instead for sharded;
                                    // some things to try:
                                    // - block sharding so each core gets correct block along out width
                                    // - overlap copy with mcasting to hide away the local copy time (maybe offload some
                                    // work to writer)

                                    // Copy to cb_id_in1 to mcast
                                    uint32_t in1_sharded_l1_addr = in1_sharded_l1_addr_Nt;
                                    uint32_t write_offset = 0;
                                    for (uint32_t kv_heads_id = 0; kv_heads_id < num_kv_heads; kv_heads_id++) {
                                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                            noc.async_read(
                                                local_src,
                                                cb_in1_obj,
                                                in1_block_w_tile_read_bytes,
                                                {.noc_x = local_noc_x,
                                                 .noc_y = local_noc_y,
                                                 .addr = in1_sharded_l1_addr},
                                                {.offset_bytes = write_offset});
                                            in1_sharded_l1_addr += Nt_bytes;  // Increment by Nt to get to next kt
                                            write_offset += in1_block_w_tile_bytes;
                                        }
                                        // Next head follows after finishing KtNt, so no need to increment
                                        // write_offset
                                    }
                                    // These indices are local to each core, so don't modify when looping
                                    // num_rows_in_one_tile
                                    noc.async_read_barrier();
                                }
#else
                                uint32_t in1_tensor_id_along_Kt = in1_tensor_id;
                                uint32_t write_offset = 0;
                                for (uint32_t kv_heads_id = 0; kv_heads_id < num_kv_heads; kv_heads_id++) {
                                    for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                        uint32_t in1_tensor_current_id = in1_tensor_id_along_Kt;
                                        for (uint32_t w = 0; w < out_subblock_w_; w++) {
                                            noc.async_read(
                                                s1,
                                                cb_in1_obj,
                                                in1_tile_bytes,
                                                {.page_id = in1_tensor_current_id},
                                                {.offset_bytes = write_offset});
                                            in1_tensor_current_id++;  // Increment to get next Nt
                                            write_offset += in1_tile_bytes;
                                        }
                                        write_offset += in1_block_addr_skip;
                                        in1_tensor_id_along_Kt += Nt;  // Increment by Nt to get next Kt
                                    }
                                    // Next head follows after finishing KtNt, so no need to increment
                                }
                                noc.async_read_barrier();
#endif

                                if (mcast_in1_to_local_cb) {  // directly mcast data in in1 sharded cb
                                    in1_sender_pipe->send(
                                        in1_sharded_cb_addr, l1_write_addr_in1, in1_block_num_tiles * in1_tile_bytes);
                                } else {  // mcast from l1_write_addr_in1 which is populated locally by copying from in1
                                          // sharded or interleaved
                                    in1_sender_pipe->send(
                                        l1_write_addr_in1, l1_write_addr_in1, in1_block_num_tiles * in1_tile_bytes);
                                }
                            } else if (in1_mcast_args.can_receive()) {
                                // MCAST RECEIVER: receive all kv_heads in one user batch
                                // All cores in mcast grid needs to participate in receiving otherwise data corruption
                                // since we mcast from and to the same CB

                                in1_receiver_pipe->receive(tile_row_id);
                            }
                            if (has_work_for_q_heads_bool) {
                                cb_in1_obj.push_back(in1_block_num_tiles);
                            } else {
                                // Mcast is in lockstep; this makes write ptr addresses are synced properly for cores
                                // that only send and have no compute / writer active
                                cb_in1_obj.push_back(in1_block_num_tiles);
                                cb_in1_obj.pop_front(in1_block_num_tiles);
                            }

#ifndef IN1_SHARDED
                            in1_tensor_id += in1_CKtNt;
#endif
                        }  // in1_num_subblocks loop
                    }  // in0_num_blocks_w loop
                }  // 32 tiles loop

#ifdef IN1_SHARDED
                in1_sharded_l1_addr_Nt += in1_block_w_tile_bytes;
#else
                in1_tensor_id_along_Nt += out_block_w;
#endif
            }  // in1_num_blocks loop
        }  // Mt loop

#ifndef IN1_SHARDED
        in1_batch += in1_CKtNt_mul_32;  // different depending on transpose_hw
#endif
    }  // B loop
    noc.async_write_barrier();
}
