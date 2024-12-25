// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "tools/profiler/kernel_profiler.hpp"

// This address corresponds to:
// auto buf = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
// uint32_t* data = buf->data + 16;
// On tensix cores, on WH, data pointer value is 0x001e4, and it has space for 32 4 byte values.
// On ncrisc, we use upper 16 elements of the array so, the address is 0x001e4 + 16*4 = 0x00224
// There is 16 values available to use on ncrisc
// MAKE SURE TO DOUBLE CHECK THIS VALUE WHEN REBASING
volatile uint32_t* dbg_dump_ncrisc = (volatile uint32_t*)0x00224;

void kernel_main() {
    uint32_t rt_args_idx = 0;
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in0 mcast args
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args
    const uint32_t last_block_h = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;

    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(2);
    constexpr uint32_t in0_tensor_next_inner_dim_block_stride = get_compile_time_arg_val(3);
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = get_compile_time_arg_val(4);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(6);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(7);
    constexpr bool extract_shard_sub_blocks = (bool)get_compile_time_arg_val(8);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(10);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(12);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(13);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(16);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(17);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(18);  // if 0
    constexpr uint32_t batch = get_compile_time_arg_val(19);

    constexpr bool fuse_op = (bool)get_compile_time_arg_val(20);

    MatmulOpReceiver fused_op_receiver;
    if constexpr (fuse_op) {
        fused_op_receiver = MatmulOpReceiver(
            true, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks_inner_dim,
            in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
        );
    }

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_single_tile_size_bytes;

#ifdef IN0_SHARDED
    // In case we need to send multiple blocks per shard, in0 sharded cb is cb2 and we extract the sub-blocks to cb0
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t shard_num_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride_bytes =
        in0_tensor_next_h_dim_block_stride * in0_single_tile_size_bytes;

    uint32_t noc_shard_read_start_addr = 0;
    if constexpr (extract_shard_sub_blocks) {
        constexpr uint32_t cb_id_in2 = 2;  // in0 sharded cb if extract_shard_sub_blocks
        noc_shard_read_start_addr = get_read_ptr(cb_id_in2);
    } else {
        cb_reserve_back(cb_id_in0, shard_num_tiles);
        cb_push_back(cb_id_in0, shard_num_tiles);
    }
#else
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);
    constexpr const uint32_t in0_tile_hw = get_tile_hw(cb_id_in0);
    const InterleavedAddrGenFast<in0_is_dram, in0_tile_hw> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = in0_single_tile_size_bytes, .data_format = in0_data_format};
#endif

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    const uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x,
        in0_mcast_dest_noc_start_y,
        in0_mcast_dest_noc_end_x,
        in0_mcast_dest_noc_end_y,
        in0_mcast_receiver_semaphore_addr);

    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y, in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y, 0);

#ifdef IN0_SHARDED
    uint32_t in0_start_address = get_write_ptr(cb_id_in0);
#endif
#endif

    for (uint32_t b = 0; b < batch; ++b) {
#ifdef IN0_SHARDED
        uint32_t in0_tensor_current_h_dim_block_start_addr = noc_shard_read_start_addr;
#endif
        uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
#ifdef IN0_SHARDED
                uint32_t in0_tensor_current_inner_dim_block_start_addr = in0_tensor_current_h_dim_block_start_addr;
#endif
                uint32_t in0_tensor_current_inner_dim_block_start_tile_id = in0_tensor_current_h_dim_block_tile_id;
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    if constexpr (fuse_op) {
                        fused_op_receiver.update_current_block_start_tile_id(
                            block, in0_tensor_current_inner_dim_block_start_tile_id, in0_tensor_start_tile_id);
                    }
                    *(dbg_dump_ncrisc + 0) = block;
#ifndef IN0_SHARDED
                    // Operand 0
                    cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

#ifndef SKIP_MCAST
                    uint32_t in0_start_address =
                        l1_write_addr_in0;  // copy start address of block, to be used for mcasting
#endif

                    // Copy in0 block into CB, as the default kernel
                    uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_inner_dim_block_start_tile_id;
                    *(dbg_dump_ncrisc + 1) = l1_write_addr_in0;
                    *(dbg_dump_ncrisc + 2) = 0x00baba01;
                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                        uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                        for (uint32_t w = 0; w < in0_block_w; ++w) {
                            if (bh < num_blocks_h_dim - 1 || h < last_block_h) {
                                noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                            }
                            l1_write_addr_in0 += in0_single_tile_size_bytes;
                            in0_tensor_tile_id += in0_tensor_stride_w;
                        }
                        in0_tensor_row_start_tile_id += in0_tensor_stride_h;
                    }
                    in0_tensor_current_inner_dim_block_start_tile_id += in0_tensor_next_inner_dim_block_stride;

                    // Barrier! make sure the reads are done
                    *(dbg_dump_ncrisc + 2) = 0x04baba01;

                    noc_async_read_barrier();
                    *(dbg_dump_ncrisc + 2) = 0x08baba01;

#else
                    if constexpr (extract_shard_sub_blocks) {
                        // Operand 0
                        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

#ifndef SKIP_MCAST
                        in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting
#endif

                        uint64_t noc_shard_read_addr = get_noc_addr(in0_tensor_current_inner_dim_block_start_addr);

                        for (uint32_t i = 0; i < in0_block_h; i++) {
                            noc_async_read(noc_shard_read_addr, l1_write_addr_in0, shard_read_width);

                            l1_write_addr_in0 += shard_read_width;
                            noc_shard_read_addr += shard_read_stride;
                        }

                        in0_tensor_current_inner_dim_block_start_addr += shard_read_width;
                        noc_async_read_barrier();
                    }
#endif

#ifndef SKIP_MCAST
                    // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr (i.e.
                    // its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for
                    // the next block

                    *(dbg_dump_ncrisc + 2) = 0x0Cbaba01;
                    *(dbg_dump_ncrisc + 3) = (uint32_t)in0_mcast_sender_semaphore_addr_ptr;
                    *(dbg_dump_ncrisc + 4) = in0_mcast_num_dests;

                    noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
                    noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                    *(dbg_dump_ncrisc + 2) = 0x0Dbaba01;

                    // Now we have the block in the CB address, we can mcast to dests!
                    uint64_t in0_multicast_data_addr = in0_multicast_data_noc | in0_start_address;

                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_async_write_multicast(
                        in0_start_address,
                        in0_multicast_data_addr,
                        in0_block_size_bytes,
                        in0_mcast_num_cores,
                        true,
                        true);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc,
                    // same cmd_buf Also, this only works because we are setting VCs statically (using
                    // NOC_CMD_STATIC_VC).

#ifdef ARCH_BLACKHOLE
                    // On Blackhole the flush is needed because NoC latency is higherthan L1 <-> RISCV
                    // latency which means data could be changed before write is issued.
                    noc_async_writes_flushed();
#endif
                    *(dbg_dump_ncrisc + 2) = 0x0Ebaba01;
                    // We should also multicast the flag to destinations
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_semaphore_set_multicast(
                        in0_mcast_receiver_semaphore_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_cores);

                    *(dbg_dump_ncrisc + 2) = 0x0Fbaba01;

#endif

#ifndef IN0_SHARDED
                    cb_push_back(cb_id_in0, in0_block_num_tiles);
                    *(dbg_dump_ncrisc + 2) = 0x08baba02;
#else
                    if constexpr (extract_shard_sub_blocks) {
                        cb_push_back(cb_id_in0, in0_block_num_tiles);
                    }
#endif
                }
            }
#ifdef IN0_SHARDED
            in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride_bytes;
#endif
            in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
        }
        in0_tensor_start_tile_id += MtKt;
    }
    noc_async_write_barrier();
}
