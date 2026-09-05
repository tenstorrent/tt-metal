// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

void kernel_main() {
    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(2);
    constexpr uint32_t in0_last_ktile_h = get_compile_time_arg_val(3);
    // in0 mcast args
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(6);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(7);
    // block args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    // in0 mcast args
    constexpr uint32_t in0_mcast_dest_noc_start_x = get_compile_time_arg_val(9);
    constexpr uint32_t in0_mcast_dest_noc_start_y = get_compile_time_arg_val(10);
    constexpr uint32_t in0_mcast_dest_noc_end_x = get_compile_time_arg_val(11);
    constexpr uint32_t in0_mcast_dest_noc_end_y = get_compile_time_arg_val(12);
    constexpr uint32_t num_blocks_per_shard = get_compile_time_arg_val(14);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(15);
    // A block may span several storage shards (in0_block_w = shards_per_block * shard width). The
    // sender of such a block gathers the other shards over the NoC into the block's CB slot before
    // multicasting, so the multicast block count is K / in0_block_w regardless of the storage grid.
    constexpr uint32_t shards_per_block = get_compile_time_arg_val(16);
    constexpr uint32_t in0_shard_width_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t in0_block_h = in0_block_num_tiles / in0_block_w;
    constexpr uint32_t num_storage_cores =
        shards_per_block > 1 ? num_blocks * shards_per_block : num_blocks / num_blocks_per_shard;
    static_assert(shards_per_block == 1 || num_blocks_per_shard == 1, "a multi-shard block is one block per sender");

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg_val<uint32_t>(0);
    // if not worker core, skip
    if (worker_core_type == 0) {
        return;
    }
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const bool is_last_ktile_padded = static_cast<bool>(get_arg_val<uint32_t>(2));

    tt_l1_ptr uint32_t* in0_mcast_sender_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(3 + num_storage_cores));

    const uint32_t sender_block_id = sender_id * num_blocks_per_shard;
    // Storage core that sends block `block` (its shard, or the first shard of a multi-shard block).
    auto sender_core_of_block = [](uint32_t block) { return (block / num_blocks_per_shard) * shards_per_block; };
    // Storage cores whose index is not a multiple of shards_per_block never send: their shard is
    // gathered by the sender of the block they belong to.
    const bool sends_blocks = (sender_id % shards_per_block) == 0;

    constexpr uint32_t dfb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t dfb_id_in1 = get_named_compile_time_arg_val("cb_in1");          // staging on non-worker senders
    constexpr uint32_t dfb_id_in2 = get_named_compile_time_arg_val("cb_in0_sharded");  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    DataflowBuffer dfb_in1(dfb_id_in1);
    DataflowBuffer dfb_in2(dfb_id_in2);
    Semaphore<> sender_sem(get_compile_time_arg_val(4));
    Semaphore<> receiver_sem(get_compile_time_arg_val(5));

    uint32_t l1_write_addr_in0;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

    uint32_t local_read_addr = dfb_in2.get_read_ptr();

    // Gather the shards_per_block shards of this sender's block from the storage cores (this one
    // included; the shard CB sits at the same L1 address on every core) into the block's CB slot.
    // The reads are issued before the receiver handshake so they overlap it; the caller barriers.
    auto gather_block_shards = [&](uint32_t dst_addr) {
        UnicastEndpoint src_core;
        for (uint32_t j = 0; j < shards_per_block; ++j) {
            const uint32_t src_idx = sender_id + j;
            noc.async_read(
                src_core,
                CoreLocalMem<uint32_t>(dst_addr + j * in0_shard_width_bytes),
                in0_shard_width_bytes,
                {.noc_x = in0_mcast_sender_noc_x[src_idx],
                 .noc_y = in0_mcast_sender_noc_y[src_idx],
                 .addr = local_read_addr},
                {});
        }
    };

    if (worker_core_type == 1) {  // mcast sender + no compute
        if (!sends_blocks) {
            return;
        }

        for (uint32_t i = 0; i < num_blocks_per_shard; ++i) {
            const uint32_t block_id = sender_block_id / shards_per_block + i;

            // Operand 0
            l1_write_addr_in0 = dfb_in0.get_write_ptr();
            if (block_id % 2 != 0) {  // double buffer
                l1_write_addr_in0 += in0_block_size_bytes;
            }

            if constexpr (shards_per_block > 1) {
                // Stage in this core's in1 CB: it does no compute, so that CB is idle, and unlike its
                // in0 CB slot it is never written by the other senders' multicasts (this core sits
                // inside the multicast rectangle). The gather overlaps the credit wait below.
                const uint32_t staging_addr = dfb_in1.get_write_ptr();
                noc.async_write_barrier();  // the previous multicast out of the staging area
                gather_block_shards(staging_addr);
                local_read_addr = staging_addr;
            }

            // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
            sender_sem.wait(in0_mcast_num_dests);
            sender_sem.set(0);
            if constexpr (shards_per_block > 1) {
                noc.async_read_barrier();
            }

            // Now we have the block in the CB address, we can mcast to dests!

            // Zero out padded regions for tiles in the last K-column/row
            if constexpr (in0_last_ktile_w > 0) {
                if (is_last_ktile_padded && (i == num_blocks_per_shard - 1)) {
                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                        auto in0_last_ktile_ptr =
                            local_read_addr + (h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes;
                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_ptr);
                    }
                }
            }
            if constexpr (in0_last_ktile_h > 0) {
                if (is_last_ktile_padded && (i == num_blocks_per_shard - 1)) {
                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                        auto in0_last_ktile_ptr =
                            local_read_addr + ((in0_block_h - 1) * in0_block_w + w) * in0_single_tile_size_bytes;
                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(in0_last_ktile_ptr);
                    }
                }
            }

#ifndef SKIP_MCAST
            // num_dests must not include source, since we are NOT really doing a local copy!
            MulticastEndpoint mcast_dst;
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(local_read_addr),
                mcast_dst,
                in0_block_size_bytes,
                in0_mcast_num_cores - 1,
                {},
                {.noc_x_start = in0_mcast_dest_noc_start_x,
                 .noc_y_start = in0_mcast_dest_noc_start_y,
                 .noc_x_end = in0_mcast_dest_noc_end_x,
                 .noc_y_end = in0_mcast_dest_noc_end_y,
                 .addr = l1_write_addr_in0},
                true);
#endif

            receiver_sem.set_multicast(
                noc,
                in0_mcast_dest_noc_start_x,
                in0_mcast_dest_noc_start_y,
                in0_mcast_dest_noc_end_x,
                in0_mcast_dest_noc_end_y,
                in0_mcast_num_cores - 1);

            if constexpr (shards_per_block == 1) {
                local_read_addr += in0_block_size_bytes;
            }
        }

    } else if (worker_core_type == 2) {  // mcast sender + compute

        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = sender_core_of_block(block);

            dfb_in0.reserve_back(in0_block_num_tiles);
            // Set in0 semaphore value to INVALID
            receiver_sem.set(INVALID);

            if (sends_blocks && block_id == sender_id) {
                uint32_t l1_write_addr_in0 = dfb_in0.get_write_ptr();
                // copy start address of block, to be used for mcasting

                if constexpr (shards_per_block > 1) {
                    // reserve_back above guarantees the compute has drained this slot; the previous
                    // multicast out of it was flushed before its VALID went out.
                    gather_block_shards(l1_write_addr_in0);
                    local_read_addr = l1_write_addr_in0;
                }

                // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                sender_sem.wait(in0_mcast_num_dests - 1);
                sender_sem.set(0);
                if constexpr (shards_per_block > 1) {
                    noc.async_read_barrier();
                }

                // Zero out padded regions for tiles in the last K-column/row
                if constexpr (in0_last_ktile_w > 0) {
                    if (is_last_ktile_padded && (block == num_blocks - 1)) {
                        for (uint32_t h = 0; h < in0_block_h; ++h) {
                            auto in0_last_ktile_ptr =
                                local_read_addr + (h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes;
                            pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_ptr);
                        }
                    }
                }
                if constexpr (in0_last_ktile_h > 0) {
                    if (is_last_ktile_padded && (block == num_blocks - 1)) {
                        for (uint32_t w = 0; w < in0_block_w; ++w) {
                            auto in0_last_ktile_ptr =
                                local_read_addr + ((in0_block_h - 1) * in0_block_w + w) * in0_single_tile_size_bytes;
                            pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(in0_last_ktile_ptr);
                        }
                    }
                }
#ifndef SKIP_MCAST
                MulticastEndpoint mcast_dst;
                if constexpr (shards_per_block > 1) {
                    // The gathered block already sits in this core's slot: multicast to the others only.
                    noc.async_write_multicast(
                        CoreLocalMem<uint32_t>(local_read_addr),
                        mcast_dst,
                        in0_block_size_bytes,
                        in0_mcast_num_cores - 1,
                        {},
                        {.noc_x_start = in0_mcast_dest_noc_start_x,
                         .noc_y_start = in0_mcast_dest_noc_start_y,
                         .noc_x_end = in0_mcast_dest_noc_end_x,
                         .noc_y_end = in0_mcast_dest_noc_end_y,
                         .addr = l1_write_addr_in0},
                        true);
                } else {
                    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                        CoreLocalMem<uint32_t>(local_read_addr),
                        mcast_dst,
                        in0_block_size_bytes,
                        in0_mcast_num_cores,
                        {},
                        {.noc_x_start = in0_mcast_dest_noc_start_x,
                         .noc_y_start = in0_mcast_dest_noc_start_y,
                         .noc_x_end = in0_mcast_dest_noc_end_x,
                         .noc_y_end = in0_mcast_dest_noc_end_y,
                         .addr = l1_write_addr_in0},
                        true);
                }
#endif
                // Set local semaphore to VALID. For single-core configurations, this is all we need.
                receiver_sem.set(VALID);
                if constexpr (in0_mcast_num_cores > 1) {
                    receiver_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                        noc,
                        in0_mcast_dest_noc_start_x,
                        in0_mcast_dest_noc_start_y,
                        in0_mcast_dest_noc_end_x,
                        in0_mcast_dest_noc_end_y,
                        in0_mcast_num_cores);
                    // Flush to ensure the NoC has read the VALID value from receiver_sem's L1
                    // address before the next iteration overwrites it with INVALID.
                    noc.async_writes_flushed();
                }

                if constexpr (shards_per_block == 1) {
                    local_read_addr += in0_block_size_bytes;
                }

            } else {
                // Atomic increment source core counter
                sender_sem.up(noc, in0_mcast_sender_noc_x[block_id], in0_mcast_sender_noc_y[block_id], 1);
            }

            receiver_sem.wait(VALID);
            dfb_in0.push_back(in0_block_num_tiles);
        }
    } else {  // mcast receiver + compute

        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = sender_core_of_block(block);

            // get the mcast sender noc

            // Operand 0
            dfb_in0.reserve_back(in0_block_num_tiles);

            // Set in0 semaphore value to INVALID
            receiver_sem.set(INVALID);
            // Atomic increment source core counter
            sender_sem.up(noc, in0_mcast_sender_noc_x[block_id], in0_mcast_sender_noc_y[block_id], 1);
            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            receiver_sem.wait(VALID);

            dfb_in0.push_back(in0_block_num_tiles);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
