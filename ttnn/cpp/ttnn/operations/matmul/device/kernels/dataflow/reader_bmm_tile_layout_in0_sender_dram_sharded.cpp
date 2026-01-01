// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

#ifdef ARCH_QUASAR
// Quasar emulator self-loopback workaround: a NoC transfer whose source and destination are both this
// core's L1 (the mcast group degenerates to the sender alone on a single-node grid) is done with the shared RISC copy
// helper, which also issues the fence that orders the uncached-alias stores ahead of the semaphore / credit that
// publishes them. The helper takes uncached-alias addresses (what the DFB getters hand out on Quasar DM); a
// LocalTensorAccessor base is the plain (cached) L1 address, so normalise both operands first.
#include "ttnn/operations/kernel_helper_functions/local_l1_copy.hpp"
#include "internal/local_mem/address_access.h"
FORCE_INLINE uint32_t l1_uncached(uint32_t addr) {
    return addr >= MEM_L1_UNCACHED_BASE ? addr : MEM_L1_UNCACHED_BASE + (addr - MEM_L1_BASE);
}
#endif

// A K block within one activation shard is strided in M. Multicast its row slices
// directly, avoiding a staging copy. Whole shards and gathered blocks are contiguous.
template <bool IncludeSource>
FORCE_INLINE void multicast_block(const Noc& noc, uint32_t src, uint32_t dst) {
    constexpr uint32_t tile_bytes = get_tile_size(dfb::in0);
    constexpr uint32_t row_bytes = get_arg(args::in0_block_w) * tile_bytes;
    constexpr uint32_t source_stride =
        get_arg(args::shards_per_block) > 1 ? row_bytes : get_arg(args::in0_shard_width_bytes);
    constexpr uint32_t rows = get_arg(args::in0_block_num_tiles) / get_arg(args::in0_block_w);
    constexpr bool contiguous = rows == 1 || source_stride == row_bytes;
    constexpr uint32_t transfers = contiguous ? 1 : rows;
    constexpr uint32_t bytes = contiguous ? get_arg(args::in0_block_size_bytes) : row_bytes;
    constexpr uint32_t cores = get_arg(args::in0_mcast_num_cores);
    const MulticastEndpoint endpoint;
    for (uint32_t row = 0; row < transfers; ++row) {
#ifdef ARCH_QUASAR
        if constexpr (IncludeSource && cores == 1) {
            local_l1_copy(l1_uncached(dst + row * row_bytes), l1_uncached(src + row * source_stride), bytes);
        } else
#endif
            if constexpr (IncludeSource) {
            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                CoreLocalMem<uint32_t>(src + row * source_stride),
                endpoint,
                bytes,
                cores,
                {},
                {.noc_x_start = get_arg(args::in0_mcast_dest_noc_start_x),
                 .noc_y_start = get_arg(args::in0_mcast_dest_noc_start_y),
                 .noc_x_end = get_arg(args::in0_mcast_dest_noc_end_x),
                 .noc_y_end = get_arg(args::in0_mcast_dest_noc_end_y),
                 .addr = dst + row * row_bytes},
                true);
        } else {
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(src + row * source_stride),
                endpoint,
                bytes,
                cores - 1,
                {},
                {.noc_x_start = get_arg(args::in0_mcast_dest_noc_start_x),
                 .noc_y_start = get_arg(args::in0_mcast_dest_noc_start_y),
                 .noc_x_end = get_arg(args::in0_mcast_dest_noc_end_x),
                 .noc_y_end = get_arg(args::in0_mcast_dest_noc_end_y),
                 .addr = dst + row * row_bytes},
                true);
        }
    }
}

void kernel_main() {
    // COMPILE TIME ARGS
    // in0 block args
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto in0_block_size_bytes = get_arg(args::in0_block_size_bytes);
    constexpr auto in0_last_ktile_w = get_arg(args::in0_last_ktile_w);
    constexpr auto in0_last_ktile_h = get_arg(args::in0_last_ktile_h);
    // in0 mcast args
    constexpr auto in0_mcast_num_dests = get_arg(args::in0_mcast_num_dests);
    constexpr auto in0_mcast_num_cores = get_arg(args::in0_mcast_num_cores);
    // block args
    constexpr auto num_blocks = get_arg(args::num_blocks);
    // in0 mcast args
    constexpr auto in0_mcast_dest_noc_start_x = get_arg(args::in0_mcast_dest_noc_start_x);
    constexpr auto in0_mcast_dest_noc_start_y = get_arg(args::in0_mcast_dest_noc_start_y);
    constexpr auto in0_mcast_dest_noc_end_x = get_arg(args::in0_mcast_dest_noc_end_x);
    constexpr auto in0_mcast_dest_noc_end_y = get_arg(args::in0_mcast_dest_noc_end_y);
    constexpr auto num_blocks_per_shard = get_arg(args::num_blocks_per_shard);
    constexpr auto in0_block_w = get_arg(args::in0_block_w);
    // A block may span several storage shards (in0_block_w = shards_per_block * shard width). The
    // sender of such a block gathers the other shards over the NoC into the block's buffer slot
    // before multicasting, so the multicast block count is K / in0_block_w regardless of the
    // storage grid. 1 is the stock path: a block is a shard or a fraction of one.
    constexpr auto shards_per_block = get_arg(args::shards_per_block);
    constexpr auto in0_shard_width_bytes = get_arg(args::in0_shard_width_bytes);
    constexpr uint32_t in0_block_h = in0_block_num_tiles / in0_block_w;
    constexpr uint32_t in0_block_row_bytes = in0_block_w * get_tile_size(dfb::in0);
    constexpr uint32_t source_row_bytes = shards_per_block > 1 ? in0_block_row_bytes : in0_shard_width_bytes;
    constexpr uint32_t num_storage_cores =
        shards_per_block > 1 ? num_blocks * shards_per_block : num_blocks / num_blocks_per_shard;
    static_assert(shards_per_block == 1 || num_blocks_per_shard == 1, "a multi-shard block is one block per sender");

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg(args::worker_core_type);
    // if not worker core, skip
    if (worker_core_type == 0) {
        return;
    }
    const uint32_t sender_id = get_arg(args::sender_id);
    const bool is_last_ktile_padded = static_cast<bool>(get_arg(args::is_last_ktile_padded));

    // The mcast senders' noc coordinates arrive as a runtime vararg block: the noc-x value for
    // each storage core, then the noc-y value for each. A receiver picks the pair belonging to the
    // block it is waiting on, so the index is data-derived and the block stays positional.
    constexpr uint32_t sender_noc_x_base = 0;
    constexpr uint32_t sender_noc_y_base = num_storage_cores;

    const uint32_t sender_block_id = sender_id * num_blocks_per_shard;
    // Storage core that sends block `block` (its shard, or the first shard of a multi-shard block).
    auto sender_core_of_block = [](uint32_t block) { return (block / num_blocks_per_shard) * shards_per_block; };
    // Storage cores whose index is not a multiple of shards_per_block never send: their shard is
    // gathered by the sender of the block they belong to.
    const bool sends_blocks = (sender_id % shards_per_block) == 0;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb::in0);
    constexpr DataFormat in0_data_format = get_dataformat(dfb::in0);

    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    // The resident in0 width shard: only its L1 base address is needed (no FIFO traffic).
    const LocalTensorAccessor<uint32_t> in0_shard(tensor::in0_shard);
    Semaphore<> sender_sem(sem::in0_mcast_sender);
    Semaphore<> receiver_sem(sem::in0_mcast_receiver);
#ifdef IN0_MULTISHARD
    // A storage core that does no compute assembles its multi-shard block here: its in0 slot is
    // inside the multicast rectangle and receives the other senders' blocks.
    Scratchpad<uint32_t> stage(scratch::in0_stage);
#endif

    uint32_t l1_write_addr_in0;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

    // Raw address needed: the shard sits at the same L1 address on every core, so this core's base is also the
    // remote read address of the other storage cores' shards; it is a pad_last_ktile / local_l1_copy operand too.
    uint32_t local_read_addr = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(in0_shard.local_mem()));
    const uint32_t shard_base = local_read_addr;

    // Gather the shards_per_block shards of this sender's block from the storage cores (this one
    // included; the shard buffer sits at the same L1 address on every core) into dst_addr. The
    // reads are issued before the receiver handshake so they overlap it; the caller barriers.
    auto gather_block_shards = [&](uint32_t dst_addr) {
        const UnicastEndpoint src_core;
        for (uint32_t j = 0; j < shards_per_block; ++j) {
            const uint32_t src_idx = sender_id + j;
            for (uint32_t row = 0; row < in0_block_h; ++row) {
                noc.async_read(
                    src_core,
                    CoreLocalMem<uint32_t>(dst_addr + row * in0_block_row_bytes + j * in0_shard_width_bytes),
                    in0_shard_width_bytes,
                    {.noc_x = get_vararg(sender_noc_x_base + src_idx),
                     .noc_y = get_vararg(sender_noc_y_base + src_idx),
                     .addr = shard_base + row * in0_shard_width_bytes},
                    {});
            }
        }
    };

    if (worker_core_type == 1) {  // mcast sender + no compute
        if (!sends_blocks) {
            return;
        }

        for (uint32_t i = 0; i < num_blocks_per_shard; ++i) {
            const uint32_t block_id = (sender_block_id / shards_per_block) + i;

            // Operand 0
            l1_write_addr_in0 = dfb_in0.get_write_ptr();
            if (block_id % 2 != 0) {  // double buffer
                l1_write_addr_in0 += in0_block_size_bytes;
            }

            // copy start address of block, to be used for mcasting

#ifdef IN0_MULTISHARD
            // The gather overlaps the credit wait below.
            // Raw address needed: gather_block_shards and local_read_addr (see above) take raw L1 addresses.
            const uint32_t staging_addr =
                static_cast<uint32_t>(local_mem::detail::AddressAccess::address(stage.local_mem()));
            gather_block_shards(staging_addr);
            local_read_addr = staging_addr;
#endif

            // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
            sender_sem.wait(in0_mcast_num_dests);
            sender_sem.set(0);
#ifdef IN0_MULTISHARD
            noc.async_read_barrier();
#endif

            // Now we have the block in the buffer address, we can mcast to dests!

            // Zero out padded regions for tiles in the last K-column/row
            if constexpr (in0_last_ktile_w > 0) {
                if (is_last_ktile_padded && (i == num_blocks_per_shard - 1)) {
                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                        auto in0_last_ktile_ptr =
                            local_read_addr + h * source_row_bytes + (in0_block_w - 1) * in0_single_tile_size_bytes;
                        pad_last_ktile<in0_data_format, in0_last_ktile_w, get_arg(args::in0_tile_height)>(
                            in0_last_ktile_ptr);
                    }
                }
            }
            if constexpr (in0_last_ktile_h > 0) {
                if (is_last_ktile_padded && (i == num_blocks_per_shard - 1)) {
                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                        auto in0_last_ktile_ptr =
                            local_read_addr + (in0_block_h - 1) * source_row_bytes + w * in0_single_tile_size_bytes;
                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(in0_last_ktile_ptr);
                    }
                }
            }

#ifndef SKIP_MCAST
            // num_dests must not include source, since we are NOT really doing a local copy!
            multicast_block<false>(noc, local_read_addr, l1_write_addr_in0);
#endif

            receiver_sem.set_multicast(
                noc,
                in0_mcast_dest_noc_start_x,
                in0_mcast_dest_noc_start_y,
                in0_mcast_dest_noc_end_x,
                in0_mcast_dest_noc_end_y,
                in0_mcast_num_cores - 1);

            if constexpr (shards_per_block == 1) {
                local_read_addr += in0_block_row_bytes;
            }
        }

    } else if (worker_core_type == 2) {  // mcast sender + compute

        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = sender_core_of_block(block);

            dfb_in0.reserve_back(in0_block_num_tiles);
            // Set in0 semaphore value to INVALID
            receiver_sem.set(INVALID);

            if (sends_blocks && block_id == sender_id) {
                const uint32_t mcast_l1_write_addr_in0 = dfb_in0.get_write_ptr();
                // copy start address of block, to be used for mcasting

                if constexpr (shards_per_block > 1) {
                    // reserve_back above guarantees the compute has drained this slot; the previous
                    // multicast out of it was flushed before its VALID went out.
                    gather_block_shards(mcast_l1_write_addr_in0);
                    local_read_addr = mcast_l1_write_addr_in0;
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
                                local_read_addr + h * source_row_bytes + (in0_block_w - 1) * in0_single_tile_size_bytes;
                            pad_last_ktile<in0_data_format, in0_last_ktile_w, get_arg(args::in0_tile_height)>(
                                in0_last_ktile_ptr);
                        }
                    }
                }
                if constexpr (in0_last_ktile_h > 0) {
                    if (is_last_ktile_padded && (block == num_blocks - 1)) {
                        for (uint32_t w = 0; w < in0_block_w; ++w) {
                            auto in0_last_ktile_ptr =
                                local_read_addr + (in0_block_h - 1) * source_row_bytes + w * in0_single_tile_size_bytes;
                            pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(in0_last_ktile_ptr);
                        }
                    }
                }
#ifndef SKIP_MCAST
                if constexpr (shards_per_block > 1) {
                    // The gathered block already sits in this core's slot: multicast to the others only.
                    multicast_block<false>(noc, local_read_addr, mcast_l1_write_addr_in0);
                } else {
                    multicast_block<true>(noc, local_read_addr, mcast_l1_write_addr_in0);
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
                    local_read_addr += in0_block_row_bytes;
                }

            } else {
                // Atomic increment source core counter
                sender_sem.up(
                    noc, get_vararg(sender_noc_x_base + block_id), get_vararg(sender_noc_y_base + block_id), 1);
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
            sender_sem.up(noc, get_vararg(sender_noc_x_base + block_id), get_vararg(sender_noc_y_base + block_id), 1);
            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            receiver_sem.wait(VALID);

            dfb_in0.push_back(in0_block_num_tiles);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
