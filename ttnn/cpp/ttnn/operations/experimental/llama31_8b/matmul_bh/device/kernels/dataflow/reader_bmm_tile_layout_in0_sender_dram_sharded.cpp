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
#include "ttnn/operations/kernel_helper_functions/local_l1_copy.hpp"
#include "internal/local_mem/address_access.h"
FORCE_INLINE uint32_t l1_uncached(uint32_t addr) {
    return addr >= MEM_L1_UNCACHED_BASE ? addr : MEM_L1_UNCACHED_BASE + (addr - MEM_L1_BASE);
}
#endif

template <bool IncludeSource>
FORCE_INLINE void multicast_block(const Noc& noc, uint32_t src, uint32_t dst) {
    constexpr uint32_t tile_bytes = get_tile_size(dfb::in0);
    constexpr uint32_t row_bytes = get_arg(args::in0_block_w) * tile_bytes;
    constexpr uint32_t source_stride = get_arg(args::in0_shard_width_bytes);
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
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto in0_block_size_bytes = get_arg(args::in0_block_size_bytes);
    constexpr auto in0_last_ktile_w = get_arg(args::in0_last_ktile_w);
    constexpr auto in0_last_ktile_h = get_arg(args::in0_last_ktile_h);
    constexpr auto in0_mcast_num_dests = get_arg(args::in0_mcast_num_dests);
    constexpr auto in0_mcast_num_cores = get_arg(args::in0_mcast_num_cores);
    constexpr auto num_blocks = get_arg(args::num_blocks);
    constexpr auto in0_mcast_dest_noc_start_x = get_arg(args::in0_mcast_dest_noc_start_x);
    constexpr auto in0_mcast_dest_noc_start_y = get_arg(args::in0_mcast_dest_noc_start_y);
    constexpr auto in0_mcast_dest_noc_end_x = get_arg(args::in0_mcast_dest_noc_end_x);
    constexpr auto in0_mcast_dest_noc_end_y = get_arg(args::in0_mcast_dest_noc_end_y);
    constexpr auto num_storage_cores = get_arg(args::num_storage_cores);
    constexpr auto in0_block_w = get_arg(args::in0_block_w);
    constexpr auto in0_shard_width_bytes = get_arg(args::in0_shard_width_bytes);
    constexpr uint32_t in0_block_h = in0_block_num_tiles / in0_block_w;
    constexpr uint32_t in0_block_row_bytes = in0_block_w * get_tile_size(dfb::in0);
    constexpr uint32_t source_row_bytes = in0_shard_width_bytes;

    const uint32_t worker_core_type = get_arg(args::worker_core_type);
    if (worker_core_type == 0) {
        return;
    }
    const uint32_t sender_id = get_arg(args::sender_id);
    const bool in0_sender_slot = get_arg(args::in0_sender_slot) != 0;
    const bool is_last_ktile_padded = static_cast<bool>(get_arg(args::is_last_ktile_padded));

    constexpr uint32_t sender_noc_x_base = 0;
    constexpr uint32_t sender_noc_y_base = num_storage_cores;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb::in0);
    constexpr DataFormat in0_data_format = get_dataformat(dfb::in0);

    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    const LocalTensorAccessor<uint32_t> in0_shard(tensor::in0_shard);
    Semaphore<> sender_sem(sem::in0_mcast_sender);
    Semaphore<> receiver_sem(sem::in0_mcast_receiver);

    uint32_t l1_write_addr_in0;

    receiver_sem.set(VALID);

    uint32_t local_read_addr = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(in0_shard.local_mem()));

    if (worker_core_type == 1) {
        l1_write_addr_in0 = dfb_in0.get_write_ptr();
        if (in0_sender_slot) {
            l1_write_addr_in0 += in0_block_size_bytes;
        }

        sender_sem.wait(in0_mcast_num_dests);
        sender_sem.set(0);

        if constexpr (in0_last_ktile_w > 0) {
            if (is_last_ktile_padded) {
                for (uint32_t h = 0; h < in0_block_h; ++h) {
                    auto in0_last_ktile_ptr =
                        local_read_addr + h * source_row_bytes + (in0_block_w - 1) * in0_single_tile_size_bytes;
                    pad_last_ktile<in0_data_format, in0_last_ktile_w, get_arg(args::in0_tile_height)>(
                        in0_last_ktile_ptr);
                }
            }
        }
        if constexpr (in0_last_ktile_h > 0) {
            if (is_last_ktile_padded) {
                for (uint32_t w = 0; w < in0_block_w; ++w) {
                    auto in0_last_ktile_ptr =
                        local_read_addr + (in0_block_h - 1) * source_row_bytes + w * in0_single_tile_size_bytes;
                    pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(in0_last_ktile_ptr);
                }
            }
        }

#ifndef SKIP_MCAST
        multicast_block<false>(noc, local_read_addr, l1_write_addr_in0);
#endif

        receiver_sem.set_multicast(
            noc,
            in0_mcast_dest_noc_start_x,
            in0_mcast_dest_noc_start_y,
            in0_mcast_dest_noc_end_x,
            in0_mcast_dest_noc_end_y,
            in0_mcast_num_cores - 1);
    } else if (worker_core_type == 2) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in0.reserve_back(in0_block_num_tiles);
            receiver_sem.set(INVALID);

            if (block == sender_id) {
                const uint32_t mcast_l1_write_addr_in0 = dfb_in0.get_write_ptr();

                sender_sem.wait(in0_mcast_num_dests - 1);
                sender_sem.set(0);

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
                multicast_block<true>(noc, local_read_addr, mcast_l1_write_addr_in0);
#endif
                receiver_sem.set(VALID);
                if constexpr (in0_mcast_num_cores > 1) {
                    receiver_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                        noc,
                        in0_mcast_dest_noc_start_x,
                        in0_mcast_dest_noc_start_y,
                        in0_mcast_dest_noc_end_x,
                        in0_mcast_dest_noc_end_y,
                        in0_mcast_num_cores);
                    noc.async_writes_flushed();
                }

                local_read_addr += in0_block_row_bytes;

            } else {
                sender_sem.up(noc, get_vararg(sender_noc_x_base + block), get_vararg(sender_noc_y_base + block), 1);
            }

            receiver_sem.wait(VALID);
            dfb_in0.push_back(in0_block_num_tiles);
        }
    } else {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in0.reserve_back(in0_block_num_tiles);

            receiver_sem.set(INVALID);
            sender_sem.up(noc, get_vararg(sender_noc_x_base + block), get_vararg(sender_noc_y_base + block), 1);
            receiver_sem.wait(VALID);

            dfb_in0.push_back(in0_block_num_tiles);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
