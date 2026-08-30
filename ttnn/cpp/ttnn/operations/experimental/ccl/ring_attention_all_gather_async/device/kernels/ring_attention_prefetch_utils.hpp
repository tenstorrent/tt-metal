// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/paged_kv_utils.hpp"

template <typename Accessor, typename Endpoint>
FORCE_INLINE void async_read_accessor_page(
    const Noc& noc, const Accessor& accessor, const Endpoint& dst, uint32_t page_bytes, uint32_t page_id) {
    noc.async_read(accessor, dst, page_bytes, {.page_id = page_id}, {});
}

template <typename ReaderType, typename Endpoint>
FORCE_INLINE void async_read_accessor_page(
    const Noc& noc,
    const PagedKVAccessor<ReaderType>& accessor,
    const Endpoint& dst,
    uint32_t page_bytes,
    uint32_t page_id) {
    accessor.async_read_page(noc, dst, page_id, page_bytes);
}

// Batch packetized DRAM reads into a CB while keeping multiple packets in flight.  The caller
// provides the source page mapping; this supports both the dense reader's contiguous two-page
// reads and the neighbor-halo reader's one-page reads without duplicating FIFO-wrap arithmetic.
template <
    uint32_t input_page_size,
    uint32_t packet_size_in_pages,
    uint32_t prefetch_packets,
    uint32_t contig_pages_advanced,
    typename Accessor,
    typename PageIdFn>
FORCE_INLINE void prefetch_batch_read_tiles(
    const Noc& noc,
    CircularBuffer& cb_output,
    uint32_t& tiles_read,
    uint32_t tiles_to_read,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    const Accessor& accessor,
    PageIdFn&& next_page_id) {
    constexpr uint32_t payload_size_bytes = input_page_size * contig_pages_advanced;
    while (tiles_read < tiles_to_read) {
        const uint32_t remaining_tiles = tiles_to_read - tiles_read;
        const uint32_t remaining_packets = (remaining_tiles + packet_size_in_pages - 1) / packet_size_in_pages;
        const uint32_t batch_packets = std::min(remaining_packets, prefetch_packets);
        const uint32_t batch_pages = batch_packets * packet_size_in_pages;

        cb_output.reserve_back(batch_pages);
        uint32_t l1_write_addr = cb_output.get_write_ptr();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            const uint32_t pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
            for (uint32_t page = 0; page < pages_to_read; page += contig_pages_advanced) {
                if (l1_write_addr >= cb_fifo_limit) {
                    l1_write_addr -= cb_fifo_size;
                }
                async_read_accessor_page(
                    noc, accessor, CoreLocalMem<uint8_t>(l1_write_addr), input_page_size, next_page_id(tiles_read));
                l1_write_addr += payload_size_bytes;
                tiles_read += contig_pages_advanced;
            }
            l1_write_addr += (packet_size_in_pages - pages_to_read) * input_page_size;
        }
        noc.async_read_barrier();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            cb_output.push_back(packet_size_in_pages);
        }
    }
}
