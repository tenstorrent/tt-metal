// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

// Batch packetized DRAM reads into a CB while keeping multiple packets in flight. The caller
// provides the source page mapping; this supports both the dense and neighbor-halo readers
// without duplicating FIFO-wrap arithmetic.
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
                noc.async_read(
                    accessor,
                    CoreLocalMem<uint8_t>(l1_write_addr),
                    input_page_size,
                    {.page_id = next_page_id(tiles_read)},
                    {});
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

// Read an arbitrary packet sequence while preserving one prefetch window across logical slice boundaries.
// next_packet returns the first logical page id and writes the number of physically contiguous pages in that packet.
template <
    uint32_t input_page_size,
    uint32_t packet_size_in_pages,
    uint32_t prefetch_packets,
    typename Accessor,
    typename NextPacketFn>
FORCE_INLINE void prefetch_batch_read_physically_contiguous_packets(
    const Noc& noc,
    CircularBuffer& cb_output,
    uint32_t total_packets,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    const Accessor& accessor,
    NextPacketFn&& next_packet) {
    uint32_t packets_read = 0;
    while (packets_read < total_packets) {
        const uint32_t batch_packets = std::min(total_packets - packets_read, prefetch_packets);
        cb_output.reserve_back(batch_packets * packet_size_in_pages);
        uint32_t l1_write_addr = cb_output.get_write_ptr();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            if (l1_write_addr >= cb_fifo_limit) {
                l1_write_addr -= cb_fifo_size;
            }
            uint32_t pages_to_read = 0;
            const uint32_t first_page_id = next_packet(pages_to_read);
            const uint64_t first_noc_addr = accessor.get_noc_addr(first_page_id, 0, noc.get_noc_id());
            noc.async_read(
                tensor_accessor::Page(first_noc_addr, 0),
                CoreLocalMem<uint8_t>(l1_write_addr),
                pages_to_read * input_page_size,
                {},
                {});
            l1_write_addr += packet_size_in_pages * input_page_size;
        }
        packets_read += batch_packets;
        noc.async_read_barrier();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            cb_output.push_back(packet_size_in_pages);
        }
    }
}

// Read an arbitrary packet sequence through the tensor accessor. Unlike the physically-contiguous
// variant above, each page may have an independent logical page id, so this is valid for sharded inputs.
template <
    uint32_t input_page_size,
    uint32_t packet_size_in_pages,
    uint32_t prefetch_packets,
    typename Accessor,
    typename NextPacketFn,
    typename PacketPageIdFn>
FORCE_INLINE void prefetch_batch_read_packets(
    const Noc& noc,
    CircularBuffer& cb_output,
    uint32_t total_packets,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    const Accessor& accessor,
    NextPacketFn&& next_packet,
    PacketPageIdFn&& packet_page_id) {
    uint32_t packets_read = 0;
    while (packets_read < total_packets) {
        const uint32_t batch_packets = std::min(total_packets - packets_read, prefetch_packets);
        cb_output.reserve_back(batch_packets * packet_size_in_pages);
        uint32_t l1_write_addr = cb_output.get_write_ptr();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            uint32_t pages_to_read = 0;
            const uint32_t first_page_id = next_packet(pages_to_read);
            for (uint32_t page = 0; page < pages_to_read; ++page) {
                if (l1_write_addr >= cb_fifo_limit) {
                    l1_write_addr -= cb_fifo_size;
                }
                noc.async_read(
                    accessor,
                    CoreLocalMem<uint8_t>(l1_write_addr),
                    input_page_size,
                    {.page_id = packet_page_id(first_page_id, page)},
                    {});
                l1_write_addr += input_page_size;
            }
            l1_write_addr += (packet_size_in_pages - pages_to_read) * input_page_size;
        }
        packets_read += batch_packets;
        noc.async_read_barrier();
        for (uint32_t packet = 0; packet < batch_packets; ++packet) {
            cb_output.push_back(packet_size_in_pages);
        }
    }
}

// Read a bank-owned logical page sequence as physically contiguous packet-sized NOC transactions.
// In an interleaved DRAM buffer, logical pages separated by the DRAM-bank count are adjacent within
// one bank. The caller is responsible for supplying exactly that mapping.
template <
    uint32_t input_page_size,
    uint32_t packet_size_in_pages,
    uint32_t prefetch_packets,
    typename Accessor,
    typename PageIdFn>
FORCE_INLINE void prefetch_batch_read_physically_contiguous_tiles(
    const Noc& noc,
    CircularBuffer& cb_output,
    uint32_t& tiles_read,
    uint32_t tiles_to_read,
    uint32_t cb_fifo_limit,
    uint32_t cb_fifo_size,
    const Accessor& accessor,
    PageIdFn&& next_page_id) {
    const uint32_t total_packets = (tiles_to_read - tiles_read + packet_size_in_pages - 1) / packet_size_in_pages;
    prefetch_batch_read_physically_contiguous_packets<input_page_size, packet_size_in_pages, prefetch_packets>(
        noc, cb_output, total_packets, cb_fifo_limit, cb_fifo_size, accessor, [&](uint32_t& pages_to_read) {
            pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
            const uint32_t first_page_id = next_page_id(tiles_read);
            tiles_read += pages_to_read;
            return first_page_id;
        });
}
