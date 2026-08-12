// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;

using PacketHeaderPtr = decltype(PacketHeaderPool::allocate_header());

// Helper class to polymorphically manage mux eager staging
template <class ConnectionType>
struct MuxFlusher {
public:
    MuxFlusher(ConnectionType&) {};
    void flush() {};
};

template <>
struct MuxFlusher<tt::tt_fabric::FabricMuxV2Sender<true, 0>> {
    tt::tt_fabric::FabricMuxV2Sender<true, 0>& m_mux_sender;
    bool m_flushed;

public:
    MuxFlusher(tt::tt_fabric::FabricMuxV2Sender<true, 0>& mux_sender) : m_mux_sender(mux_sender), m_flushed(false) {};
    void flush() {
        if (!m_flushed) {
            m_mux_sender.flush</*blocking=*/true>();
            m_flushed = true;
        }
    };
};

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_named_compile_time_arg_val("my_chip_id");
constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
constexpr uint32_t cb_compute_output_id = get_named_compile_time_arg_val("cb_compute_output_id");
constexpr uint32_t cb_reader_output_id = get_named_compile_time_arg_val("cb_reader_output_id");
constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
constexpr uint32_t num_tiles_to_write_per_packet = get_named_compile_time_arg_val("num_tiles_to_write_per_packet");
constexpr uint32_t output_batch_num_pages = get_named_compile_time_arg_val("output_batch_num_pages");
constexpr uint32_t input_channel_num_pages = get_named_compile_time_arg_val("input_channel_num_pages");
constexpr uint32_t output_channel_num_pages = get_named_compile_time_arg_val("output_channel_num_pages");
constexpr uint32_t input_tensor_B = get_named_compile_time_arg_val("input_tensor_B");
constexpr uint32_t input_tensor_Wt = get_named_compile_time_arg_val("input_tensor_Wt");
constexpr uint32_t slice_C = get_named_compile_time_arg_val("slice_C");
constexpr uint32_t slice_Ht = get_named_compile_time_arg_val("slice_Ht");
constexpr uint32_t slice_Wt = get_named_compile_time_arg_val("slice_Wt");
constexpr uint32_t dim = get_named_compile_time_arg_val("dim");
// Selects how partial sums are staged on the receiving device. See IntermSink.
constexpr bool contiguous_interm = get_named_compile_time_arg_val("contiguous_interm") != 0;
// Chunk-paged layout only: chunks per (slice, channel) in the staging buffers, and the max number
// of tiles that fit in one fabric packet (used to split a chunk into contiguous unicast packets).
constexpr uint32_t chunks_per_channel = get_named_compile_time_arg_val("chunks_per_channel");
constexpr uint32_t interm_tiles_per_packet = get_named_compile_time_arg_val("interm_tiles_per_packet");
// The V2 fabric mux client (FabricMuxV2Sender) is built entirely from runtime args, so there are no
// worker-side mux compile-time args in either the mux or the direct-fabric path.
constexpr uint32_t num_ct_args = 0;

// Routing info uses positional args after fabric mux args
constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<num_ct_args>();
constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        num_ct_args + ccl_routing_utils::num_line_unicast_args>();

constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<
        num_ct_args + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        num_ct_args + 2 * ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();

// Tile id of the first tile of `slice_idx` in a tensor laid out like the input tensor.
FORCE_INLINE uint32_t slice_base_tile_id(uint32_t slice_idx) {
    if constexpr (dim == 3) {
        return slice_idx * slice_Wt;
    } else if constexpr (dim == 2) {
        return slice_idx * slice_Ht * slice_Wt;
    } else if constexpr (dim == 1) {
        return slice_idx * slice_C * slice_Ht * slice_Wt;
    } else {
        ASSERT(false);
        return 0;
    }
}

// The tensors a staging strategy may write to. Which ones it actually touches depends on the
// staging layout; see IntermSink.
template <typename IntermAcc, typename OutputAcc, typename PenultIntermAcc>
struct IntermTensors {
    const IntermAcc& interm;
    const OutputAcc& output;
    const PenultIntermAcc& penult_interm;
};
// convenience deduction guide
template <typename IntermAcc, typename OutputAcc, typename PenultIntermAcc>
IntermTensors(const IntermAcc&, const OutputAcc&, const PenultIntermAcc&)
    -> IntermTensors<IntermAcc, OutputAcc, PenultIntermAcc>;

// Stages one chunk of partial sums, either on the next device over the fabric or locally.
//
// Two staging layouts are supported, selected by the `contiguous_interm` compile-time arg. Both
// specializations own the packet headers for their remote writes and all destination addressing,
// and expose the same hooks, called from the same places in kernel_main:
//
//   set_packet_header_states(route)    once, before the first send
//   begin_iteration(b, slice_idx)      once per ring iteration
//   begin_channel(c)                   once per channel of the slice
//   skip_chunk(tiles)                  a chunk this worker/direction does not own
//   write_chunk_remote(...)            send one chunk to the next device; returns true if this
//                                      chunk's semaphore increment was fused onto a data packet
//   write_chunk_local(...)             last iteration: write one chunk to the local output tensor
//
// IntermSink<false> - tiled layout. The remote intermediate mirrors the input tensor's tiled,
//   row-strided addressing (one tile per page), so a packet covers up to
//   num_tiles_to_write_per_packet non-adjacent destinations and must be sent as a scatter write.
//   The 2nd-last iteration's contribution is scattered straight into the remote output tensor.
//
// IntermSink<true> - chunk-paged layout. The remote intermediate (and a dedicated penult intermediate
//   for the 2nd-last iteration's contribution) store one whole chunk per page, so a chunk's tiles
//   are contiguous at the destination and each packet is a plain unicast write. See
//   rs-contiguous-interm-design.
template <bool Contiguous>
struct IntermSink;

template <>
struct IntermSink</*Contiguous=*/false> {
    // Per-worker starting offsets into the slice (workers split the slice by row/column).
    const uint32_t start_pages_read_in_row;
    const uint32_t start_row_offset;
    const uint32_t start_tiles_read;

    uint32_t interm_slice_base = 0;  // first interm tile of the current slice
    uint32_t output_batch_base = 0;  // first output tile of the current batch
    uint32_t interm_tile_id_start = 0;
    uint32_t output_tile_id_start = 0;
    uint32_t pages_read_in_row = 0;
    uint32_t row_offset = 0;
    uint32_t output_tiles_read = 0;

    static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported (scatter write maximum)");
    uint64_t remote_noc_addrs[4] = {0, 0, 0, 0};
    uint16_t chunk_sizes[3] = {page_size, page_size, page_size};

    PacketHeaderPtr pkt_scatter_hdr = nullptr;
    PacketHeaderPtr pkt_unicast_hdr = nullptr;
    // Fused write + atomic-inc headers, used to fold a chunk's semaphore increment into its final
    // data packet (unicast for a 1-tile tail, scatter for a 2-tile tail).
    PacketHeaderPtr pkt_hdr_fused_unicast = nullptr;
    PacketHeaderPtr pkt_hdr_fused_scatter = nullptr;

    IntermSink(uint32_t pages_read_in_row_start, uint32_t row_offset_start, uint32_t tiles_read_start) :
        start_pages_read_in_row(pages_read_in_row_start),
        start_row_offset(row_offset_start),
        start_tiles_read(tiles_read_start) {}

    void set_packet_header_states(const ccl_routing_utils::line_unicast_route_info_t& unicast_route_info) {
        pkt_scatter_hdr = PacketHeaderPool::allocate_header();
        pkt_unicast_hdr = PacketHeaderPool::allocate_header();
        pkt_hdr_fused_unicast = PacketHeaderPool::allocate_header();
        pkt_hdr_fused_scatter = PacketHeaderPool::allocate_header();
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fused_unicast, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fused_scatter, unicast_route_info);

        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            NocUnicastScatterCommandHeader(remote_noc_addrs, chunk_sizes, num_tiles_to_write_per_packet),
            page_size * num_tiles_to_write_per_packet);

        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

        // Fused-packet state: payload size, increment value (1) and flush are constant across the
        // run; only the write and semaphore destination addresses are patched per packet.
        fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
            UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
            UnicastFusedAtomicIncUpdateMask::Flush>(
            pkt_hdr_fused_unicast,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                0,                          // write dst (patched per packet)
                0,                          // semaphore dst (patched per packet)
                static_cast<uint32_t>(1)},  // increment 1
            page_size);

        fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state<
            UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize |
            UnicastFusedScatterWriteAtomicIncUpdateMask::WriteChunkSizes |
            UnicastFusedScatterWriteAtomicIncUpdateMask::Val | UnicastFusedScatterWriteAtomicIncUpdateMask::Flush>(
            pkt_hdr_fused_scatter,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {0, 0},                              // write dsts (patched per packet)
                0,                                   // semaphore dst (patched per packet)
                {static_cast<uint16_t>(page_size)},  // first chunk size (second is implicit)
                static_cast<uint16_t>(1)},           // increment 1
            static_cast<uint16_t>(page_size * 2));
    }

    void begin_iteration(uint32_t b, uint32_t slice_idx) {
        interm_slice_base = slice_base_tile_id(slice_idx);
        output_batch_base = b * output_batch_num_pages;
    }

    void begin_channel(uint32_t c) {
        interm_tile_id_start = interm_slice_base + c * input_channel_num_pages;
        output_tile_id_start = output_batch_base + c * output_channel_num_pages;
        pages_read_in_row = start_pages_read_in_row;
        row_offset = start_row_offset;
        output_tiles_read = start_tiles_read;
    }

    void skip_chunk(uint32_t tiles) {
        for (uint32_t k = 0; k < tiles; ++k) {
            next_interm_tile_id();
            next_output_tile_id();
        }
    }

    template <typename Connection, typename Flusher, typename Tensors>
    bool write_chunk_remote(
        Connection* connection,
        Flusher& mux_flusher,
        const Noc& noc,
        const Tensors& tensors,
        size_t l1_read_addr,
        uint32_t /*tiles_read*/,
        uint32_t tiles_to_read,
        bool write_to_interm,
        bool fuse_seminc,
        uint64_t sem_noc_addr) {
        bool seminc_fused = false;
        for (uint32_t j = 0; j < tiles_to_read; j += num_tiles_to_write_per_packet) {
            const uint32_t tiles_in_packet = std::min(tiles_to_read - j, num_tiles_to_write_per_packet);

            for (uint32_t k = 0; k < tiles_in_packet; ++k) {
                // Advanced for every tile, whichever tensor is targeted, to keep both counters
                // aligned with the slice.
                const uint32_t interm_tile_id = next_interm_tile_id();
                const uint32_t output_tile_id = next_output_tile_id();
                remote_noc_addrs[k] =
                    write_to_interm
                        ? tt::tt_fabric::linear::addrgen_detail::get_noc_address(tensors.interm, interm_tile_id, 0)
                        : tt::tt_fabric::linear::addrgen_detail::get_noc_address(tensors.output, output_tile_id, 0);
            }

            const bool last_packet = (j + num_tiles_to_write_per_packet >= tiles_to_read);
            seminc_fused |=
                send_write_packet(connection, l1_read_addr, tiles_in_packet, fuse_seminc && last_packet, sem_noc_addr);
            mux_flusher.flush();
            noc.async_writes_flushed();
            l1_read_addr += page_size * tiles_in_packet;
        }
        return seminc_fused;
    }

    template <typename Tensors>
    void write_chunk_local(
        const Noc& noc, const Tensors& tensors, CircularBuffer& cb_out, uint32_t tiles_to_read, bool write_to_interm) {
        size_t l1_read_offset = 0;
        for (uint32_t j = 0; j < tiles_to_read; ++j) {
            const uint32_t interm_tile_id = next_interm_tile_id();
            const uint32_t output_tile_id = next_output_tile_id();
            if (write_to_interm) {
                noc.async_write(
                    cb_out, tensors.interm, page_size, {.offset_bytes = l1_read_offset}, {.page_id = interm_tile_id});
            } else {
                noc.async_write(
                    cb_out, tensors.output, page_size, {.offset_bytes = l1_read_offset}, {.page_id = output_tile_id});
            }
            l1_read_offset += page_size;
        }
    }

private:
    uint32_t next_interm_tile_id() {
        uint32_t tile_id = interm_tile_id_start + row_offset + pages_read_in_row;
        ++pages_read_in_row;
        if (pages_read_in_row == slice_Wt) {
            row_offset += input_tensor_Wt;
            pages_read_in_row -= slice_Wt;
        }
        return tile_id;
    }

    uint32_t next_output_tile_id() { return output_tile_id_start + (output_tiles_read++); }

    // Write one packet worth of tiles (addresses staged in remote_noc_addrs) to the remote tensor.
    // When fuse_seminc is set, this chunk's semaphore increment is folded onto the packet and the
    // function returns true. The fused fabric ops carry at most a 2-tile scatter write plus the
    // semaphore chunk, so packets wider than 2 tiles cannot fuse (caller falls back to send_seminc).
    template <typename Connection>
    bool send_write_packet(
        Connection* connection, size_t l1_read_addr, uint32_t num_tiles, bool fuse_seminc, uint64_t sem_noc_addr) {
        if (fuse_seminc && num_tiles <= 2) {
            if (num_tiles == 2) {
                fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state<
                    UnicastFusedScatterWriteAtomicIncUpdateMask::WriteDstAddrs |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::SemaphoreDstAddr>(
                    connection,
                    pkt_hdr_fused_scatter,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                        {remote_noc_addrs[0], remote_noc_addrs[1]},
                        sem_noc_addr,
                        {static_cast<uint16_t>(page_size)},
                        static_cast<uint16_t>(1)});
            } else {
                fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                    UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                    connection,
                    pkt_hdr_fused_unicast,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        remote_noc_addrs[0], sem_noc_addr, static_cast<uint32_t>(1)});
            }
            return true;
        }
        if (num_tiles > 1) {
            fabric_unicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                UnicastScatterWriteUpdateMask::PayloadSize>(
                connection,
                pkt_scatter_hdr,
                l1_read_addr,
                NocUnicastScatterCommandHeader(remote_noc_addrs, chunk_sizes, num_tiles),
                page_size * num_tiles);
        } else {
            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                connection, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{remote_noc_addrs[0]});
        }
        return false;
    }
};

template <>
struct IntermSink</*Contiguous=*/true> {
    const uint32_t start_tiles_read;

    uint32_t output_batch_base = 0;
    uint32_t output_tile_id_start = 0;
    uint32_t output_tiles_read = 0;

    uint32_t interm_slice_chunk_base = 0;
    uint32_t interm_channel_chunk_base = 0;
    uint32_t penult_interm_channel_chunk_base = 0;

    // Headers for contiguous writes to the chunk-paged staging buffers (the main intermediate and the
    // penult intermediate region both use these). Their payload size is patched per packet.
    PacketHeaderPtr pkt_interm_unicast_hdr = nullptr;
    PacketHeaderPtr pkt_interm_fused_hdr = nullptr;

    // The per-worker row/column starts are meaningless here: chunk pages are addressed straight
    // from tiles_read, so only the (tiled) local output tensor needs a counter.
    IntermSink(uint32_t, uint32_t, uint32_t tiles_read_start) : start_tiles_read(tiles_read_start) {}

    void set_packet_header_states(const ccl_routing_utils::line_unicast_route_info_t& unicast_route_info) {
        pkt_interm_unicast_hdr = PacketHeaderPool::allocate_header();
        pkt_interm_fused_hdr = PacketHeaderPool::allocate_header();
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_interm_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_interm_fused_hdr, unicast_route_info);

        // The destination address and payload size are patched per packet, so PayloadSize is part
        // of the per-packet update mask too. Only the route and (for the fused header) the
        // increment value / flush are fixed here.
        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_interm_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
            UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
            UnicastFusedAtomicIncUpdateMask::Flush>(
            pkt_interm_fused_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                0,                          // write dst (patched per packet)
                0,                          // semaphore dst (patched per packet)
                static_cast<uint32_t>(1)},  // increment 1
            page_size);
    }

    void begin_iteration(uint32_t b, uint32_t slice_idx) {
        interm_slice_chunk_base = slice_idx * slice_C * chunks_per_channel;
        output_batch_base = b * output_batch_num_pages;
    }

    void begin_channel(uint32_t c) {
        interm_channel_chunk_base = interm_slice_chunk_base + c * chunks_per_channel;
        // The penult intermediate has no slice_idx axis (each device receives exactly one such
        // contribution, from exactly one neighbor, at exactly one iteration).
        penult_interm_channel_chunk_base = c * chunks_per_channel;
        output_tile_id_start = output_batch_base + c * output_channel_num_pages;
        output_tiles_read = start_tiles_read;
    }

    // Only the tiled output tensor needs its counter kept aligned; the staging buffers are
    // addressed statelessly from tiles_read.
    void skip_chunk(uint32_t tiles) {
        for (uint32_t k = 0; k < tiles; ++k) {
            next_output_tile_id();
        }
    }

    // Both write_to_interm (mid-ring hops) and the penult intermediate target a chunk-paged
    // staging buffer with the same packetization: one page holds the whole chunk, so each fabric
    // packet is a single contiguous unicast write. The packet that reaches
    // chunks_per_sync fuses the semaphore increment onto itself.
    template <typename Connection, typename Flusher, typename Tensors>
    bool write_chunk_remote(
        Connection* connection,
        Flusher& mux_flusher,
        const Noc& noc,
        const Tensors& tensors,
        size_t l1_read_addr,
        uint32_t tiles_read,
        uint32_t tiles_to_read,
        bool write_to_interm,
        bool fuse_seminc,
        uint64_t sem_noc_addr) {
        // The 2nd-last iteration stages this direction's contribution in the dedicated penult
        // intermediate The receiver's final iteration reads it back as the 3rd term of its local 3-way reduce.
        const uint32_t channel_chunk_base =
            write_to_interm ? interm_channel_chunk_base : penult_interm_channel_chunk_base;
        const uint32_t chunk_page_id = channel_chunk_base + tiles_read / tile_granularity;
        const uint32_t in_chunk_offset = (tiles_read % tile_granularity) * page_size;

        bool seminc_fused = false;
        for (uint32_t j = 0; j < tiles_to_read; j += interm_tiles_per_packet) {
            const uint32_t tiles_in_packet = std::min(tiles_to_read - j, interm_tiles_per_packet);
            const uint16_t payload_bytes = static_cast<uint16_t>(tiles_in_packet * page_size);
            const uint64_t dst_noc_addr =
                write_to_interm ? tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                      tensors.interm, chunk_page_id, in_chunk_offset + j * page_size)
                                : tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                      tensors.penult_interm, chunk_page_id, in_chunk_offset + j * page_size);
            const bool last_packet = (j + interm_tiles_per_packet >= tiles_to_read);
            if (fuse_seminc && last_packet) {
                fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                    UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                    UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                    connection,
                    pkt_interm_fused_hdr,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        dst_noc_addr, sem_noc_addr, static_cast<uint32_t>(1)},
                    payload_bytes);
                seminc_fused = true;
            } else {
                fabric_unicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    connection,
                    pkt_interm_unicast_hdr,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr},
                    payload_bytes);
            }

            mux_flusher.flush();

            noc.async_writes_flushed();
            l1_read_addr += payload_bytes;
        }
        return seminc_fused;
    }

    // The last iteration is the only one that writes locally, and it never targets the
    // intermediate, so write_to_interm is always false here.
    template <typename Tensors>
    void write_chunk_local(
        const Noc& noc,
        const Tensors& tensors,
        CircularBuffer& cb_out,
        uint32_t tiles_to_read,
        [[maybe_unused]] bool write_to_interm) {
        ASSERT(!write_to_interm);
        size_t l1_read_offset = 0;
        for (uint32_t j = 0; j < tiles_to_read; ++j) {
            noc.async_write(
                cb_out,
                tensors.output,
                page_size,
                {.offset_bytes = l1_read_offset},
                {.page_id = next_output_tile_id()});
            l1_read_offset += page_size;
        }
    }

private:
    uint32_t next_output_tile_id() { return output_tile_id_start + (output_tiles_read++); }
};

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t interm_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t this_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t this_core_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 1 is forward, 0 is backward
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    // Chunk-paged layout only: staging buffer for the 2nd-last iteration's direct-to-remote
    // contribution. The tiled layout scatter-writes that contribution into the remote output tensor
    // instead and leaves this address at 0.
    address_t penult_intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
#ifdef USE_WORKER_MUX
    size_t mux_arg_idx = arg_idx;
    auto mux_sender = tt::tt_fabric::FabricMuxV2Sender</*EAGER_STAGING=*/true>::build_from_args(mux_arg_idx);
    arg_idx = mux_arg_idx;
#endif

    const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

    constexpr auto interm_tensor_args = TensorAccessorArgs<ct_idx>();
    auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);

    constexpr auto output_tensor_args = TensorAccessorArgs<interm_tensor_args.next_compile_time_args_offset()>();
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    constexpr auto penult_intermediate_tensor_args =
        TensorAccessorArgs<output_tensor_args.next_compile_time_args_offset()>();
    auto penult_intermediate_tensor_accessor =
        TensorAccessor(penult_intermediate_tensor_args, penult_intermediate_tensor_address);

    IntermTensors interm_tensors{interm_tensor_accessor, output_tensor_accessor, penult_intermediate_tensor_accessor};
    IntermSink<contiguous_interm> interm_sink(start_pages_read_in_row, start_row_offset, start_tiles_read);

#ifndef USE_WORKER_MUX
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
#endif
    // pre-populate packet headers (the data-path headers belong to interm_sink)
    auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();
    auto pkt_hdr_mcastseminc = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
    interm_sink.set_packet_header_states(unicast_route_info);

#ifdef USE_WORKER_MUX
    // Blocking open: waits for the mux to be READY, then requests the connection.
    mux_sender.open();
    auto* fabric_direction_connection = &mux_sender;
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    auto* fabric_direction_connection =
        direction ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection();
#endif

    MuxFlusher mf(*fabric_direction_connection);

    Noc noc_obj;
    CircularBuffer cb_compute_output(cb_compute_output_id);
    CircularBuffer cb_reader_output(cb_reader_output_id);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_mcastseminc,
        static_cast<uint8_t>(multicast_route_info.start_distance_in_hops),
        static_cast<uint8_t>(multicast_route_info.range_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_seminc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    if (use_barrier_sem) {
        // Use neighbor unicast instead of multicast to support reshaped 'logical linear' mesh devices
        uint64_t opposite_barrier_sem_noc_addr = safe_get_noc_addr(opposite_core_x, opposite_core_y, barrier_sem, 0);
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_seminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_barrier_sem_noc_addr, 0});

        // we need to complete the fabric mux connection init immediately after any fabric transaction
        mf.flush();

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    // Relevant for 2nd-last iter:
    // In 2nd-last iter we send the full tensor slice. But in preparation for the last iter where each dir
    // processes half tensor slice, in 2nd-last iter we send sem increments to both forward and backward workers.
    // For example, if we send 2 even chunks and 2 odd chunks, we need to send 2 sem incrs to forward worker
    // and 2 sem incrs to backward worker.
    uint64_t this_core_sem_noc_addr = safe_get_noc_addr(this_core_x, this_core_y, out_ready_sem, 0);
    uint64_t opposite_core_sem_noc_addr = safe_get_noc_addr(opposite_core_x, opposite_core_y, out_ready_sem, 0);
    uint64_t even_core_sem_noc_addr = direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;
    uint64_t odd_core_sem_noc_addr = !direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;

    // Emit a standalone atomic increment to a remote worker's out_ready_sem.
    auto send_seminc = [&](uint64_t sem_noc_addr) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_seminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
        mf.flush();
        noc_obj.async_writes_flushed();
    };

    // The batch_ready_sem is incremented once per batch by the opposite-direction neighbour;
    // instead of resetting it with set(0) after every batch
    uint32_t batch_ready_target = 0;

    for (uint32_t b = 0; b < input_tensor_B; ++b) {
        constexpr uint32_t ring_size_by_2 = ring_size / 2;
        int slice_idx = my_chip_id + ring_size_by_2;  // start with slice belonging to device half-way across in ring
        uint32_t num_iters = ring_size_by_2 + 1;
        for (uint32_t i = 0; i < num_iters; ++i) {
            // State machine for control variables
            bool even_chunks, odd_chunks, reduce_even_chunks, reduce_odd_chunks, write_to_remote, write_to_interm,
                separate_even_odd_sems;
            if (i == 0) {
                even_chunks = direction;     // process the even chunks (half the tensor slice)
                odd_chunks = !direction;     // process the odd chunks (other half of tensor slice)
                reduce_even_chunks = false;  // grab output from compute or reader
                reduce_odd_chunks = false;   // grab output from compute or reader
                write_to_remote = true;      // write to remote device or local device
                write_to_interm = true;      // write to interm_tensor or output_tensor
                separate_even_odd_sems =
                    false;  // 2nd-last iter: send sem incrs separately for even & odd chunks to diff workers
            } else if (i == ring_size_by_2) {
                even_chunks = direction;
                odd_chunks = !direction;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                write_to_remote = false;
                write_to_interm = false;
                separate_even_odd_sems = false;
            } else if (i == 1 || i == ring_size_by_2 - 1) {  // these two cases can coincide (ring_size = 4)
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = (i == 1) ? direction : even_chunks;
                reduce_odd_chunks = (i == 1) ? !direction : odd_chunks;
                write_to_remote = true;
                write_to_interm = (i == ring_size_by_2 - 1) ? direction : true;
                separate_even_odd_sems = (i == ring_size_by_2 - 1);
            } else {
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                write_to_remote = true;
                write_to_interm = true;
                separate_even_odd_sems = false;
            }

            // below code does 'slice_idx = slice_idx % ring_size'
            if (slice_idx < 0) {
                slice_idx += ring_size;
            } else if (slice_idx >= (int)ring_size) {
                slice_idx = (uint32_t)slice_idx - ring_size;
            }

            interm_sink.begin_iteration(b, slice_idx);

            uint32_t chunk_count = 0;
            uint32_t even_chunk_count = 0;
            uint32_t odd_chunk_count = 0;
            for (uint32_t c = 0; c < slice_C; ++c) {
                // reset addr counters
                interm_sink.begin_channel(c);
                uint32_t tiles_read = start_tiles_read;
                uint32_t total_tiles_to_read = start_tiles_to_read;

                while (tiles_read < total_tiles_to_read) {
                    const auto [is_even_chunk, tiles_to_read] =
                        reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);

                    if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                        // Skip this chunk
                        tiles_read += tiles_to_read;
                        interm_sink.skip_chunk(tiles_to_read);
                    } else {
                        const bool reduce_interm =
                            (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);
                        CircularBuffer& cb_out =
                            reduce_interm ? cb_compute_output : cb_reader_output;  // from compute or reader

                        if (write_to_remote) {
                            // Pick the semaphore this chunk signals and the counter that paces it. In
                            // separate-sem mode even/odd chunks signal different workers; otherwise every
                            // chunk signals this worker's peer. The counter is advanced after the writes, so
                            // fuse_seminc predicts against counter + 1: true means this chunk's final packet
                            // reaches chunks_per_sync and can carry the increment itself.
                            uint32_t& sync_counter = separate_even_odd_sems
                                                         ? (is_even_chunk ? even_chunk_count : odd_chunk_count)
                                                         : chunk_count;
                            const uint64_t sem_noc_addr =
                                separate_even_odd_sems
                                    ? (is_even_chunk ? even_core_sem_noc_addr : odd_core_sem_noc_addr)
                                    : this_core_sem_noc_addr;
                            const bool fuse_seminc = (sync_counter + 1 == chunks_per_sync);

                            // Write tiles to remote tensor over Fabric
                            cb_out.wait_front(tile_granularity);
                            const bool seminc_fused = interm_sink.write_chunk_remote(
                                fabric_direction_connection,
                                mf,
                                noc_obj,
                                interm_tensors,
                                cb_out.get_read_ptr(),
                                tiles_read,
                                tiles_to_read,
                                write_to_interm,
                                fuse_seminc,
                                sem_noc_addr);
                            tiles_read += tiles_to_read;
                            cb_out.pop_front(tile_granularity);

                            // Advance this chunk's sync counter; emit the increment now unless it was already
                            // fused onto the final data packet above.
                            if (++sync_counter == chunks_per_sync) {
                                sync_counter = 0;
                                if (!seminc_fused) {
                                    send_seminc(sem_noc_addr);
                                }
                            }
                        } else {
                            // Write tiles to local tensor
                            cb_out.wait_front(tile_granularity);
                            interm_sink.write_chunk_local(
                                noc_obj, interm_tensors, cb_out, tiles_to_read, write_to_interm);
                            tiles_read += tiles_to_read;
                            noc_obj.async_write_barrier();
                            cb_out.pop_front(tile_granularity);
                        }  // if remote or local
                    }  // if skip or process
                }  // while total_tiles_to_read
            }  // for slice_C

            // Flush any residual chunks whose counter never reached chunks_per_sync inside the loop.
            if (write_to_remote) {
                if (separate_even_odd_sems) {
                    if (even_chunks && even_chunk_count != 0) {
                        send_seminc(even_core_sem_noc_addr);
                    }
                    if (odd_chunks && odd_chunk_count != 0) {
                        send_seminc(odd_core_sem_noc_addr);
                    }
                } else {
                    if (chunk_count != 0) {
                        send_seminc(this_core_sem_noc_addr);
                    }
                }
            }

            // Next slice idx
            slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
        }

        // Batch-ready barrier: a global all-workers sync so the next batch cannot clobber the reused
        // intermediate scratch or out_ready_sem while this batch is still being consumed. Skipped on the
        // final batch — there is no next batch to protect, and the reader gates receive-side completion.
        // input_tensor_B is a compile-time constant, so this whole block compiles away for B == 1.
        if (b + 1 < input_tensor_B) {
            // Use neighbor unicast instead of multicast to support reshaped 'logical linear' mesh devices
            uint64_t opposite_batch_ready_sem_noc_addr =
                safe_get_noc_addr(opposite_core_x, opposite_core_y, batch_ready_sem, 0);
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_seminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_batch_ready_sem_noc_addr, 0});
            noc_obj.async_writes_flushed();

            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ++batch_ready_target);
        }
    }

    // Reset the out_ready semaphores once, only after all batches
    if constexpr (input_tensor_B > 1) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);
    }

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();
#ifdef USE_WORKER_MUX
    // Close this client's connection. The V2 mux auto-terminates once all of its clients have closed,
    // so no termination-master coordination or explicit terminate signal is needed.
    mux_sender.close();
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
#endif

    noc_obj.async_write_barrier();
}
