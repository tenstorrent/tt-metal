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
constexpr uint32_t output_batch_num_pages = get_named_compile_time_arg_val("output_batch_num_pages");
constexpr uint32_t output_channel_num_pages = get_named_compile_time_arg_val("output_channel_num_pages");
constexpr uint32_t input_tensor_B = get_named_compile_time_arg_val("input_tensor_B");
constexpr uint32_t slice_C = get_named_compile_time_arg_val("slice_C");
// Contiguous intermediate staging: number of chunks per (slice, channel), and the max number of
// tiles that fit in one fabric packet (used to split a chunk into contiguous fused-unicast packets).
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
    // Consumed for positional-arg alignment with the shared writer RT-arg layout; the contiguous
    // intermediate is addressed from tiles_read alone, so these row/col starts are unused here.
    [[maybe_unused]] const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    // Shortcut staging buffer for the 2nd-last iteration's direct-to-remote contribution (replaces
    // the legacy scatter-write to output_tensor; see rs-contiguous-interm-design).
    address_t shortcut_tensor_address = get_arg_val<address_t>(arg_idx++);
#ifdef USE_WORKER_MUX
    // The V2 mux client args are the last runtime args; FabricMuxV2Sender::build_from_args consumes
    // exactly what FabricMuxV2Config::append_client_connection_rt_args serialized on the host.
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

    constexpr auto shortcut_tensor_args = TensorAccessorArgs<output_tensor_args.next_compile_time_args_offset()>();
    auto shortcut_tensor_accessor = TensorAccessor(shortcut_tensor_args, shortcut_tensor_address);

#ifndef USE_WORKER_MUX
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
#endif
    // pre-populate packet headers
    auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();
    auto pkt_hdr_mcastseminc = PacketHeaderPool::allocate_header();
    // Headers for contiguous writes to the chunk-paged staging buffers (main intermediate and the
    // 2nd-last-iteration shortcut region both use these; see write_contig_chunk below). Their payload
    // size is patched per packet (PayloadSize mask).
    auto pkt_interm_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_interm_fused_hdr = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_interm_unicast_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_interm_fused_hdr, unicast_route_info);

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
    if (use_barrier_sem) {
        // multicast to entire ring of workers for both this dir and opposite dir
        ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_mcastseminc, multicast_route_info);

        uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(this_core_x, this_core_y, barrier_sem, 0);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_mcastseminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

        barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(opposite_core_x, opposite_core_y, barrier_sem, 0);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_mcastseminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

        mf.flush();

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 2 * (ring_size - 1));
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_seminc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    // Contiguous staging writes (main intermediate + shortcut region): the destination address and
    // payload size are patched per packet, so PayloadSize is included in the per-packet update mask
    // below. Only route + (for the fused header) the increment value/flush are fixed here.
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

    // Relevant for 2nd-last iter:
    // In 2nd-last iter we send the full tensor slice. But in preparation for the last iter where each dir
    // processes half tensor slice, in 2nd-last iter we send sem increments to both forward and backward workers.
    // For example, if we send 2 even chunks and 2 odd chunks, we need to send 2 sem incrs to forward worker
    // and 2 sem incrs to backward worker.
    uint64_t this_core_sem_noc_addr = safe_get_noc_addr(this_core_x, this_core_y, out_ready_sem, 0);
    uint64_t opposite_core_sem_noc_addr = safe_get_noc_addr(opposite_core_x, opposite_core_y, out_ready_sem, 0);
    uint64_t even_core_sem_noc_addr = direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;
    uint64_t odd_core_sem_noc_addr = !direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;

    // ---- Fabric send helpers (capture the per-direction connection and pre-configured headers) ----

    // Emit a standalone atomic increment to a remote worker's out_ready_sem.
    auto send_seminc = [&](uint64_t sem_noc_addr) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_seminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
        mf.flush();
        noc_obj.async_writes_flushed();
    };

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

            // The intermediate is a chunk-paged staging buffer: chunk (slice_idx, c, chunk-in-channel)
            // maps to page id `interm_chunk_base + chunk_in_channel`, addressed contiguously below. No
            // per-tile row/col tracking is needed (unlike the tiled output tensor).
            uint32_t interm_chunk_base = (slice_idx * slice_C) * chunks_per_channel;

            // address incrementer for output_tensor
            uint32_t output_tile_id_start = b * output_batch_num_pages;
            uint32_t output_tiles_read = start_tiles_read;
            auto get_next_output_tile_id = [&]() -> uint32_t { return output_tile_id_start + (output_tiles_read++); };

            uint32_t chunk_count = 0;
            uint32_t even_chunk_count = 0;
            uint32_t odd_chunk_count = 0;
            for (uint32_t c = 0; c < slice_C; ++c) {
                // reset addr counters
                output_tiles_read = start_tiles_read;
                uint32_t tiles_read = start_tiles_read;
                uint32_t total_tiles_to_read = start_tiles_to_read;
                // Base chunk page id for this (slice, channel).
                const uint32_t interm_channel_chunk_base = interm_chunk_base + c * chunks_per_channel;
                // Shortcut staging buffer has no slice_idx axis (each device receives exactly one
                // such contribution, from exactly one neighbor, at exactly one iteration), so its
                // base is the channel offset alone.
                const uint32_t shortcut_channel_chunk_base = c * chunks_per_channel;

                while (tiles_read < total_tiles_to_read) {
                    const auto [is_even_chunk, tiles_to_read] =
                        reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);

                    if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                        // Skip this chunk. Advance the output tile counter so tiled-output addressing stays
                        // aligned; the intermediate is addressed statelessly from tiles_read.
                        tiles_read += tiles_to_read;
                        for (uint32_t k = 0; k < tiles_to_read; ++k) {
                            get_next_output_tile_id();
                        }
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
                            bool seminc_fused = false;

                            // Write tiles to remote tensor over Fabric. Both write_to_interm (mid-ring
                            // hops) and the 2nd-last-iteration shortcut now target a chunk-paged
                            // staging buffer (main intermediate or the dedicated shortcut region) with
                            // the same contiguous fused-unicast packetization: one page = one chunk, so
                            // the whole chunk is contiguous at the destination and each fabric packet
                            // is a single unicast write (no scatter). The packet that reaches
                            // chunks_per_sync fuses its semaphore increment; no standalone seminc.
                            cb_out.wait_front(tile_granularity);
                            size_t l1_read_addr = cb_out.get_read_ptr();
                            auto write_contig_chunk = [&](auto& dst_accessor, uint32_t channel_chunk_base) {
                                const uint32_t chunk_page_id = channel_chunk_base + tiles_read / tile_granularity;
                                const uint32_t in_chunk_offset = (tiles_read % tile_granularity) * page_size;
                                for (uint32_t j = 0; j < tiles_to_read; j += interm_tiles_per_packet) {
                                    const uint32_t tiles_in_packet =
                                        std::min(tiles_to_read - j, interm_tiles_per_packet);
                                    const uint16_t payload_bytes = static_cast<uint16_t>(tiles_in_packet * page_size);
                                    const uint64_t dst_noc_addr =
                                        tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                            dst_accessor, chunk_page_id, in_chunk_offset + j * page_size);
                                    const bool last_packet = (j + interm_tiles_per_packet >= tiles_to_read);
                                    if (fuse_seminc && last_packet) {
                                        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                                            UnicastFusedAtomicIncUpdateMask::PayloadSize |
                                            UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                                            UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                                            fabric_direction_connection,
                                            pkt_interm_fused_hdr,
                                            l1_read_addr,
                                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                                dst_noc_addr, sem_noc_addr, static_cast<uint32_t>(1)},
                                            payload_bytes);
                                        seminc_fused = true;
                                    } else {
                                        fabric_unicast_noc_unicast_write_with_state<
                                            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                                            fabric_direction_connection,
                                            pkt_interm_unicast_hdr,
                                            l1_read_addr,
                                            tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr},
                                            payload_bytes);
                                    }

                                    mf.flush();

                                    noc_obj.async_writes_flushed();
                                    l1_read_addr += payload_bytes;
                                    tiles_read += tiles_in_packet;
                                }
                            };
                            if (write_to_interm) {
                                write_contig_chunk(interm_tensor_accessor, interm_channel_chunk_base);
                            } else {
                                // 2nd-last iteration shortcut: stages this direction's contribution into
                                // the dedicated shortcut buffer instead of scatter-writing into the tiled
                                // output tensor. The receiver's final iteration reads it back as the 3rd
                                // term of its local 3-way reduce. See rs-contiguous-interm-design.
                                write_contig_chunk(shortcut_tensor_accessor, shortcut_channel_chunk_base);
                            }
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
                            // Write tiles to the local output tensor (last iter; never targets the
                            // intermediate, so write_to_interm is always false here).
                            cb_out.wait_front(tile_granularity);
                            size_t l1_read_offset = 0;
                            for (uint32_t j = 0; j < tiles_to_read; ++j) {
                                auto output_tile_id = get_next_output_tile_id();
                                noc_obj.async_write(
                                    cb_out,
                                    output_tensor_accessor,
                                    page_size,
                                    {.offset_bytes = l1_read_offset},
                                    {.page_id = output_tile_id});
                                l1_read_offset += page_size;
                                tiles_read++;
                            }
                            noc_obj.async_write_barrier();
                            cb_out.pop_front(tile_granularity);
                        }  // if remote or local
                    }  // if skip or process
                }  // while total_tiles_to_read

                output_tile_id_start += output_channel_num_pages;
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
            // multicast to entire ring of workers for both this dir and opposite dir
            uint64_t batch_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(this_core_x, this_core_y, batch_ready_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_mcastseminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
            noc_obj.async_writes_flushed();

            batch_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(opposite_core_x, opposite_core_y, batch_ready_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_mcastseminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
            noc_obj.async_writes_flushed();

            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 2 * (ring_size - 1));
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);  // reset before next batch
        }
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
