// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
constexpr uint32_t output_num_pages = get_named_compile_time_arg_val("output_num_pages");
constexpr uint32_t batch_num_pages = get_named_compile_time_arg_val("batch_num_pages");
constexpr uint32_t slice_B = get_named_compile_time_arg_val("slice_B");

// The V2 fabric mux client (FabricMuxV2Sender) is built entirely from runtime args, so there are no
// worker-side mux compile-time args in either the mux or the direct-fabric path.
constexpr uint32_t num_ct_args = 0;

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
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 1 is forward, 0 is backward
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

#ifdef USE_WORKER_MUX
    // The V2 mux client args are the last runtime args; FabricMuxV2Sender::build_from_args consumes
    // exactly what FabricMuxV2Config::append_client_connection_rt_args serialized on the host.
    size_t mux_arg_idx = arg_idx;
    auto mux_sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(mux_arg_idx);
    arg_idx = mux_arg_idx;
#endif

    const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx>();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_address);

    constexpr auto output_tensor_args = TensorAccessorArgs<intermediate_tensor_args.next_compile_time_args_offset()>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address);

    Noc noc_obj;
    CircularBuffer cb_compute_output(cb_compute_output_id);
    CircularBuffer cb_reader_output(cb_reader_output_id);

#ifndef USE_WORKER_MUX
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
#endif

    // pre-populate packet headers
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();
    auto pkt_hdr_mcastseminc = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);

#ifdef USE_WORKER_MUX
    // Blocking open: waits for the mux to be READY, then requests the connection.
    mux_sender.open();
    auto* fabric_connection_ptr = &mux_sender;
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }
    auto* fabric_connection_ptr =
        direction ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection();
#endif

    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_mcastseminc,
        static_cast<uint8_t>(multicast_route_info.start_distance_in_hops),
        static_cast<uint8_t>(multicast_route_info.range_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1
    if (use_barrier_sem) {
        // multicast to entire ring of workers going in the same direction
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
        ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_mcastseminc, multicast_route_info);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection_ptr,
            pkt_hdr_mcastseminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        pkt_scatter_hdr,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)}),
        page_size * 2);

    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_seminc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        // If not the last slice, write what's on cb_output forward
        CircularBuffer& cb_output = i > 0 ? cb_compute_output : cb_reader_output;

        uint32_t actual_slice_idx;
        if (direction) {
            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
        } else {
            actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
        }

        if (i < (ring_size - 1)) {
            uint32_t intermediate_tile_id_start = actual_slice_idx * output_num_pages;

            uint32_t chunk_count = 0;
            for (uint32_t b = 0; b < slice_B; ++b) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                if (!direction) {
                    uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    tiles_read += backwards_offset;
                }

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    uint32_t tiles_read_in_current_direction = 0;
                    uint32_t tiles_to_read_in_current_direction = 0;
                    if (direction) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }

                    cb_output.wait_front(tile_granularity);
                    size_t l1_read_addr = cb_output.get_read_ptr();
                    while (tiles_read_in_current_direction < tiles_to_read_in_current_direction) {
                        uint32_t tiles_remaining_to_read_in_current_direction =
                            tiles_to_read_in_current_direction - tiles_read_in_current_direction;
                        uint32_t tiles_to_put_in_current_packet =
                            std::min(tiles_remaining_to_read_in_current_direction, num_tiles_to_write_per_packet);

                        uint32_t intermediate_tile_one_id = intermediate_tile_id_start + tiles_read;
                        tiles_read++;
                        tiles_read_in_current_direction++;

                        auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_addrgen, intermediate_tile_one_id, 0);

                        // Will have more cases once scatter-write supports more than 2 distinct addresses
                        switch (tiles_to_put_in_current_packet) {
                            case 2: {
                                uint32_t intermediate_tile_two_id = intermediate_tile_id_start + tiles_read;
                                tiles_read++;
                                tiles_read_in_current_direction++;

                                auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                    intermediate_addrgen, intermediate_tile_two_id, 0);
                                fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                                    fabric_connection_ptr,
                                    pkt_scatter_hdr,
                                    l1_read_addr,
                                    NocUnicastScatterCommandHeader({noc_address0, noc_address1}));
                                l1_read_addr += page_size * 2;
                                break;
                            }
                            case 1:
                            default: {
                                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                                    fabric_connection_ptr,
                                    pkt_unicast_hdr,
                                    l1_read_addr,
                                    NocUnicastCommandHeader{noc_address0});
                                l1_read_addr += page_size;
                                break;
                            }
                        }
                        noc_obj.async_writes_flushed();
                    }
                    cb_output.pop_front(tile_granularity);

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if (!direction) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }
                        tiles_read += tiles_to_read_in_other_direction;
                    }

                    chunk_count++;
                    if (chunk_count % chunks_per_sync == 0) {
                        // 2. unicast output ready semaphore
                        uint64_t out_ready_sem_noc_addr_in_pkt =
                            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                            fabric_connection_ptr,
                            pkt_hdr_seminc,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                    }
                }
                intermediate_tile_id_start += batch_num_pages;
            }

            if (chunk_count % chunks_per_sync != 0) {
                // 2. unicast output ready semaphore
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_connection_ptr,
                    pkt_hdr_seminc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
            }
            noc_obj.async_writes_flushed();
        } else {
            // Otherwise, on the last slice, write it to output buffer
            uint32_t output_tile_id_start = 0;
            for (uint32_t b = 0; b < slice_B; ++b) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                if (!direction) {
                    tiles_read += std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                }
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    uint32_t tiles_to_read_in_current_direction = 0;
                    if (direction) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }

                    cb_output.wait_front(tile_granularity);
                    size_t l1_read_offset = 0;
                    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                        uint32_t output_tile_id = output_tile_id_start + tiles_read;
                        noc_obj.async_write(
                            cb_output,
                            output_addrgen,
                            page_size,
                            {.offset_bytes = l1_read_offset},
                            {.page_id = output_tile_id});
                        l1_read_offset += page_size;
                        tiles_read++;
                    }

                    noc_obj.async_write_barrier();
                    cb_output.pop_front(tile_granularity);

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if (!direction) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }
                        tiles_read += tiles_to_read_in_other_direction;
                    }
                }
                output_tile_id_start += batch_num_pages;
            }
        }

        // Next slice idx
        if (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }

    uint64_t batch_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection_ptr,
        pkt_hdr_mcastseminc,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
    noc_obj.async_writes_flushed();

    // Reset the global semaphore
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ring_size - 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);

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
