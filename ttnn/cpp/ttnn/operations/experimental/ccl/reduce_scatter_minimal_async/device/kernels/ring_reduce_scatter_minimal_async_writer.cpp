// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
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

#include "api/debug/dprint.h"

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(3);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(4);
constexpr uint32_t page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(6);
constexpr uint32_t output_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t output_channel_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(11);
constexpr uint32_t slice_C = get_compile_time_arg_val(12);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(13);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(14);
constexpr uint32_t dim = get_compile_time_arg_val(15);
#ifdef USE_WORKER_MUX
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(16);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(19);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(20);

constexpr uint32_t num_ct_args = 21;
#else
constexpr uint32_t num_ct_args = 16;
#endif

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
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
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
#ifdef USE_WORKER_MUX
    const bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
#endif

    const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

    constexpr auto interm_tensor_args = TensorAccessorArgs<ct_idx>();
    auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);
    constexpr uint32_t ct_idx2 = ct_idx + interm_tensor_args.num_compile_time_args();

    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx2>();
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);
    constexpr uint32_t ct_idx3 = ct_idx2 + output_tensor_args.num_compile_time_args();

#ifdef USE_WORKER_MUX
    auto mux_connection_handle = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address);

    // need to wait for fabric mux to be ready to accept connections
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
#else
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
    tt::tt_fabric::fabric_client_connect(mux_connection_handle);
    auto* fabric_direction_connection = &mux_connection_handle;
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    auto* fabric_direction_connection =
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
            fabric_direction_connection,
            pkt_hdr_mcastseminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported");
    uint64_t remote_noc_addrs[4] = {0, 0, 0, 0};
    uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        pkt_scatter_hdr,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        NocUnicastScatterCommandHeader(remote_noc_addrs, chunk_sizes, num_tiles_to_write_per_packet),
        page_size * num_tiles_to_write_per_packet);

    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_seminc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

        for (uint32_t i = 0; i < ring_size; ++i) {
            const bool full_slice = false;                                               // TODO ...
            const bool even_chunks = direction;  // TODO ... when sending half the slice, even or odd chunks
            const bool write_to_remote = (i < (ring_size - 1));                          // TODO ... which device
            const bool write_to_interm = (i < (ring_size - 1));                          // TODO ... which tensor
            uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;  // TODO ...

            // slice_idx = slice_idx % ring_size
            if (direction) {
                slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
            } else {
                slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
            }

            // address incrementer for interm_tensor
            uint32_t interm_tile_id_start;
            if constexpr (dim == 3) {
                interm_tile_id_start = slice_idx * slice_Wt;
            } else if constexpr (dim == 2) {
                interm_tile_id_start = slice_idx * slice_Ht * slice_Wt;
            } else if constexpr (dim == 1) {
                interm_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt;
            } else {
                ASSERT(false);
            }
            uint32_t interm_pages_read_in_row = start_pages_read_in_row;
            uint32_t interm_row_offset = start_row_offset;
            auto get_next_interm_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = interm_tile_id_start + interm_row_offset + interm_pages_read_in_row;
                interm_pages_read_in_row++;
                if (interm_pages_read_in_row == slice_Wt) {
                    interm_row_offset += input_tensor_Wt;
                    interm_pages_read_in_row -= slice_Wt;
                }
                return tile_id;
            };

            // address incrementer for output_tensor
            uint32_t output_tile_id_start = b * output_batch_num_pages;
            uint32_t output_tiles_read = start_tiles_read;
            auto get_next_output_tile_id = [&]() -> uint32_t { return output_tile_id_start + (output_tiles_read++); };

            // pick the correct address generator and incrementer based on write_to_interm
            auto get_next_tile_id = [&]() -> uint32_t {
                if (write_to_interm) {
                    return get_next_interm_tile_id();
                } else {
                    return get_next_output_tile_id();
                }
            };
            auto get_remote_tile_addr = [&](uint32_t tile_id) -> uint64_t {
                if (write_to_interm) {
                    return tt::tt_fabric::linear::addrgen_detail::get_noc_address(interm_tensor_accessor, tile_id, 0);
                } else {
                    return tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_tensor_accessor, tile_id, 0);
                }
            };
            auto get_local_tile_addr = [&](uint32_t tile_id) -> uint64_t {
                if (write_to_interm) {
                    return interm_tensor_accessor.get_noc_addr(tile_id);
                } else {
                    return output_tensor_accessor.get_noc_addr(tile_id);
                }
            };

            uint32_t chunk_count = 0;
            for (uint32_t c = 0; c < slice_C; ++c) {
                // reset addr counters
                output_tiles_read = start_tiles_read;
                interm_pages_read_in_row = start_pages_read_in_row;
                interm_row_offset = start_row_offset;
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                if (!full_slice && !even_chunks) {
                    uint32_t first_even_chunk = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    tiles_read += first_even_chunk;
                    for (uint32_t k = 0; k < first_even_chunk; ++k) {
                        get_next_tile_id();
                    }
                }

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    uint32_t tiles_to_read_in_current_direction = 0;
                    if (full_slice || !even_chunks) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    }

                    if (write_to_remote) {
                        // Write tiles to remote tensor over Fabric
                        cb_wait_front(cb_output_id, tile_granularity);
                        size_t l1_read_addr = get_read_ptr(cb_output_id);
                        for (uint32_t j = 0; j < tiles_to_read_in_current_direction;
                             j += num_tiles_to_write_per_packet) {
                            uint32_t tiles_to_put_in_current_packet =
                                std::min(tiles_to_read_in_current_direction - j, num_tiles_to_write_per_packet);

                            for (uint32_t k = 0; k < tiles_to_put_in_current_packet; ++k) {
                                remote_noc_addrs[k] = get_remote_tile_addr(get_next_tile_id());
                            }

                            if (tiles_to_put_in_current_packet > 1) {
                                fabric_unicast_noc_scatter_write_with_state<
                                    UnicastScatterWriteUpdateMask::DstAddrs |
                                    UnicastScatterWriteUpdateMask::ChunkSizes |
                                    UnicastScatterWriteUpdateMask::PayloadSize>(
                                    fabric_direction_connection,
                                    pkt_scatter_hdr,
                                    l1_read_addr,
                                    NocUnicastScatterCommandHeader(
                                        remote_noc_addrs, chunk_sizes, tiles_to_put_in_current_packet),
                                    page_size * tiles_to_put_in_current_packet);
                            } else {
                                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                                    fabric_direction_connection,
                                    pkt_unicast_hdr,
                                    l1_read_addr,
                                    NocUnicastCommandHeader{remote_noc_addrs[0]});
                            }
                            l1_read_addr += page_size * tiles_to_put_in_current_packet;
                            tiles_read += tiles_to_put_in_current_packet;
                            noc_async_writes_flushed();
                        }
                        cb_pop_front(cb_output_id, tile_granularity);

                        // ++chunk_count % chunks_per_sync
                        chunk_count = (chunk_count == chunks_per_sync) ? 0 : (chunk_count + 1);
                        if (chunk_count == chunks_per_sync) {
                            // 2. unicast output ready semaphore
                            uint64_t out_ready_sem_noc_addr_in_pkt =
                                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                                fabric_direction_connection,
                                pkt_hdr_seminc,
                                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                        }
                    } else {
                        // Write tiles to local tensor
                        cb_wait_front(cb_output_id, tile_granularity);
                        size_t l1_read_addr = get_read_ptr(cb_output_id);
                        for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                            // DPRINT << "W: output_tile_id=" << output_tile_id << ENDL();
                            uint64_t local_noc_addr = get_local_tile_addr(get_next_tile_id());
                            noc_async_write(l1_read_addr, local_noc_addr, page_size);
                            l1_read_addr += page_size;
                            tiles_read++;
                        }
                        noc_async_write_barrier();
                        cb_pop_front(cb_output_id, tile_granularity);
                    }

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (!full_slice && tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if (!even_chunks) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }

                        tiles_read += tiles_to_read_in_other_direction;
                        for (uint32_t k = 0; k < tiles_to_read_in_other_direction; ++k) {
                            get_next_tile_id();
                        }
                    }
                }

                interm_tile_id_start += input_channel_num_pages;
                output_tile_id_start += output_channel_num_pages;
            }

            if (write_to_remote && chunk_count != chunks_per_sync) {
                // 2. unicast output ready semaphore
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_seminc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                noc_async_writes_flushed();
            }

            // Next slice idx
            slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
        }

        // 2. mcast half batch ready semaphore
        uint64_t batch_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_mcastseminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
        noc_async_writes_flushed();

        // Reset the global semaphore before the next batch
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
#ifdef USE_WORKER_MUX
    tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);
    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
#endif

    noc_async_write_barrier();
}
