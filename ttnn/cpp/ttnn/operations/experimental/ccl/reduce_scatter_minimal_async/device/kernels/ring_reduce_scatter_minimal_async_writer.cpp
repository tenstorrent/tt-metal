// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);  // 4
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(4);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t intermediate_page_size = get_compile_time_arg_val(6);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(7);
constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t ring_size = get_compile_time_arg_val(9);
constexpr uint32_t num_batches = get_compile_time_arg_val(10);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(11);
constexpr bool direction = get_compile_time_arg_val(12);
constexpr uint32_t chunks_per_sync = get_compile_time_arg_val(13);

constexpr bool is_termination_master = get_compile_time_arg_val(14);
constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(15);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(16);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(21);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(22);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(23);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(24);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(25);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(26);
constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<27>();
constexpr ccl_routing_utils::line_multicast_route_info_t multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<27 + ccl_routing_utils::num_line_unicast_args>();

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
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);

    uint32_t slice_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);

    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mux_clients = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t ct_idx =
        27 + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args;

#ifdef INTERMEDIATE_IS_SHARDED
    constexpr uint32_t ct_offset = 7;

    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx),       // Memory layout
        get_compile_time_arg_val(ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(ct_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_addrgen = {
        .bank_base_address = intermediate_address, .shard_array = intermediate_mapping_table};

    arg_idx += intermediate_rt_increment;

#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_address, intermediate_page_size);
#endif

#ifdef OUTPUT_IS_SHARDED
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx + ct_offset),       // Memory layout
        get_compile_time_arg_val(ct_idx + ct_offset + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + ct_offset + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + ct_offset + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(ct_idx + ct_offset + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(ct_idx + ct_offset + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + ct_offset + 6)>;  // pages_per_shard_y

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address, intermediate_page_size);
#endif

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

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, unicast_route_info);

    volatile PACKET_HEADER_TYPE* pkt_hdr_seminc =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);

    tt::tt_fabric::fabric_client_connect(mux_connection_handle);

    if (use_barrier_sem) {
        // multicast to entire ring of workers going in the same direction
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
        pkt_hdr_seminc->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, static_cast<uint16_t>(1), 32});
        ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_seminc, multicast_route_info);
        tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, pkt_hdr_seminc);

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    uint32_t chunk_count = 0;
    for (uint32_t b = 0; b < num_batches; b++) {
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

        uint32_t batch_slice_offset = batch_slice_num_pages * b;
        for (uint32_t i = 0; i < ring_size; ++i) {
            uint32_t actual_slice_idx;
            if constexpr (direction) {
                actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
            } else {
                actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
            }

            uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;
            // If not the last slice, write what's on cb_output_id forward
            if (i < (ring_size - 1)) {
                uint32_t stride_Wt = input_tensor_Wt;
                uint32_t pages_read_in_row = start_pages_read_in_row;
                uint32_t row_offset = start_row_offset;
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;
                uint32_t input_tile_id_start = actual_slice_idx * slice_Wt;
                if constexpr (!direction) {
                    uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    for (uint32_t k = 0; k < backwards_offset; ++k) {
                        pages_read_in_row++;
                        if (pages_read_in_row == slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = pages_read_in_row - slice_Wt;
                        }
                    }
                    tiles_read += backwards_offset;
                }

                chunk_count = 0;
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    uint32_t tiles_read_in_current_direction = 0;
                    uint32_t tiles_to_read_in_current_direction = 0;
                    if constexpr (direction) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }

                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    while (tiles_read_in_current_direction < tiles_to_read_in_current_direction) {
                        uint32_t tiles_remaining_to_read_in_current_direction =
                            tiles_to_read_in_current_direction - tiles_read_in_current_direction;
                        uint32_t tiles_to_put_in_current_packet =
                            std::min(tiles_remaining_to_read_in_current_direction, num_tiles_to_write_per_packet);

                        // Will have more cases once scatter-write supports more than 2 distinct addresses
                        switch (tiles_to_put_in_current_packet) {
                            case 2: {
                                uint32_t tile_one_id = input_tile_id_start + row_offset + pages_read_in_row;
                                pages_read_in_row++;
                                if (pages_read_in_row == slice_Wt) {
                                    row_offset += stride_Wt;
                                    pages_read_in_row = 0;
                                }

                                uint32_t tile_two_id = input_tile_id_start + row_offset + pages_read_in_row;
                                pages_read_in_row++;
                                if (pages_read_in_row == slice_Wt) {
                                    row_offset += stride_Wt;
                                    pages_read_in_row = 0;
                                }

                                scatter_write_for_fabric_write<true, fabric_mux_num_buffers_per_channel>(
                                    intermediate_addrgen,
                                    tile_one_id,
                                    tile_two_id,
                                    pkt_hdr,
                                    mux_connection_handle,
                                    l1_read_addr,
                                    intermediate_page_size);
                                tiles_read += 2;
                                tiles_read_in_current_direction += 2;
                                break;
                            }
                            case 1:
                            default: {
                                uint32_t tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                                pages_read_in_row++;
                                if (pages_read_in_row == slice_Wt) {
                                    row_offset += stride_Wt;
                                    pages_read_in_row = 0;
                                }

                                write_for_fabric_write<true,fabric_mux_num_buffers_per_channel>(
                                    intermediate_addrgen,
                                    tile_id,
                                    pkt_hdr,
                                    mux_connection_handle,
                                    l1_read_addr,
                                    intermediate_page_size);
                                tiles_read++;
                                tiles_read_in_current_direction++;
                                break;
                            }
                        }
                    }
                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if constexpr (!direction) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }

                        for (uint32_t k = 0; k < tiles_to_read_in_other_direction; ++k) {
                            pages_read_in_row++;
                            if (pages_read_in_row == slice_Wt) {
                                row_offset += stride_Wt;
                                pages_read_in_row = pages_read_in_row - slice_Wt;
                            }
                        }
                        tiles_read += tiles_to_read_in_other_direction;
                    }

                    chunk_count++;
                    if (chunk_count % chunks_per_sync == 0) {
                        // 2. unicast output ready semaphore
                        uint64_t out_ready_sem_noc_addr_in_pkt =
                            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                        pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                            out_ready_sem_noc_addr_in_pkt,
                            static_cast<uint16_t>(1),  // increment 1
                            32});
                        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
                        tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, pkt_hdr_seminc);
                    }
                }

                if (chunk_count % chunks_per_sync != 0) {
                    // 2. unicast output ready semaphore
                    uint64_t out_ready_sem_noc_addr_in_pkt =
                        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                    pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                        out_ready_sem_noc_addr_in_pkt,
                        static_cast<uint16_t>(1),  // increment 1
                        32});
                        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
                    tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, pkt_hdr_seminc);
                }
                noc_async_writes_flushed();
            } else {
                // Otherwise, on the last slice, write it to output buffer
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;
                uint32_t tile_id_start = batch_slice_offset;
                if constexpr (!direction) {
                    tiles_read += std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                }
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    uint32_t tiles_to_read_in_current_direction = 0;
                    if constexpr (direction) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }

                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                        uint32_t tile_id = tile_id_start + tiles_read;
                        uint64_t local_noc_addr = get_noc_addr(tile_id, output_addrgen);
                        noc_async_write(l1_read_addr, local_noc_addr, intermediate_page_size);
                        l1_read_addr += intermediate_page_size;
                        tiles_read++;
                    }

                    noc_async_write_barrier();
                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if constexpr (!direction) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }
                        tiles_read += tiles_to_read_in_other_direction;
                    }
                }

                // 2. mcast half batch ready semaphore
                uint64_t batch_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
                pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    batch_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the mcast packet
                ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_seminc, multicast_route_info);
                tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, pkt_hdr_seminc);
                noc_async_writes_flushed();
            }

            // Next slice idx
            if constexpr (direction) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }
        // Reset the global semaphore before the next batch
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);
    if constexpr (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }

    noc_async_write_barrier();
}
