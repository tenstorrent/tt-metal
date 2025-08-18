// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);  // 4
constexpr BufferType intermediate_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr BufferType output_type = static_cast<BufferType>(get_compile_time_arg_val(4));
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(5);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(6);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(7);
constexpr uint32_t intermediate_page_size = get_compile_time_arg_val(8);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(9);
constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t ring_size = get_compile_time_arg_val(11);
constexpr uint32_t num_batches = get_compile_time_arg_val(12);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(13);
constexpr bool is_forward = get_compile_time_arg_val(14);
constexpr bool is_first_device_in_direction = get_compile_time_arg_val(15);
constexpr uint32_t num_targets_in_direction = get_compile_time_arg_val(16);
constexpr uint32_t num_intermediate_reduction_steps = get_compile_time_arg_val(17);
constexpr bool do_final_reduction = get_compile_time_arg_val(18);
constexpr uint32_t num_total_reduction_steps = get_compile_time_arg_val(19);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(20);
constexpr uint32_t chunks_per_sync = get_compile_time_arg_val(21);

constexpr bool is_termination_master = get_compile_time_arg_val(22);
constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(23);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(24);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(25);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(26);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(27);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(28);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(29);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(30);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(31);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(32);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(33);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(34);
constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<35>();
constexpr ccl_routing_utils::line_multicast_route_info_t multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<35 + ccl_routing_utils::num_line_unicast_args>();

constexpr uint32_t batch_num_pages = batch_slice_num_pages * ring_size;
constexpr uint32_t intermediate_num_pages = batch_num_pages * num_batches;
constexpr uint32_t intermediate_full_offset = is_forward ? 0 : intermediate_num_pages;

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
    size_t final_reduction_slot_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fwd_bwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    bool signal_on_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    bool wait_on_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
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

    constexpr uint32_t ct_idx = 35 + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args;

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
    constexpr uint32_t ct_offset = 0;

    constexpr bool intermediate_is_dram = intermediate_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_addrgen = InterleavedAddrGenFast<intermediate_is_dram>{
        .bank_base_address = intermediate_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};
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
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    auto output_addrgen = InterleavedAddrGenFast<output_is_dram>{
        .bank_base_address = output_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};
#endif

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection;
    if (mux_connection_valid) {
        mux_connection = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
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
        mux_connection_handle = &mux_connection;
    } else {
        mux_connection_handle = nullptr;
    }

    if (mux_connection_valid) {
        // need to wait for fabric mux to be ready to accept connections
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
        tt::tt_fabric::fabric_client_connect_start(*mux_connection_handle);
    }

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

    uint32_t slice_Wt = input_tensor_Wt / ring_size;

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_connect_finish(*mux_connection_handle);
    }

    // Due to the existing direction of fabric connections, forward writers will signal to backward writers
    // and backward writers will signal to forward writers
    if (signal_on_barrier_sem) {
        uint64_t barrier_sem_noc_addr_in_pkt =
            safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0);
        pkt_hdr_seminc->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, static_cast<uint16_t>(1), 32});
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
        tt::tt_fabric::fabric_atomic_inc(*mux_connection_handle, pkt_hdr_seminc);
        noc_async_writes_flushed();
    }
    if (wait_on_barrier_sem) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        static_cast<uint16_t>(0xFFFF)});
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);

    uint32_t chunk_count = 0;

    for (uint32_t b = 0; b < num_batches; b++) {
        int slice_idx = is_forward ? ring_size - 1 : 0;

        uint32_t batch_slice_offset = batch_slice_num_pages * b;
        uint32_t batch_offset = batch_num_pages * b;
        for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
            chunk_count = 0;

            constexpr uint32_t cb_output_id = is_first_device_in_direction ? cb_reader_output_id : cb_compute_output_id;

            uint32_t stride_Wt = input_tensor_Wt;
            uint32_t pages_read_in_row = (link * batch_slice_num_pages / num_links) % slice_Wt;
            uint32_t row_offset = (link * batch_slice_num_pages / num_links) / slice_Wt * stride_Wt;
            uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
            uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;

            uint32_t input_tile_id_start = intermediate_full_offset + batch_offset + slice_idx * slice_Wt;

            // Write to remote intermediate buffer
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                cb_wait_front(cb_output_id, tile_granularity);
                size_t l1_read_addr = get_read_ptr(cb_output_id);

                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t num_pages_to_write = std::min(contig_pages_advanced, num_pages_to_read - j);

                    uint32_t first_tile_id = input_tile_id_start + row_offset + pages_read_in_row;

                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }

                    if (num_pages_to_write == 1) {
                        write_for_fabric_write<true, fabric_mux_num_buffers_per_channel>(
                            intermediate_addrgen,
                            first_tile_id,
                            pkt_hdr,
                            *mux_connection_handle,
                            l1_read_addr,
                            intermediate_page_size);
                    } else if (num_pages_to_write == 2) {
                        uint32_t second_tile_id = input_tile_id_start + row_offset + pages_read_in_row;

                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }

                        scatter_write_for_fabric_write<true, fabric_mux_num_buffers_per_channel>(
                            intermediate_addrgen,
                            first_tile_id,
                            second_tile_id,
                            pkt_hdr,
                            *mux_connection_handle,
                            l1_read_addr);
                    } else {
                        ASSERT(false);
                    }

                    tiles_read += num_pages_to_write;
                }
                cb_pop_front(cb_output_id, tile_granularity);

                chunk_count++;
                if (chunk_count % chunks_per_sync == 0) {
                    // 2. unicast output ready semaphore
                    tt::tt_fabric::fabric_atomic_inc(*mux_connection_handle, pkt_hdr_seminc);
                }
            }
            if (chunk_count % chunks_per_sync != 0) {
                // 2. unicast output ready semaphore
                tt::tt_fabric::fabric_atomic_inc(*mux_connection_handle, pkt_hdr_seminc);
            }

            // Next slice idx
            if constexpr (is_forward) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }

        // Do write of final reduction and sync local FWD/BWD cores
        if constexpr (do_final_reduction) {
            // Write output
            uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
            uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;
            uint32_t tile_id_start = batch_slice_offset;

            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                cb_wait_front(cb_compute_output_id, tile_granularity);
                size_t l1_read_addr = get_read_ptr(cb_compute_output_id);

                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    uint32_t tile_id = tile_id_start + tiles_read;
                    uint64_t local_noc_addr = get_noc_addr(tile_id, output_addrgen);
                    noc_async_write(l1_read_addr, local_noc_addr, intermediate_page_size);
                    l1_read_addr += intermediate_page_size;
                    tiles_read++;
                }

                if constexpr (sync_with_other_direction && is_forward) {
                    noc_async_write_barrier();
                } else {
                    noc_async_writes_flushed();
                }
                cb_pop_front(cb_compute_output_id, tile_granularity);
                if constexpr (sync_with_other_direction && is_forward) {
                    // Tell local backwards reader that it can proceed
                    uint64_t fwd_bwd_sem_noc_addr =
                        safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, fwd_bwd_sem_addr, 0);
                    noc_semaphore_inc(fwd_bwd_sem_noc_addr, 1);
                }
            }

            noc_async_write_barrier();
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

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
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
