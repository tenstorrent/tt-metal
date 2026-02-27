// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include "strided_ring_reduce_scatter_common.hpp"

using address_t = uint32_t;
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
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(11);
constexpr uint32_t slice_C = get_compile_time_arg_val(12);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(13);
constexpr uint32_t dim = get_compile_time_arg_val(14);
constexpr uint32_t mm_M_unit_blocks_per_core = get_compile_time_arg_val(15);
constexpr uint32_t mm_N_full_blocks_per_slice = get_compile_time_arg_val(16);
constexpr uint32_t mm_block_ht = get_compile_time_arg_val(17);
constexpr uint32_t mm_cores_y = get_compile_time_arg_val(18);
constexpr uint32_t N_full_block_wt = get_compile_time_arg_val(19);
constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(20);
constexpr uint32_t chunks_per_mm_N_full_block = get_compile_time_arg_val(21);
constexpr uint32_t slice_Ht_per_core = get_compile_time_arg_val(22);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(23);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(24);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(25);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(26);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(27);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(28);

constexpr uint32_t num_ct_args = 29;

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
    const address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 1 is forward, 0 is backward
    const uint32_t worker_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(arg_idx++);
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
    const uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    const auto& unicast_route_info = direction ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = direction ? forward_multicast_route_info : backward_multicast_route_info;
    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

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
    arg_idx += intermediate_rt_increment;

#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_address, page_size);
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
    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address, page_size);
#endif

    if (mux_connection_valid) {
        auto mux_connection_handle =
            tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
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

        auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
        auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
        auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();
        auto pkt_hdr_mcastseminc = PacketHeaderPool::allocate_header();
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);

        tt::tt_fabric::fabric_client_connect(mux_connection_handle);

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
                &mux_connection_handle,
                pkt_hdr_mcastseminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        }

        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_seminc,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0,                           // ignore
                static_cast<uint32_t>(1)});  // increment 1

        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)}),
            page_size * 2);

        ASSERT(dim == 3);
        ASSERT(slice_C == 1);

        const uint32_t batch_size = input_tensor_B;
        const uint32_t last_mm_core_idx = mm_cores_y - 1;
        // Use actual row count per core (not padded), so coordinates_to_slice_coordinates
        // produces correct absolute row offsets across MM cores.
        const uint32_t tiles_ht_per_core = slice_Ht_per_core;
        const uint32_t effective_worker_id = worker_id + (direction ? num_workers : 0);
        const uint32_t effective_advance_by_tiles = 2 * num_workers;

        for (uint32_t b = 0; b < batch_size; b++) {
            const uint32_t output_tile_id_start = b * output_batch_num_pages;

            for (uint32_t m_block_iter = 0; m_block_iter < mm_M_unit_blocks_per_core; m_block_iter++) {
                const uint32_t current_mm_block_ht =
                    get_current_mm_block_ht(m_block_iter, mm_M_unit_blocks_per_core, mm_block_ht, slice_Ht_per_core);
                for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_full_block; chunk_idx++) {
                    const uint32_t effective_chunk_width_in_tiles =
                        get_effective_chunk_width_in_tiles(chunk_idx, chunk_width_in_tiles, N_full_block_wt);
                    const uint32_t effective_subchunk_size = current_mm_block_ht * effective_chunk_width_in_tiles;
                    int32_t slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

                    for (uint32_t i = 0; i < ring_size; i++) {
                        const uint32_t actual_slice_idx = wrap_slice_idx(slice_idx, direction, ring_size);
                        const uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;

                        for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_full_blocks_per_slice;
                             chunk_piece_idx++) {
                            uint32_t tile_row_in_mm_M_unit_block = 0;
                            uint32_t chunk_col_in_tiles = 0;
                            uint32_t mm_core_idx = 0;
                            get_next_tile_coordinates(
                                tile_row_in_mm_M_unit_block,
                                chunk_col_in_tiles,
                                mm_core_idx,
                                effective_worker_id,
                                effective_subchunk_size,
                                effective_chunk_width_in_tiles,
                                current_mm_block_ht);
                            uint32_t tiles_to_read = how_many_tiles_to_read_formula(
                                tile_row_in_mm_M_unit_block,
                                chunk_col_in_tiles,
                                mm_core_idx,
                                effective_advance_by_tiles,
                                last_mm_core_idx,
                                effective_subchunk_size,
                                effective_chunk_width_in_tiles);

                            while (tiles_to_read > 0) {
                                const uint32_t tiles_to_read_in_this_step = std::min(tiles_to_read, tile_granularity);
                                tiles_to_read -= tiles_to_read_in_this_step;

                                cb_wait_front(cb_output_id, tile_granularity);
                                size_t l1_read_addr = get_read_ptr(cb_output_id);

                                uint32_t tiles_remaining_in_step = tiles_to_read_in_this_step;
                                while (tiles_remaining_in_step > 0) {
                                    const uint32_t tiles_to_put_in_current_packet =
                                        (i < (ring_size - 1))
                                            ? std::min(tiles_remaining_in_step, num_tiles_to_write_per_packet)
                                            : 1;
                                    tiles_remaining_in_step -= tiles_to_put_in_current_packet;
                                    const bool can_use_two_tiles_in_packet = tiles_to_put_in_current_packet == 2;

                                    // Tile 1
                                    const auto [slice_row, slice_col] = coordinates_to_slice_coordinates(
                                        tile_row_in_mm_M_unit_block,
                                        chunk_col_in_tiles,
                                        mm_core_idx,
                                        chunk_piece_idx,
                                        m_block_iter,
                                        chunk_idx,
                                        N_full_block_wt,
                                        tiles_ht_per_core,
                                        mm_block_ht,  // full stride between blocks for absolute row calculation
                                        chunk_width_in_tiles);
                                    const bool tile1_in_bounds = slice_row < slice_Ht;
                                    const uint32_t slice_tile_idx =
                                        slice_coordinates_to_slice_tile_index(slice_row, slice_col, slice_Wt);
                                    const uint32_t global_tile_idx = slice_coordinates_to_global_tile_index(
                                        slice_row, slice_col, actual_slice_idx, slice_Wt, input_tensor_Wt);
                                    get_next_tile_coordinates(
                                        tile_row_in_mm_M_unit_block,
                                        chunk_col_in_tiles,
                                        mm_core_idx,
                                        effective_advance_by_tiles,
                                        effective_subchunk_size,
                                        effective_chunk_width_in_tiles,
                                        current_mm_block_ht);

                                    // Tile 2 (only when the packet can hold 2 tiles)
                                    bool tile2_in_bounds = false;
                                    uint32_t global_tile_idx_two = 0;
                                    if (can_use_two_tiles_in_packet) {
                                        const auto [slice_row_two, slice_col_two] = coordinates_to_slice_coordinates(
                                            tile_row_in_mm_M_unit_block,
                                            chunk_col_in_tiles,
                                            mm_core_idx,
                                            chunk_piece_idx,
                                            m_block_iter,
                                            chunk_idx,
                                            N_full_block_wt,
                                            tiles_ht_per_core,
                                            mm_block_ht,  // full stride between blocks for absolute row calculation
                                            chunk_width_in_tiles);
                                        tile2_in_bounds = slice_row_two < slice_Ht;
                                        global_tile_idx_two = slice_coordinates_to_global_tile_index(
                                            slice_row_two, slice_col_two, actual_slice_idx, slice_Wt, input_tensor_Wt);
                                        get_next_tile_coordinates(
                                            tile_row_in_mm_M_unit_block,
                                            chunk_col_in_tiles,
                                            mm_core_idx,
                                            effective_advance_by_tiles,
                                            effective_subchunk_size,
                                            effective_chunk_width_in_tiles,
                                            current_mm_block_ht);
                                    }

                                    if (i < (ring_size - 1)) {
                                        // Write tile(s) to the intermediate buffer on the neighboring device.
                                        if (tile1_in_bounds && can_use_two_tiles_in_packet && tile2_in_bounds) {
                                            const auto noc_address0 =
                                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                    intermediate_addrgen, global_tile_idx, 0);
                                            const auto noc_address1 =
                                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                    intermediate_addrgen, global_tile_idx_two, 0);
                                            fabric_unicast_noc_scatter_write_with_state<
                                                UnicastScatterWriteUpdateMask::DstAddrs>(
                                                &mux_connection_handle,
                                                pkt_scatter_hdr,
                                                l1_read_addr,
                                                NocUnicastScatterCommandHeader({noc_address0, noc_address1}));
                                            l1_read_addr += page_size * 2;
                                        } else if (tile1_in_bounds) {
                                            const auto noc_address0 =
                                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                    intermediate_addrgen, global_tile_idx, 0);
                                            fabric_unicast_noc_unicast_write_with_state<
                                                UnicastWriteUpdateMask::DstAddr>(
                                                &mux_connection_handle,
                                                pkt_unicast_hdr,
                                                l1_read_addr,
                                                NocUnicastCommandHeader{noc_address0});
                                            l1_read_addr += page_size * tiles_to_put_in_current_packet;
                                        } else {
                                            l1_read_addr += page_size * tiles_to_put_in_current_packet;
                                        }
                                        noc_async_writes_flushed();
                                    } else {
                                        // Write the tile to the output buffer on this device.
                                        if (tile1_in_bounds) {
                                            const uint32_t output_tile_id = output_tile_id_start + slice_tile_idx;
                                            const uint64_t local_noc_addr =
                                                get_noc_addr(output_tile_id, output_addrgen);
                                            noc_async_write(l1_read_addr, local_noc_addr, page_size);
                                        }
                                        l1_read_addr += page_size;
                                    }
                                }
                                noc_async_write_barrier();
                                cb_pop_front(cb_output_id, tile_granularity);
                            }
                        }

                        if (i < (ring_size - 1)) {
                            // Signal reader on the neighboring device after all chunk_piece_idx tiles for this ring
                            // iteration are written.
                            const uint64_t out_ready_sem_noc_addr_in_pkt =
                                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                                &mux_connection_handle,
                                pkt_hdr_seminc,
                                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                            noc_async_writes_flushed();
                        } else {
                            noc_async_write_barrier();
                        }

                        // Move to the next slice
                        slice_idx += direction ? -1 : 1;
                    }
                }
            }

            // Signal batch done and wait for all other chips to finish before next batch.
            // This is needed to avoid race conditions in the intermediate buffer.
            const uint64_t batch_ready_sem_noc_addr_in_pkt =
                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                &mux_connection_handle,
                pkt_hdr_mcastseminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
            noc_async_writes_flushed();

            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ring_size - 1);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);
        }

        noc_async_write_barrier();
        noc_async_atomic_barrier();

        tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);

        if (is_termination_master) {
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
            noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            const uint64_t dest_addr =
                safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
            noc_semaphore_inc(dest_addr, 1);
            noc_async_atomic_barrier();
        }
    }

    noc_async_write_barrier();
}
