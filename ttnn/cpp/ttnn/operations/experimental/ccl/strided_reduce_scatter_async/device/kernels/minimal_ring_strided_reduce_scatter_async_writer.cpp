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
#include <utility>
#include "strided_ring_reduce_scatter_common.hpp"

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
constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(16);
constexpr uint32_t mm_N_blocks_per_slice = get_compile_time_arg_val(17);
constexpr uint32_t mm_block_ht = get_compile_time_arg_val(18);
constexpr uint32_t mm_cores_y = get_compile_time_arg_val(19);
constexpr uint32_t N_block_wt = get_compile_time_arg_val(20);
constexpr uint32_t chunk_width_in_tiles = get_compile_time_arg_val(21);
constexpr uint32_t chunks_per_mm_N_block = get_compile_time_arg_val(22);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(23);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(24);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(25);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(26);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(27);

constexpr uint32_t num_ct_args = 28;

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
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
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
    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;
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

        // Skip barrier semaphore in minimal version (no fabric multicast to increment it)

        // Let's set some particular values for the params used
        const uint32_t batch_size = input_tensor_B;
        const uint32_t last_mm_core_idx = mm_cores_y - 1;
        const uint32_t tiles_ht_per_core = mm_block_ht * M_blocks_per_core;

        uint32_t effective_worker_id = worker_id + (direction ? num_workers : 0);
        const uint32_t effective_advance_by_tiles = 2 * num_workers;

        ASSERT(dim == 3);
        ASSERT(slice_C == 1);

        DPRINT << "The writer kernel running its loop." << ENDL();
        DPRINT << "my_chip_id: " << my_chip_id << ENDL();
        DPRINT << "ring_size: " << ring_size << ENDL();
        DPRINT << "tile_granularity: " << tile_granularity << ENDL();
        DPRINT << "page_size: " << page_size << ENDL();
        DPRINT << "output_batch_num_pages: " << output_batch_num_pages << ENDL();
        DPRINT << "input_tensor_B: " << input_tensor_B << ENDL();
        DPRINT << "input_tensor_Wt: " << input_tensor_Wt << ENDL();
        DPRINT << "slice_C: " << slice_C << ENDL();
        DPRINT << "slice_Ht: " << slice_Ht << ENDL();
        DPRINT << "slice_Wt: " << slice_Wt << ENDL();
        DPRINT << "dim: " << dim << ENDL();
        DPRINT << "start_pages_read_in_row: " << start_pages_read_in_row << ENDL();
        DPRINT << "start_row_offset: " << start_row_offset << ENDL();
        DPRINT << "start_tiles_read: " << start_tiles_read << ENDL();
        DPRINT << "start_tiles_to_read: " << start_tiles_to_read << ENDL();
        DPRINT << "chunk_width_in_tiles: " << chunk_width_in_tiles << ENDL();
        DPRINT << "direction: " << (uint32_t)direction << ENDL();
        DPRINT << "chunks_per_sync: " << chunks_per_sync << ENDL();
        DPRINT << "worker_id: " << worker_id << ENDL();
        DPRINT << "num_workers: " << num_workers << ENDL();
        DPRINT << "batch_size: " << input_tensor_B << ENDL();

        DPRINT << " start_row_offset: " << start_row_offset << ENDL();

        for (uint32_t b = 0; b < batch_size; b++) {
            DPRINT << "================================================" << ENDL();
            DPRINT << "batch: " << b << " started" << ENDL();

            for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
                DPRINT << "--------------------------------" << ENDL();
                DPRINT << "m_block_iter: " << m_block_iter << " started" << ENDL();
                uint32_t output_tile_id_start = b * output_batch_num_pages;

                for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_block; chunk_idx++) {
                    DPRINT << "chunk_idx: " << chunk_idx << " started" << ENDL();
                    uint32_t effective_chunk_width_in_tiles =
                        get_effective_chunk_width_in_tiles(chunk_idx, chunk_width_in_tiles, slice_Wt);
                    int32_t slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

                    for (uint32_t i = 0; i < ring_size; i++) {
                        DPRINT << "************************************************" << ENDL();
                        DPRINT << "ring iteration: " << i << " started" << ENDL();
                        DPRINT << "slice_idx: " << slice_idx << ENDL();
                        DPRINT << "direction: " << (uint32_t)direction << ENDL();

                        uint32_t actual_slice_idx;
                        if (direction) {
                            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
                        } else {
                            actual_slice_idx =
                                slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
                        }
                        DPRINT << "actual_slice_idx: " << actual_slice_idx << ", m_block_iter: " << m_block_iter
                               << ", chunk_idx: " << chunk_idx << ENDL();
                        uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;

                        for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_blocks_per_slice; chunk_piece_idx++) {
                            DPRINT << "chunk_piece_idx: " << chunk_piece_idx << " started" << ENDL();
                            uint32_t first_tile_row_in_mm_M_block = 0;
                            uint32_t first_chunk_col_in_tiles = 0;
                            uint32_t first_mm_core_idx = 0;
                            uint32_t global_tile_idx;
                            uint32_t slice_tile_idx;
                            uint32_t effective_chunk_piece_size = mm_block_ht * effective_chunk_width_in_tiles;
                            get_next_tile_coordinates(
                                first_tile_row_in_mm_M_block,
                                first_chunk_col_in_tiles,
                                first_mm_core_idx,
                                effective_worker_id,
                                effective_chunk_piece_size,
                                effective_chunk_width_in_tiles,
                                mm_block_ht);
                            uint32_t tiles_to_read = how_many_tiles_to_read_formula(
                                first_tile_row_in_mm_M_block,
                                first_chunk_col_in_tiles,
                                first_mm_core_idx,
                                effective_advance_by_tiles,
                                last_mm_core_idx,
                                effective_chunk_piece_size,
                                effective_chunk_width_in_tiles);
                            DPRINT << "tiles_to_read: " << tiles_to_read << ENDL();

                            while (tiles_to_read > 0) {
                                uint32_t tiles_to_read_in_this_step = std::min(tiles_to_read, tile_granularity);
                                tiles_to_read -= tiles_to_read_in_this_step;

                                DPRINT << "Waiting for tiles in the output buffer" << ENDL();
                                cb_wait_front(cb_output_id, tile_granularity);
                                DPRINT << "OK done waiting for tiles in the output buffer" << ENDL();
                                size_t l1_read_addr = get_read_ptr(cb_output_id);

                                uint32_t tiles_remaining_in_step = tiles_to_read_in_this_step;
                                while (tiles_remaining_in_step > 0) {
                                    uint32_t tiles_to_put_in_current_packet =
                                        (i < (ring_size - 1))
                                            ? std::min(tiles_remaining_in_step, num_tiles_to_write_per_packet)
                                            : 1;
                                    tiles_remaining_in_step -= tiles_to_put_in_current_packet;

                                    auto tile_indices = coordinates_to_tile_indices(
                                        first_tile_row_in_mm_M_block,
                                        first_chunk_col_in_tiles,
                                        first_mm_core_idx,
                                        chunk_piece_idx,
                                        m_block_iter,
                                        chunk_idx,
                                        N_block_wt,
                                        tiles_ht_per_core,
                                        mm_block_ht,
                                        chunk_width_in_tiles,
                                        actual_slice_idx,
                                        slice_Wt,
                                        input_tensor_Wt);
                                    slice_tile_idx = tile_indices.slice;
                                    global_tile_idx = tile_indices.global;

                                    get_next_tile_coordinates(
                                        first_tile_row_in_mm_M_block,
                                        first_chunk_col_in_tiles,
                                        first_mm_core_idx,
                                        effective_advance_by_tiles,
                                        effective_chunk_piece_size,
                                        effective_chunk_width_in_tiles,
                                        mm_block_ht);
                                    DPRINT << "global_tile_idx: " << global_tile_idx << ENDL();

                                    if (i < (ring_size - 1)) {
                                        auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                            intermediate_addrgen, global_tile_idx, 0);

                                        switch (tiles_to_put_in_current_packet) {
                                            case 2: {
                                                auto tile_indices_two = coordinates_to_tile_indices(
                                                    first_tile_row_in_mm_M_block,
                                                    first_chunk_col_in_tiles,
                                                    first_mm_core_idx,
                                                    chunk_piece_idx,
                                                    m_block_iter,
                                                    chunk_idx,
                                                    N_block_wt,
                                                    tiles_ht_per_core,
                                                    mm_block_ht,
                                                    chunk_width_in_tiles,
                                                    actual_slice_idx,
                                                    slice_Wt,
                                                    input_tensor_Wt);
                                                get_next_tile_coordinates(
                                                    first_tile_row_in_mm_M_block,
                                                    first_chunk_col_in_tiles,
                                                    first_mm_core_idx,
                                                    effective_advance_by_tiles,
                                                    effective_chunk_piece_size,
                                                    effective_chunk_width_in_tiles,
                                                    mm_block_ht);

                                                auto noc_address1 =
                                                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                        intermediate_addrgen, tile_indices_two.global, 0);
                                                fabric_unicast_noc_scatter_write_with_state<
                                                    UnicastScatterWriteUpdateMask::DstAddrs>(
                                                    &mux_connection_handle,
                                                    pkt_scatter_hdr,
                                                    l1_read_addr,
                                                    NocUnicastScatterCommandHeader({noc_address0, noc_address1}));
                                                l1_read_addr += page_size * 2;
                                                break;
                                            }
                                            case 1:
                                            default: {
                                                fabric_unicast_noc_unicast_write_with_state<
                                                    UnicastWriteUpdateMask::DstAddr>(
                                                    &mux_connection_handle,
                                                    pkt_unicast_hdr,
                                                    l1_read_addr,
                                                    NocUnicastCommandHeader{noc_address0});
                                                l1_read_addr += page_size;
                                                break;
                                            }
                                        }
                                        noc_async_writes_flushed();
                                    } else {
                                        uint32_t output_tile_id = output_tile_id_start + slice_tile_idx;
                                        uint64_t local_noc_addr = get_noc_addr(output_tile_id, output_addrgen);
                                        noc_async_write(l1_read_addr, local_noc_addr, page_size);
                                        l1_read_addr += page_size;
                                    }
                                }
                                noc_async_write_barrier();
                                cb_pop_front(cb_output_id, tile_granularity);
                                DPRINT << "tiles_read" << ENDL();
                            }

                            DPRINT << "chunk_piece_idx: " << chunk_piece_idx << " done" << ENDL();
                        }

                        // Signal reader after all chunk_piece_idx tiles for this ring iteration are written
                        if (i < (ring_size - 1)) {
                            uint64_t out_ready_sem_noc_addr_in_pkt =
                                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                                &mux_connection_handle,
                                pkt_hdr_seminc,
                                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                            noc_async_writes_flushed();
                        } else {
                            noc_async_write_barrier();
                        }

                        if (direction) {
                            slice_idx--;
                        } else {
                            slice_idx++;
                        }
                        DPRINT << "ring iteration: " << i << " done" << ENDL();
                    }
                    DPRINT << "chunk_idx: " << chunk_idx << " done" << ENDL();
                }
                DPRINT << "m_block_iter: " << m_block_iter << " done" << ENDL();
            }

            // Signal batch done and wait for all other chips to finish before next batch
            uint64_t batch_ready_sem_noc_addr_in_pkt =
                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                &mux_connection_handle,
                pkt_hdr_mcastseminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
            noc_async_writes_flushed();

            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ring_size - 1);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);
            DPRINT << "batch: " << b << " done" << ENDL();
        }

        noc_async_write_barrier();
        noc_async_atomic_barrier();

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
    }

    noc_async_write_barrier();
}
