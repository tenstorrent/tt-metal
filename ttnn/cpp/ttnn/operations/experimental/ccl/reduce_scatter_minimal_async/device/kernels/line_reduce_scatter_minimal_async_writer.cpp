// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
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
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(2);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(5);
constexpr uint32_t input_num_pages = get_compile_time_arg_val(6);
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t output_batch_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t output_channel_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(11);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(12);
constexpr uint32_t slice_C = get_compile_time_arg_val(13);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(14);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(15);
constexpr uint32_t dim = get_compile_time_arg_val(16);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(17);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(21);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(22);

constexpr uint32_t num_ct_args = 23;

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

namespace detail {

inline bool do_forward_sync(const bool is_forward) {
    if constexpr (!sync_with_other_direction) {
        return false;
    }

    return is_forward;
}
}  // namespace detail
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
    uint32_t fwd_bwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool is_forward = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_device_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_targets_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const bool do_final_reduction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
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

    const auto& unicast_route_info = (is_forward) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (is_forward) ? forward_multicast_route_info : backward_multicast_route_info;

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
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_addrgen = {
        .bank_base_address = intermediate_address, .shard_array = intermediate_mapping_table};

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
    experimental::ShardedAddrGen<output_tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address, page_size);
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

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_connect_finish(*mux_connection_handle);
    }

    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();

    if (use_barrier_sem) {
        if (num_targets_in_direction) {
            ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_seminc, multicast_route_info);
            fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                pkt_hdr_seminc,
                static_cast<uint8_t>(multicast_route_info.start_distance_in_hops),
                static_cast<uint8_t>(multicast_route_info.range_hops),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    0,                           // ignore
                    static_cast<uint32_t>(1)});  // increment 1

            // multicast to both the forward and backward worker on all devices in your line that your write to
            // device going in the same direction
            uint64_t same_direction_barrier_sem_noc_addr_in_pkt =
                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                mux_connection_handle,
                pkt_hdr_seminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same_direction_barrier_sem_noc_addr_in_pkt, 0});

            // device going in the opposite direction
            uint64_t opposite_direction_barrier_sem_noc_addr_in_pkt =
                safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                mux_connection_handle,
                pkt_hdr_seminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_direction_barrier_sem_noc_addr_in_pkt, 0});
        }

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    // only initialize if we're actually going to send anything over fabric
    if (num_targets_in_direction) {
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);

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
    }

    const uint32_t intermediate_full_offset = is_forward ? 0 : input_num_pages;

    uint32_t chunk_count = 0;
    for (uint32_t b = 0; b < input_tensor_B; b++) {
        int slice_idx = is_forward ? ring_size - 1 : 0;

        uint32_t batch_offset = input_batch_num_pages * b;
        for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
            const uint32_t cb_output_id = is_first_device_in_direction ? cb_reader_output_id : cb_compute_output_id;
            chunk_count = 0;

            uint32_t intermediate_tile_id_start;
            if constexpr (dim == 3) {
                intermediate_tile_id_start = slice_idx * slice_Wt + batch_offset + intermediate_full_offset;
            } else if constexpr (dim == 2) {
                intermediate_tile_id_start = slice_idx * slice_Ht * slice_Wt + batch_offset + intermediate_full_offset;
            } else if constexpr (dim == 1) {
                intermediate_tile_id_start =
                    slice_idx * slice_C * slice_Ht * slice_Wt + batch_offset + intermediate_full_offset;
            } else {
                ASSERT(false);
            }
            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t intermediate_pages_read_in_row = start_pages_read_in_row;
                uint32_t intermediate_row_offset = start_row_offset;

                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                // Write to remote intermediate buffer
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        uint32_t num_pages_to_write = std::min(contig_pages_advanced, num_pages_to_read - j);

                        uint32_t first_intermediate_tile_id =
                            intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                        auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_addrgen, first_intermediate_tile_id, 0);

                        intermediate_pages_read_in_row++;
                        if (intermediate_pages_read_in_row == slice_Wt) {
                            intermediate_row_offset += input_tensor_Wt;
                            intermediate_pages_read_in_row -= slice_Wt;
                        }

                        if (num_pages_to_write == 1) {
                            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                                mux_connection_handle,
                                pkt_unicast_hdr,
                                l1_read_addr,
                                NocUnicastCommandHeader{noc_address0});
                            l1_read_addr += page_size;
                        } else if (num_pages_to_write == 2) {
                            uint32_t second_intermediate_tile_id =
                                intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;

                            intermediate_pages_read_in_row++;
                            if (intermediate_pages_read_in_row == slice_Wt) {
                                intermediate_row_offset += input_tensor_Wt;
                                intermediate_pages_read_in_row -= slice_Wt;
                            }

                            auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                intermediate_addrgen, second_intermediate_tile_id, 0);
                            fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                                mux_connection_handle,
                                pkt_scatter_hdr,
                                l1_read_addr,
                                NocUnicastScatterCommandHeader({noc_address0, noc_address1}));
                            l1_read_addr += page_size * 2;
                        } else {
                            ASSERT(false);
                        }
                        noc_async_writes_flushed();
                        tiles_read += num_pages_to_write;
                    }
                    cb_pop_front(cb_output_id, tile_granularity);

                    chunk_count++;
                    if (chunk_count % chunks_per_sync == 0) {
                        // 2. unicast output ready semaphore
                        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                            mux_connection_handle,
                            pkt_hdr_seminc,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                    }
                }
                intermediate_tile_id_start += input_channel_num_pages;
            }

            if (chunk_count % chunks_per_sync != 0) {
                // 2. unicast output ready semaphore
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle,
                    pkt_hdr_seminc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
            }

            // Next slice idx
            if (is_forward) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }

        // Do write of final reduction and sync local FWD/BWD cores
        if (do_final_reduction) {
            // Write output
            uint32_t output_tile_id_start = b * output_batch_num_pages;
            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_wait_front(cb_compute_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_compute_output_id);
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint32_t output_tile_id = output_tile_id_start + tiles_read;
                        uint64_t local_noc_addr = get_noc_addr(output_tile_id, output_addrgen);
                        noc_async_write(l1_read_addr, local_noc_addr, page_size);
                        l1_read_addr += page_size;
                        tiles_read++;
                    }

                    if (detail::do_forward_sync(is_forward)) {
                        noc_async_write_barrier();
                    } else {
                        noc_async_writes_flushed();
                    }
                    cb_pop_front(cb_compute_output_id, tile_granularity);
                    if (detail::do_forward_sync(is_forward)) {
                        // Tell local backwards reader that it can proceed
                        uint64_t fwd_bwd_sem_noc_addr =
                            safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, fwd_bwd_sem_addr, 0);
                        noc_semaphore_inc(fwd_bwd_sem_noc_addr, 1);
                    }
                }
                output_tile_id_start += output_channel_num_pages;
            }
            noc_async_write_barrier();
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

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
    noc_async_atomic_barrier();
}
