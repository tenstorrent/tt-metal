// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id_dir0 = get_compile_time_arg_val(2);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);  // 1 is forward, 0 is backward
constexpr uint32_t gather_dim = get_compile_time_arg_val(9);
constexpr uint32_t input_batch_head_count = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(11);
constexpr uint32_t input_tensor_Ht = get_compile_time_arg_val(12);
constexpr uint32_t input_tensor_C = get_compile_time_arg_val(13);
constexpr uint32_t output_tensor_Wt = get_compile_time_arg_val(14);
constexpr uint32_t output_tensor_Ht = get_compile_time_arg_val(15);
constexpr uint32_t output_tensor_C = get_compile_time_arg_val(16);
constexpr uint32_t input_tile_id_start = get_compile_time_arg_val(17);
constexpr uint32_t input_tile_id_end = get_compile_time_arg_val(18);
constexpr uint32_t start_pages_read_in_row = get_compile_time_arg_val(19);
constexpr uint32_t start_row_offset = get_compile_time_arg_val(20);
constexpr bool fuse_op = get_compile_time_arg_val(21);
constexpr uint32_t chunks_per_sync = get_compile_time_arg_val(22);
constexpr uint32_t reverse = get_compile_time_arg_val(23) == 1;

constexpr bool is_termination_master = get_compile_time_arg_val(24);
constexpr uint8_t fabric_mux_x_dir0 = get_compile_time_arg_val(25);
constexpr uint8_t fabric_mux_y_dir0 = get_compile_time_arg_val(26);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(27);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(28);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(29);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(30);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(31);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(32);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(33);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(34);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(35);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(36);

constexpr uint32_t cb_output_id_dir1 = get_compile_time_arg_val(37);
constexpr uint8_t fabric_mux_x_dir1 = get_compile_time_arg_val(38);
constexpr uint8_t fabric_mux_y_dir1 = get_compile_time_arg_val(39);

constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info_dir0 =
    ccl_routing_utils::get_line_unicast_route_info_from_args<40>();
constexpr ccl_routing_utils::line_multicast_route_info_t barrier_multicast_route_info_dir0 =
    ccl_routing_utils::get_line_multicast_route_info_from_args<40 + ccl_routing_utils::num_line_unicast_args>();

constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info_dir1 =
    ccl_routing_utils::get_line_unicast_route_info_from_args<40 + (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args)>();
constexpr ccl_routing_utils::line_multicast_route_info_t barrier_multicast_route_info_dir1 =
    ccl_routing_utils::get_line_multicast_route_info_from_args<40 + (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args) + ccl_routing_utils::num_line_unicast_args>();

inline constexpr uint32_t sharded_args_start_idx =
    40 + (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args) * 2;

void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_dir0 = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_dir1 = get_arg_val<uint32_t>(arg_idx++);

    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address_dir0 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address_dir0 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address_dir0 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address_dir0 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mux_clients = get_arg_val<uint32_t>(arg_idx++);

    uint32_t local_fabric_mux_status_address_dir1 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address_dir1 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address_dir1 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address_dir1 = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    // DPRINT << "Writer arg_idx" << (uint32_t)arg_idx << ENDL();

#ifdef OUTPUT_IS_SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(sharded_args_start_idx),   // Memory layout
        get_compile_time_arg_val(sharded_args_start_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(sharded_args_start_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(sharded_args_start_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(sharded_args_start_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(sharded_args_start_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(sharded_args_start_idx + 6)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = mapping_table};

    arg_idx += rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<sharded_args_start_idx>();
    const auto output_addrgen = TensorAccessor(output_tensor_args, output_address, page_size);
#endif
    DPRINT << "Writer Setup Starting" << ENDL();
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle_dir0;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle_dir1;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection_dir0;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection_dir1;
    if (mux_connection_valid) {
        mux_connection_dir0 = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
            fabric_mux_x_dir0,
            fabric_mux_y_dir0,
            fabric_mux_channel_id,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_channel_buffer_size_bytes,
            fabric_mux_channel_base_address,
            fabric_mux_connection_info_address,
            fabric_mux_connection_handshake_address,
            fabric_mux_flow_control_address,
            fabric_mux_buffer_index_address,
            local_flow_control_address_dir0,
            local_teardown_address_dir0,
            local_buffer_index_address_dir0);
        mux_connection_dir1 = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
            fabric_mux_x_dir1,
            fabric_mux_y_dir1,
            fabric_mux_channel_id,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_channel_buffer_size_bytes,
            fabric_mux_channel_base_address,
            fabric_mux_connection_info_address,
            fabric_mux_connection_handshake_address,
            fabric_mux_flow_control_address,
            fabric_mux_buffer_index_address,
            local_flow_control_address_dir1,
            local_teardown_address_dir1,
            local_buffer_index_address_dir1);
        mux_connection_handle_dir0 = &mux_connection_dir0;
        mux_connection_handle_dir1 = &mux_connection_dir1;
    } else {
        mux_connection_handle_dir0 = nullptr;
        mux_connection_handle_dir1 = nullptr;
    }
    DPRINT << "Writer Setup Done" << ENDL();
    if (mux_connection_valid) {
        // need to wait for fabric mux to be ready to accept connections
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x_dir0, fabric_mux_y_dir0, fabric_mux_status_address, local_fabric_mux_status_address_dir0);
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x_dir1, fabric_mux_y_dir1, fabric_mux_status_address, local_fabric_mux_status_address_dir1);
    }

    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;
    uint32_t self_write_done_semaphore_addr;
    if constexpr (fuse_op) {
        self_write_done_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
        op_signaler_sender = OpSignaler(arg_idx);
    }

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_connect(*mux_connection_handle_dir0);
        tt::tt_fabric::fabric_client_connect(*mux_connection_handle_dir1);
    }

    // pre-populate packet headers
    auto pkt_scatter_hdr_dir0 = PacketHeaderPool::allocate_header();
    auto pkt_scatter_hdr_dir1 = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr_dir0 = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr_dir1 = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem_inc_dir0 = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem_inc_dir1 = PacketHeaderPool::allocate_header();

    DPRINT << "Writer use_barrier_sem" << (uint32_t)use_barrier_sem << ENDL();
    if (use_barrier_sem) {
        if constexpr (num_targets_forward_direction) {
            // only initialize if we're actually going to send something over fabric
            ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_sem_inc_dir0, barrier_multicast_route_info_dir0);
            fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                pkt_hdr_sem_inc_dir0,
                static_cast<uint8_t>(barrier_multicast_route_info_dir0.start_distance_in_hops),
                static_cast<uint8_t>(barrier_multicast_route_info_dir0.range_hops),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    0,  // ignore
                    static_cast<uint32_t>(1)});

            if constexpr (topology == Topology::Linear) {
                // multicast to both the forward and backward worker on all devices that you write to.
                // this only executes if the worker actually sends something over fabric (i.e. the writers
                // on the end of the line pointing outward don't issue sem incs)

                // device going in the same direction
                uint64_t same_direction_barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir0,
                    pkt_hdr_sem_inc_dir0,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same_direction_barrier_sem_noc_addr_in_pkt, 0});

                // device going in the opposite direction
                // uint64_t opposite_direction_barrier_sem_noc_addr_in_pkt =
                //     safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0);
                // fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                //     mux_connection_handle,
                //     pkt_hdr_sem_inc,
                //     tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_direction_barrier_sem_noc_addr_in_pkt, 0});

            } else if constexpr (topology == Topology::Ring) {
                // multicast to entire ring of workers going in the same direction
                DPRINT << "Writer Ring barrier dir0" << ENDL();
                uint64_t barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir0,
                    pkt_hdr_sem_inc_dir0,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
                DPRINT << "Writer Ring barrier sent dir0" << ENDL();
            } else {
                ASSERT(false);
            }
        }
        DPRINT << "Writer Sem start dir0" << ENDL();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        DPRINT << "Writer Sem end dir0" << ENDL();
        if constexpr (num_targets_backward_direction) {
            // only initialize if we're actually going to send something over fabric

            ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_sem_inc_dir1, barrier_multicast_route_info_dir1);
            fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                pkt_hdr_sem_inc_dir1,
                static_cast<uint8_t>(barrier_multicast_route_info_dir1.start_distance_in_hops),
                static_cast<uint8_t>(barrier_multicast_route_info_dir1.range_hops),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    0,  // ignore
                    static_cast<uint32_t>(1)});

            if constexpr (topology == Topology::Linear) {
                // multicast to both the forward and backward worker on all devices that you write to.
                // this only executes if the worker actually sends something over fabric (i.e. the writers
                // on the end of the line pointing outward don't issue sem incs)

                // device going in the same direction
                uint64_t same_direction_barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir1,
                    pkt_hdr_sem_inc_dir1,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same_direction_barrier_sem_noc_addr_in_pkt, 0});


                // device going in the opposite direction
                // uint64_t opposite_direction_barrier_sem_noc_addr_in_pkt =
                //     safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0);
                // fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                //     mux_connection_handle,
                //     pkt_hdr_sem_inc,
                //     tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_direction_barrier_sem_noc_addr_in_pkt, 0});

            } else if constexpr (topology == Topology::Ring) {
                // multicast to entire ring of workers going in the same direction
                uint64_t barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
                fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir1,
                    pkt_hdr_sem_inc_dir1,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
            } else {
                ASSERT(false);
            }
        }
        DPRINT << "Writer Sem start dir1" << ENDL();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        DPRINT << "Writer Sem end dir1" << ENDL();
    }

    uint32_t slice_writes_dir0 = 0;
    uint32_t slice_writes_dir1 = 0;

    // Write out the local slice to both DRAM and forward and backward
    uint32_t pages_read_in_row_dir0 = start_pages_read_in_row;
    uint32_t pages_read_in_row_dir1 = start_pages_read_in_row;
    uint32_t row_offset_dir0 = start_row_offset;
    uint32_t row_offset_dir1 = start_row_offset;
    uint32_t tiles_read_dir0 = input_tile_id_start;
    uint32_t tiles_read_dir1 = input_tile_id_start;
    uint32_t tiles_to_read_dir0 = input_tile_id_end;
    uint32_t tiles_to_read_dir1 = input_tile_id_end;

    uint32_t position = my_chip_id;
    if constexpr (reverse) {
        position = (ring_size - 1) - my_chip_id;
    }

    uint32_t tile_id_start_dir0;
    uint32_t tile_id_start_dir1;
    if constexpr (gather_dim == 3) {
        tile_id_start_dir0 = position * input_tensor_Wt;
        tile_id_start_dir1 = position * input_tensor_Wt;
    } else if constexpr (gather_dim == 2) {
        tile_id_start_dir0 = position * input_tensor_Ht * input_tensor_Wt;
        tile_id_start_dir1 = position * input_tensor_Ht * input_tensor_Wt;
    } else if constexpr (gather_dim == 1) {
        tile_id_start_dir0 = position * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
        tile_id_start_dir1 = position * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
    } else {
        tile_id_start_dir0 = position * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
        tile_id_start_dir1 = position * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
    }

    if constexpr (num_targets_forward_direction) {
        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr_dir0,
            static_cast<uint8_t>(unicast_route_info_dir0.distance_in_hops),
            NocUnicastScatterCommandHeader{
                {0, 0},  // ignore
                static_cast<uint16_t>(page_size)},
            page_size * 2);
        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr_dir0, static_cast<uint8_t>(unicast_route_info_dir0.distance_in_hops), nullptr, page_size);
        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_sem_inc_dir0,
            static_cast<uint8_t>(unicast_route_info_dir0.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0,  // ignore
                static_cast<uint32_t>(1)});
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr_dir0, unicast_route_info_dir0);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr_dir0, unicast_route_info_dir0);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc_dir0, unicast_route_info_dir0);
    }

    // only initialize if we're actually going to send something over fabric
    if constexpr (num_targets_backward_direction) {
        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr_dir1,
            static_cast<uint8_t>(unicast_route_info_dir1.distance_in_hops),
            NocUnicastScatterCommandHeader{
                {0, 0},  // ignore
                static_cast<uint16_t>(page_size)},
            page_size * 2);
        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr_dir1, static_cast<uint8_t>(unicast_route_info_dir1.distance_in_hops), nullptr, page_size);
        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_sem_inc_dir1,
            static_cast<uint8_t>(unicast_route_info_dir1.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0,  // ignore
                static_cast<uint32_t>(1)});
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr_dir1, unicast_route_info_dir1);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr_dir1, unicast_route_info_dir1);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc_dir1, unicast_route_info_dir1);
    }

    uint64_t out_ready_sem_noc_addr_in_pkt_dir0 =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_dir0, 0);
    uint64_t out_ready_sem_noc_addr_in_pkt_dir1 =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_dir1, 0);
    uint32_t num_channels_processed_in_current_batch = 0;
    uint32_t chunk_count_dir0 = 0;
    uint32_t chunk_count_dir1 = 0;
    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        chunk_count_dir0 = 0;

        while (tiles_read_dir0 < tiles_to_read_dir0) {
            DPRINT << "Writer tiles_read_dir0 < tiles_to_read_dir0 " << tiles_read_dir0 << " < " << tiles_to_read_dir0 << ENDL();
            uint32_t tiles_remaining_to_read = tiles_to_read_dir0 - tiles_read_dir0;
            uint32_t tiles_to_put_in_current_packet = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

            cb_wait_front(cb_output_id_dir0, num_tiles_to_write_per_packet);
            size_t l1_read_addr_dir0 = get_read_ptr(cb_output_id_dir0);

            uint32_t tile_one_id = tile_id_start_dir0 + row_offset_dir0 + pages_read_in_row_dir0;
            pages_read_in_row_dir0++;
            if (pages_read_in_row_dir0 >= input_tensor_Wt) {
                row_offset_dir0 += output_tensor_Wt;
                pages_read_in_row_dir0 = 0;
            }
            auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                output_addrgen, tile_one_id, 0);
            // Will have more cases once scatter-write supports more than 2 distinct addresses
            switch (tiles_to_put_in_current_packet) {
                case 2: {
                    uint32_t tile_two_id = tile_id_start_dir0 + row_offset_dir0 + pages_read_in_row_dir0;
                    pages_read_in_row_dir0++;
                    if (pages_read_in_row_dir0 >= input_tensor_Wt) {
                        row_offset_dir0 += output_tensor_Wt;
                        pages_read_in_row_dir0 = 0;
                    }

                    auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                        output_addrgen, tile_two_id, 0);
                    if constexpr (num_targets_forward_direction) {
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            mux_connection_handle_dir0,
                            pkt_scatter_hdr_dir0,
                            l1_read_addr_dir0,
                            NocUnicastScatterCommandHeader{{noc_address0, noc_address1}, 0});
                    }
                    uint64_t local_noc0_dest_noc_addr_tile_one = get_noc_addr(tile_one_id, output_addrgen);
                    uint64_t local_noc0_dest_noc_addr_tile_two = get_noc_addr(tile_two_id, output_addrgen);

                    noc_async_write(l1_read_addr_dir0, local_noc0_dest_noc_addr_tile_one, page_size);
                    noc_async_write(l1_read_addr_dir0 + page_size, local_noc0_dest_noc_addr_tile_two, page_size);
                    noc_async_write_barrier();
                    tiles_read_dir0 += 2;
                    break;
                }
                case 1:
                default: {
                    if constexpr (num_targets_forward_direction) {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            mux_connection_handle_dir0,
                            pkt_unicast_hdr_dir0,
                            l1_read_addr_dir0,
                            NocUnicastCommandHeader{noc_address0});
                    }
                    uint64_t local_noc0_dest_noc_addr = get_noc_addr(tile_one_id, output_addrgen);
                    noc_async_write(l1_read_addr_dir0, local_noc0_dest_noc_addr, page_size);
                    noc_async_write_barrier();
                    tiles_read_dir0++;
                    break;
                }
            }
            noc_async_writes_flushed();

            cb_pop_front(cb_output_id_dir0, num_tiles_to_write_per_packet);

            chunk_count_dir0++;
            if (chunk_count_dir0 % chunks_per_sync == 0) {
                // 2. unicast output ready semaphore
                if constexpr (num_targets_forward_direction) {
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        mux_connection_handle_dir0,
                        pkt_hdr_sem_inc_dir0,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir0, 0});
                }
            }
            noc_async_writes_flushed();
        }

        if (chunk_count_dir0 % chunks_per_sync != 0) {
            // Write the unicast packet
            if constexpr (num_targets_forward_direction) {
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir0,
                    pkt_hdr_sem_inc_dir0,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir0, 0});
            }
        }

        num_channels_processed_in_current_batch++;
        if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
            tile_id_start_dir0 += output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
        } else {
            tile_id_start_dir0 += output_tensor_Wt * output_tensor_Ht;
        }

        if (num_channels_processed_in_current_batch == input_tensor_C) {
            num_channels_processed_in_current_batch = 0;
        }

        tiles_read_dir0 = input_tile_id_start;
        tiles_to_read_dir0 = input_tile_id_end;
        pages_read_in_row_dir0 = start_pages_read_in_row;
        row_offset_dir0 = start_row_offset;
    }
    DPRINT << "Writer First Direction Done" << ENDL();


    // increment locally
    if constexpr (fuse_op && direction == 1) {
        // Synchronize and signal that the local tensor slice is available
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
        uint64_t self_write_done_semaphore_noc_addr =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, self_write_done_semaphore_addr, 0);
        noc_semaphore_inc(self_write_done_semaphore_noc_addr, 1);
    }

    uint32_t writes_expected_dir0 = 0;
    uint32_t writes_expected_dir1 = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (num_targets_backward_direction) {
            writes_expected_dir1 = num_targets_forward_direction;
        } else if constexpr (num_targets_forward_direction) {
            writes_expected_dir0 = num_targets_backward_direction;
        }
    } else if constexpr (topology == Topology::Ring) {
        // if constexpr (direction == 1) {
            writes_expected_dir1 = num_targets_backward_direction - 1;
        // } else {
            writes_expected_dir0 = num_targets_forward_direction - 1;
        // }
    }

    while (slice_writes_dir0 < writes_expected_dir0) {
        DPRINT << "Writer slice_writes_dir0 < writes_expected_dir0 " << (uint32_t)slice_writes_dir0 << " < " << (uint32_t)writes_expected_dir0 << ENDL();
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times
        int slice_chip_id_dir0;
        uint32_t actual_slice_chip_id_dir0;

        slice_chip_id_dir0 = my_chip_id - slice_writes_dir0 - 1;
        actual_slice_chip_id_dir0 = (slice_chip_id_dir0 < 0) ? ring_size + slice_chip_id_dir0 : slice_chip_id_dir0;
        if constexpr (reverse) {
            actual_slice_chip_id_dir0 = (ring_size - 1) - actual_slice_chip_id_dir0;
        }
        tiles_read_dir0 = input_tile_id_start;

        // uint32_t tiles_read_dir0 = input_tile_id_start;
        // uint32_t tiles_to_read_dir0 = input_tile_id_end;
        uint32_t tile_id_start_dir0;
        uint32_t row_offset_dir0 = start_row_offset;
        uint32_t pages_read_in_row_dir0 = start_pages_read_in_row;
        uint32_t slice_Wt = input_tensor_Wt;
        uint32_t stride_Wt = output_tensor_Wt;
        if constexpr (gather_dim == 3) {
            tile_id_start_dir0 = actual_slice_chip_id_dir0 * input_tensor_Wt;
        } else if constexpr (gather_dim == 2) {
            tile_id_start_dir0 = actual_slice_chip_id_dir0 * input_tensor_Ht * input_tensor_Wt;
        } else if constexpr (gather_dim == 1) {
            tile_id_start_dir0 = actual_slice_chip_id_dir0 * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
        } else {
            tile_id_start_dir0 = actual_slice_chip_id_dir0 * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
        }
        

        num_channels_processed_in_current_batch = 0;
        
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            chunk_count_dir0 = 0;
            DPRINT << "Writer Starting bh_idx " << (uint32_t)bh_idx << ENDL();
            while (tiles_read_dir0 < tiles_to_read_dir0) {
                DPRINT << "tiles_read_dir0 : " << (uint32_t)tiles_read_dir0 << " / " << (uint32_t)tiles_to_read_dir0 << ENDL();
                DPRINT << "Writer actual_slice_chip_id_dir0 : " << (uint32_t)actual_slice_chip_id_dir0 << ENDL();
                uint32_t tiles_remaining_to_read = tiles_to_read_dir0 - tiles_read_dir0;
                uint32_t tiles_to_put_in_current_packet =
                    std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
                // DPRINT << "Writer cb_wait_front dir0" << ENDL();
                cb_wait_front(cb_output_id_dir0, num_tiles_to_write_per_packet);
                // DPRINT << "Writer cb_wait_front dir0 done" << ENDL();
                size_t l1_read_addr = get_read_ptr(cb_output_id_dir0);
                DPRINT << "Writer l1_read_addr dir0" << (uint32_t)l1_read_addr << ENDL();

                uint32_t tile_one_id = tile_id_start_dir0 + row_offset_dir0 + pages_read_in_row_dir0;
                pages_read_in_row_dir0++;
                if (pages_read_in_row_dir0 >= input_tensor_Wt) {
                    row_offset_dir0 += output_tensor_Wt;
                    pages_read_in_row_dir0 = 0;
                }
                auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                    output_addrgen, tile_one_id, 0);
                // Will have more cases once scatter-write supports more than 2 distinct addresses
                switch (tiles_to_put_in_current_packet) {
                    case 2: {
                        uint32_t tile_two_id = tile_id_start_dir0 + row_offset_dir0 + pages_read_in_row_dir0;
                        pages_read_in_row_dir0++;
                        if (pages_read_in_row_dir0 >= slice_Wt) {
                            row_offset_dir0 += stride_Wt;
                            pages_read_in_row_dir0 = 0;
                        }

                        auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            output_addrgen, tile_two_id, 0);
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            mux_connection_handle_dir0,
                            pkt_scatter_hdr_dir0,
                            l1_read_addr,
                            NocUnicastScatterCommandHeader{{noc_address0, noc_address1}, 0});
                        tiles_read_dir0 += 2;
                        break;
                    }
                    case 1:
                    default: {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            mux_connection_handle_dir0,
                            pkt_unicast_hdr_dir0,
                            l1_read_addr,
                            NocUnicastCommandHeader{noc_address0});
                        tiles_read_dir0++;
                        break;
                    }
                }
                noc_async_writes_flushed();
                DPRINT << "Writer noc_async_writes_flushed dir0" << ENDL();

                cb_pop_front(cb_output_id_dir0, num_tiles_to_write_per_packet);
                chunk_count_dir0++;
                if (chunk_count_dir0 % chunks_per_sync == 0) {
                    // 2. unicast output ready semaphore
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        mux_connection_handle_dir0,
                        pkt_hdr_sem_inc_dir0,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir0, 0});
                }
                noc_async_writes_flushed();
                // DPRINT << "Writer tiles_read_dir0 : " << tiles_read_dir0 << " / " << tiles_to_read_dir0 << ENDL();
            }

            if (chunk_count_dir0 % chunks_per_sync != 0) {
                // Write the unicast packet
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir0,
                    pkt_hdr_sem_inc_dir0,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir0, 0});
            }

            num_channels_processed_in_current_batch++;
            if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
                tile_id_start_dir0 += output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
            } else {
                tile_id_start_dir0 += output_tensor_Wt * output_tensor_Ht;
            }

            if (num_channels_processed_in_current_batch == input_tensor_C) {
                num_channels_processed_in_current_batch = 0;
            }

            tiles_read_dir0 = input_tile_id_start;
            tiles_to_read_dir0 = input_tile_id_end;
            row_offset_dir0 = start_row_offset;
            pages_read_in_row_dir0 = start_pages_read_in_row;
        }

        slice_writes_dir0++;
    }

    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        chunk_count_dir1 = 0;

        while (tiles_read_dir1 < tiles_to_read_dir1) {
            DPRINT << "Writer tiles_read_dir1 < tiles_to_read_dir1 " << tiles_read_dir1 << " < " << tiles_to_read_dir1 << ENDL();
            uint32_t tiles_remaining_to_read = tiles_to_read_dir1 - tiles_read_dir1;
            uint32_t tiles_to_put_in_current_packet = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

            cb_wait_front(cb_output_id_dir1, num_tiles_to_write_per_packet);
            size_t l1_read_addr_dir1 = get_read_ptr(cb_output_id_dir1);

            uint32_t tile_one_id = tile_id_start_dir1 + row_offset_dir1 + pages_read_in_row_dir1;
            pages_read_in_row_dir1++;
            if (pages_read_in_row_dir1 >= input_tensor_Wt) {
                row_offset_dir1 += output_tensor_Wt;
                pages_read_in_row_dir1 = 0;
            }
            auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                output_addrgen, tile_one_id, 0);
            // Will have more cases once scatter-write supports more than 2 distinct addresses
            switch (tiles_to_put_in_current_packet) {
                case 2: {
                    uint32_t tile_two_id = tile_id_start_dir1 + row_offset_dir1 + pages_read_in_row_dir1;
                    pages_read_in_row_dir1++;
                    if (pages_read_in_row_dir1 >= input_tensor_Wt) {
                        row_offset_dir1 += output_tensor_Wt;
                        pages_read_in_row_dir1 = 0;
                    }

                    auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                        output_addrgen, tile_two_id, 0);
                    if constexpr (num_targets_backward_direction) {
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            mux_connection_handle_dir1,
                            pkt_scatter_hdr_dir1,
                            l1_read_addr_dir1,
                            NocUnicastScatterCommandHeader{{noc_address0, noc_address1}, 0});
                    }

                    uint64_t local_noc0_dest_noc_addr_tile_one = get_noc_addr(tile_one_id, output_addrgen);
                    uint64_t local_noc0_dest_noc_addr_tile_two = get_noc_addr(tile_two_id, output_addrgen);

                    noc_async_write(l1_read_addr_dir1, local_noc0_dest_noc_addr_tile_one, page_size);
                    noc_async_write(l1_read_addr_dir1 + page_size, local_noc0_dest_noc_addr_tile_two, page_size);
                    noc_async_write_barrier();
                    tiles_read_dir1 += 2;
                    break;
                }
                case 1:
                default: {
                    DPRINT << "Writer single write dir1" << ENDL();
                    if constexpr (num_targets_backward_direction) {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            mux_connection_handle_dir1,
                            pkt_unicast_hdr_dir1,
                            l1_read_addr_dir1,
                            NocUnicastCommandHeader{noc_address0});
                    }
                    DPRINT << "Writer single write dir1 done" << ENDL();

                    // uint64_t local_noc0_dest_noc_addr = get_noc_addr(tile_one_id, output_addrgen);
                    // noc_async_write(l1_read_addr_dir1, local_noc0_dest_noc_addr, page_size);
                    // noc_async_write_barrier();
                    tiles_read_dir1++;
                    break;
                }
            }
            noc_async_writes_flushed();

            cb_pop_front(cb_output_id_dir1, num_tiles_to_write_per_packet);

            chunk_count_dir1++;
            if (chunk_count_dir1 % chunks_per_sync == 0) {
                // 2. unicast output ready semaphore
                if constexpr (num_targets_backward_direction) {
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        mux_connection_handle_dir1,
                        pkt_hdr_sem_inc_dir1,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir1, 0});
                }
            }
            noc_async_writes_flushed();
        }

        if (chunk_count_dir1 % chunks_per_sync != 0) {
            // Write the unicast packet
            if constexpr (num_targets_backward_direction) {
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir1,
                    pkt_hdr_sem_inc_dir1,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir1, 0});
            }
        }

        num_channels_processed_in_current_batch++;
        if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
            tile_id_start_dir1 += output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
        } else {
            tile_id_start_dir1 += output_tensor_Wt * output_tensor_Ht;
        }

        if (num_channels_processed_in_current_batch == input_tensor_C) {
            num_channels_processed_in_current_batch = 0;
        }

        tiles_read_dir1 = input_tile_id_start;
        tiles_to_read_dir1 = input_tile_id_end;
        pages_read_in_row_dir1 = start_pages_read_in_row;
        row_offset_dir1 = start_row_offset;
    }
    DPRINT << "Writer Second Direction Done" << ENDL();
    while (slice_writes_dir1 < writes_expected_dir1) {
        DPRINT << "Writer slice_writes_dir1 < writes_expected_dir1 " << (uint32_t)slice_writes_dir1 << " < " << (uint32_t)writes_expected_dir1 << ENDL();
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times
        int slice_chip_id_dir1;
        uint32_t actual_slice_chip_id_dir1;
        // if constexpr (direction == 1) {
        slice_chip_id_dir1 = my_chip_id + slice_writes_dir1 + 1;
        actual_slice_chip_id_dir1 = (slice_chip_id_dir1 >= (int)ring_size) ? slice_chip_id_dir1 - ring_size : slice_chip_id_dir1;
        if constexpr (reverse) {
            actual_slice_chip_id_dir1 = (ring_size - 1) - actual_slice_chip_id_dir1;
        }
        tiles_read_dir1 = input_tile_id_start;

        // uint32_t tiles_read_dir1 = input_tile_id_start;
        // uint32_t tiles_to_read_dir1 = input_tile_id_end;
        uint32_t tile_id_start_dir1;
        uint32_t row_offset_dir1 = start_row_offset;
        uint32_t pages_read_in_row_dir1 = start_pages_read_in_row;
        uint32_t slice_Wt = input_tensor_Wt;
        uint32_t stride_Wt = output_tensor_Wt;
        if constexpr (gather_dim == 3) {
            tile_id_start_dir1 = actual_slice_chip_id_dir1 * input_tensor_Wt;
        } else if constexpr (gather_dim == 2) {
            tile_id_start_dir1 = actual_slice_chip_id_dir1 * input_tensor_Ht * input_tensor_Wt;
        } else if constexpr (gather_dim == 1) {
            tile_id_start_dir1 = actual_slice_chip_id_dir1 * input_tensor_C * input_tensor_Ht * input_tensor_Wt;
        } else {
            tile_id_start_dir1 = actual_slice_chip_id_dir1 * input_batch_head_count * input_tensor_Ht * input_tensor_Wt;
        }

        num_channels_processed_in_current_batch = 0;
        
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            chunk_count_dir1 = 0;
            DPRINT << "Writer Starting bh_idx " << (uint32_t)bh_idx << ENDL();
            while (tiles_read_dir1 < tiles_to_read_dir1) {
                DPRINT << "tiles_read_dir1 : " << (uint32_t)tiles_read_dir1 << " / " << (uint32_t)tiles_to_read_dir1 << ENDL();
                DPRINT << "Writer actual_slice_chip_id_dir1 : " << (uint32_t)actual_slice_chip_id_dir1 << ENDL();
                
                uint32_t tiles_remaining_to_read = tiles_to_read_dir1 - tiles_read_dir1;
                uint32_t tiles_to_put_in_current_packet =
                    std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

                cb_wait_front(cb_output_id_dir1, num_tiles_to_write_per_packet);
                size_t l1_read_addr = get_read_ptr(cb_output_id_dir1);
                DPRINT << "Writer l1_read_addr dir1" << (uint32_t)l1_read_addr << ENDL();

                uint32_t tile_one_id = tile_id_start_dir1 + row_offset_dir1 + pages_read_in_row_dir1;
                pages_read_in_row_dir1++;
                if (pages_read_in_row_dir1 >= input_tensor_Wt) {
                    row_offset_dir1 += output_tensor_Wt;
                    pages_read_in_row_dir1 = 0;
                }
                auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                    output_addrgen, tile_one_id, 0);
                // Will have more cases once scatter-write supports more than 2 distinct addresses
                switch (tiles_to_put_in_current_packet) {
                    case 2: {
                        uint32_t tile_two_id = tile_id_start_dir1 + row_offset_dir1 + pages_read_in_row_dir1;
                        pages_read_in_row_dir1++;
                        if (pages_read_in_row_dir1 >= slice_Wt) {
                            row_offset_dir1 += stride_Wt;
                            pages_read_in_row_dir1 = 0;
                        }

                        auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            output_addrgen, tile_two_id, 0);
                        fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                            mux_connection_handle_dir1,
                            pkt_scatter_hdr_dir1,
                            l1_read_addr,
                            NocUnicastScatterCommandHeader{{noc_address0, noc_address1}, 0});
                        tiles_read_dir1 += 2;
                        break;
                    }
                    case 1:
                    default: {
                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            mux_connection_handle_dir1,
                            pkt_unicast_hdr_dir1,
                            l1_read_addr,
                            NocUnicastCommandHeader{noc_address0});
                        tiles_read_dir1++;
                        break;
                    }
                }
                noc_async_writes_flushed();
                DPRINT << "Writer noc_async_writes_flushed dir1" << ENDL();

                cb_pop_front(cb_output_id_dir1, num_tiles_to_write_per_packet);
                chunk_count_dir1++;
                // DPRINT << "Writer cb_pop_front dir1" << ENDL();
                if (chunk_count_dir1 % chunks_per_sync == 0) {
                    // 2. unicast output ready semaphore
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        mux_connection_handle_dir1,
                        pkt_hdr_sem_inc_dir1,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir1, 0});
                }
                noc_async_writes_flushed();
                // DPRINT << "Writer tiles_read_dir1: " << tiles_read_dir1 << " / " << tiles_to_read_dir1 << ENDL();
            }

            if (chunk_count_dir1 % chunks_per_sync != 0) {
                // Write the unicast packet
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle_dir1,
                    pkt_hdr_sem_inc_dir1,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt_dir1, 0});
            }

            num_channels_processed_in_current_batch++;
            if (gather_dim == 1 && num_channels_processed_in_current_batch == input_tensor_C) {
                tile_id_start_dir1 += output_tensor_Wt * output_tensor_Ht * (output_tensor_C - input_tensor_C + 1);
            } else {
                tile_id_start_dir1 += output_tensor_Wt * output_tensor_Ht;
            }

            if (num_channels_processed_in_current_batch == input_tensor_C) {
                num_channels_processed_in_current_batch = 0;
            }

            tiles_read_dir1 = input_tile_id_start;
            tiles_to_read_dir1 = input_tile_id_end;
            row_offset_dir1 = start_row_offset;
            pages_read_in_row_dir1 = start_pages_read_in_row;
        }

        slice_writes_dir1++;
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    DPRINT << "Writer Done" << ENDL();
    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle_dir0);
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle_dir1);

        if constexpr (is_termination_master) {
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
            noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x_dir0, fabric_mux_y_dir0, fabric_mux_termination_signal_address);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x_dir1, fabric_mux_y_dir1, fabric_mux_termination_signal_address);
        } else {
            uint64_t dest_addr =
                safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
            noc_semaphore_inc(dest_addr, 1);
            noc_async_atomic_barrier();
        }
    }
    noc_async_write_barrier();
}
