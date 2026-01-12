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
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(5);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(6);

constexpr uint32_t initial_ct_idx = 7;

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t intermediate_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t semaphore_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t semaphore_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t op_semaphore = get_arg_val<uint32_t>(arg_idx++);
    size_t pre_op_barrier_semaphore = get_arg_val<uint32_t>(arg_idx++);
    size_t post_op_barrier_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 1 is forward, 0 is backward
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<initial_ct_idx>();
    constexpr uint32_t intermediate_ct_offset = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_address, page_size);

    constexpr uint32_t output_addrgen_ct_idx = initial_ct_idx + intermediate_ct_offset;

#ifdef OUTPUT_IS_SHARDED
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(output_addrgen_ct_idx),
        get_compile_time_arg_val(output_addrgen_ct_idx + 1),
        get_compile_time_arg_val(output_addrgen_ct_idx + 2),
        get_compile_time_arg_val(output_addrgen_ct_idx + 3),
        get_compile_time_arg_val(output_addrgen_ct_idx + 4),
        get_compile_time_arg_val(output_addrgen_ct_idx + 5),
        get_compile_time_arg_val(output_addrgen_ct_idx + 6)>;

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<output_addrgen_ct_idx>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address, page_size);
#endif

    // hardcoded constants
    constexpr uint32_t num_connections = 1;
    constexpr uint32_t tile_granularity = 2;

    size_t arg_for_fab = arg_idx;
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    // pre-populate packet headers
    auto scatter_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto unicast_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto seminc_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto mcastseminc_route_id = PacketHeaderPool::allocate_header_n(num_connections);

    // TODO: (GR) cleanup
    uint32_t unicast_num_hops = 1;
    uint32_t multicast_start_distance_in_hops = 1;
    uint8_t unicast_distances[] = {static_cast<uint8_t>(unicast_num_hops)};
    uint8_t multicast_starts[] = {static_cast<uint8_t>(multicast_start_distance_in_hops)};
    uint8_t multicast_ranges[] = {static_cast<uint8_t>(ring_size - 1)};

    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        fabric_connection,
        scatter_route_id,
        unicast_distances,
        NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)}),
        page_size * 2);

    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric_connection, unicast_route_id, unicast_distances, nullptr, page_size);

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        seminc_route_id,
        unicast_distances,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        mcastseminc_route_id,
        multicast_starts,
        multicast_ranges,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    // init barrier - multicast to entire ring of workers going in the same direction
    uint64_t pre_op_barrier_semaphore_noc_addr_in_pkt =
        safe_get_noc_addr(semaphore_noc0_x, semaphore_noc0_y, pre_op_barrier_semaphore, 0);
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        mcastseminc_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{pre_op_barrier_semaphore_noc_addr_in_pkt, 0});
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pre_op_barrier_semaphore), ring_size - 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pre_op_barrier_semaphore), 0);

    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        uint32_t actual_slice_idx;
        if (direction) {
            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
        } else {
            actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
        }

        uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;
        if (i < (ring_size - 1)) {
            uint32_t intermediate_tile_id_start = actual_slice_idx * slice_Wt;
            uint32_t intermediate_pages_read_in_row = start_pages_read_in_row;
            uint32_t intermediate_row_offset = start_row_offset;

            uint32_t tiles_read = start_tiles_read;
            uint32_t tiles_to_read = start_tiles_to_read;

            if (!direction) {
                for (uint32_t k = 0; k < tile_granularity; ++k) {
                    intermediate_pages_read_in_row++;
                    if (intermediate_pages_read_in_row == slice_Wt) {
                        intermediate_row_offset += input_tensor_Wt;
                        intermediate_pages_read_in_row -= slice_Wt;
                    }
                }
                tiles_read += tile_granularity;
            }

            while (tiles_read < tiles_to_read) {
                cb_wait_front(cb_output_id, tile_granularity);
                size_t intermediate_l1_read_addr = get_read_ptr(cb_output_id);

                uint32_t intermediate_tile_one_id =
                    intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                intermediate_pages_read_in_row++;
                if (intermediate_pages_read_in_row == slice_Wt) {
                    intermediate_row_offset += input_tensor_Wt;
                    intermediate_pages_read_in_row -= slice_Wt;
                }

                uint32_t intermediate_tile_two_id =
                    intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
                intermediate_pages_read_in_row++;
                if (intermediate_pages_read_in_row == slice_Wt) {
                    intermediate_row_offset += input_tensor_Wt;
                    intermediate_pages_read_in_row -= slice_Wt;
                }

                auto intermediate_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                    intermediate_addrgen, intermediate_tile_one_id, 0);

                auto intermediate_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                    intermediate_addrgen, intermediate_tile_two_id, 0);

                // op hardcoded for each worker handling even multiple of 2 tiles, so always use scatter_write
                fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                    fabric_connection,
                    scatter_route_id,
                    intermediate_l1_read_addr,
                    NocUnicastScatterCommandHeader({intermediate_noc_address_one, intermediate_noc_address_two}));
                tiles_read += 2;

                noc_async_writes_flushed();
                cb_pop_front(cb_output_id, tile_granularity);

                uint64_t op_semaphore_noc_addr_in_pkt =
                    safe_get_noc_addr(semaphore_noc0_x, semaphore_noc0_y, op_semaphore, 0);
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_connection,
                    seminc_route_id,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{op_semaphore_noc_addr_in_pkt, 0});

                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                if (tiles_remaining_to_read > 0) {
                    for (uint32_t k = 0; k < tile_granularity; ++k) {
                        intermediate_pages_read_in_row++;
                        if (intermediate_pages_read_in_row == slice_Wt) {
                            intermediate_row_offset += input_tensor_Wt;
                            intermediate_pages_read_in_row -= slice_Wt;
                        }
                    }
                    tiles_read += tile_granularity;
                }
            }

            noc_async_writes_flushed();
        } else {
            uint32_t tiles_read = start_tiles_read;
            uint32_t tiles_to_read = start_tiles_to_read;

            if (!direction) {
                tiles_read += tile_granularity;
            }

            while (tiles_read < tiles_to_read) {
                cb_wait_front(cb_output_id, tile_granularity);
                size_t output_l1_read_addr = get_read_ptr(cb_output_id);
                for (uint32_t j = 0; j < tile_granularity; ++j) {
                    uint64_t output_local_noc_addr = get_noc_addr(tiles_read, output_addrgen);
                    noc_async_write(output_l1_read_addr, output_local_noc_addr, page_size);
                    output_l1_read_addr += page_size;
                    tiles_read++;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_output_id, tile_granularity);

                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                if (tiles_remaining_to_read > 0) {
                    tiles_read += tile_granularity;
                }
            }
        }

        // next slice idx
        if (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }

    // end barrier
    uint64_t post_op_barrier_semaphore_noc_addr_in_pkt =
        safe_get_noc_addr(semaphore_noc0_x, semaphore_noc0_y, post_op_barrier_semaphore, 0);
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        mcastseminc_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{post_op_barrier_semaphore_noc_addr_in_pkt, 0});
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(post_op_barrier_semaphore), ring_size - 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(post_op_barrier_semaphore), 0);

    close_connections(fabric_connection);

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
