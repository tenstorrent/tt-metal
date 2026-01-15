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
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
constexpr uint32_t input_slice_0_cb_id = get_compile_time_arg_val(4);
constexpr uint32_t input_slice_1_cb_id = get_compile_time_arg_val(5);
constexpr uint32_t input_slice_2_cb_id = get_compile_time_arg_val(6);
constexpr uint32_t input_slice_3_cb_id = get_compile_time_arg_val(7);
constexpr uint32_t input_slice_4_cb_id = get_compile_time_arg_val(8);
constexpr uint32_t input_slice_5_cb_id = get_compile_time_arg_val(9);
constexpr uint32_t input_slice_6_cb_id = get_compile_time_arg_val(10);
constexpr uint32_t input_slice_7_cb_id = get_compile_time_arg_val(11);
constexpr uint32_t compute_cb_id = get_compile_time_arg_val(12);

constexpr uint32_t initial_ct_idx = 13;

// NOTE: hardcoded for ring size of 8
constexpr uint32_t input_slice_cb_ids[8] = {
    input_slice_0_cb_id,
    input_slice_1_cb_id,
    input_slice_2_cb_id,
    input_slice_3_cb_id,
    input_slice_4_cb_id,
    input_slice_5_cb_id,
    input_slice_6_cb_id,
    input_slice_7_cb_id};

void kernel_main() {
    uint32_t arg_idx = 0;

    uint32_t intermediate_slice_0_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_slice_1_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_slice_2_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_slice_3_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_slice_4_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_slice_5_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_slice_6_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_slice_7_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_address = get_arg_val<uint32_t>(arg_idx++);

    const uint8_t op_semaphore_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t op_semaphore_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t op_semaphore = get_arg_val<uint32_t>(arg_idx++);

    const uint8_t pre_op_barrier_semaphore_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t pre_op_barrier_semaphore_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t pre_op_barrier_semaphore = get_arg_val<uint32_t>(arg_idx++);

    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 1 is forward, 0 is backward
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    // intermediate TensorAccessor
    constexpr uint32_t intermediate_slice_0_ct_val = initial_ct_idx;
    constexpr auto intermediate_slice_0_tensor_args = TensorAccessorArgs<intermediate_slice_0_ct_val>();
    constexpr uint32_t intermediate_slice_0_ct_offset = intermediate_slice_0_tensor_args.num_compile_time_args();
    const auto intermediate_slice_0_tensor_accesor =
        TensorAccessor(intermediate_slice_0_tensor_args, intermediate_slice_0_address, page_size);

    constexpr uint32_t intermediate_slice_1_ct_val = intermediate_slice_0_ct_val + intermediate_slice_0_ct_offset;
    constexpr auto intermediate_slice_1_tensor_args = TensorAccessorArgs<intermediate_slice_1_ct_val>();
    constexpr uint32_t intermediate_slice_1_ct_offset = intermediate_slice_1_tensor_args.num_compile_time_args();
    const auto intermediate_slice_1_tensor_accesor =
        TensorAccessor(intermediate_slice_1_tensor_args, intermediate_slice_1_address, page_size);

    constexpr uint32_t intermediate_slice_2_ct_val = intermediate_slice_1_ct_val + intermediate_slice_1_ct_offset;
    constexpr auto intermediate_slice_2_tensor_args = TensorAccessorArgs<intermediate_slice_2_ct_val>();
    constexpr uint32_t intermediate_slice_2_ct_offset = intermediate_slice_2_tensor_args.num_compile_time_args();
    const auto intermediate_slice_2_tensor_accesor =
        TensorAccessor(intermediate_slice_2_tensor_args, intermediate_slice_2_address, page_size);

    constexpr uint32_t intermediate_slice_3_ct_val = intermediate_slice_2_ct_val + intermediate_slice_2_ct_offset;
    constexpr auto intermediate_slice_3_tensor_args = TensorAccessorArgs<intermediate_slice_3_ct_val>();
    constexpr uint32_t intermediate_slice_3_ct_offset = intermediate_slice_3_tensor_args.num_compile_time_args();
    const auto intermediate_slice_3_tensor_accesor =
        TensorAccessor(intermediate_slice_3_tensor_args, intermediate_slice_3_address, page_size);

    constexpr uint32_t intermediate_slice_4_ct_val = intermediate_slice_3_ct_val + intermediate_slice_3_ct_offset;
    constexpr auto intermediate_slice_4_tensor_args = TensorAccessorArgs<intermediate_slice_4_ct_val>();
    constexpr uint32_t intermediate_slice_4_ct_offset = intermediate_slice_4_tensor_args.num_compile_time_args();
    const auto intermediate_slice_4_tensor_accesor =
        TensorAccessor(intermediate_slice_4_tensor_args, intermediate_slice_4_address, page_size);

    constexpr uint32_t intermediate_slice_5_ct_val = intermediate_slice_4_ct_val + intermediate_slice_4_ct_offset;
    constexpr auto intermediate_slice_5_tensor_args = TensorAccessorArgs<intermediate_slice_5_ct_val>();
    constexpr uint32_t intermediate_slice_5_ct_offset = intermediate_slice_5_tensor_args.num_compile_time_args();
    const auto intermediate_slice_5_tensor_accesor =
        TensorAccessor(intermediate_slice_5_tensor_args, intermediate_slice_5_address, page_size);

    constexpr uint32_t intermediate_slice_6_ct_val = intermediate_slice_5_ct_val + intermediate_slice_5_ct_offset;
    constexpr auto intermediate_slice_6_tensor_args = TensorAccessorArgs<intermediate_slice_6_ct_val>();
    constexpr uint32_t intermediate_slice_6_ct_offset = intermediate_slice_6_tensor_args.num_compile_time_args();
    const auto intermediate_slice_6_tensor_accesor =
        TensorAccessor(intermediate_slice_6_tensor_args, intermediate_slice_6_address, page_size);

    constexpr uint32_t intermediate_slice_7_ct_val = intermediate_slice_6_ct_val + intermediate_slice_6_ct_offset;
    constexpr auto intermediate_slice_7_tensor_args = TensorAccessorArgs<intermediate_slice_7_ct_val>();
    constexpr uint32_t intermediate_slice_7_ct_offset = intermediate_slice_7_tensor_args.num_compile_time_args();
    const auto intermediate_slice_7_tensor_accesor =
        TensorAccessor(intermediate_slice_7_tensor_args, intermediate_slice_7_address, page_size);

    // output TensorAccessor
    constexpr uint32_t output_ct_val = intermediate_slice_7_ct_val + intermediate_slice_7_ct_offset;
    constexpr auto output_tensor_args = TensorAccessorArgs<output_ct_val>();
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_address, page_size);

    // connect to fabric
    size_t arg_for_fab = arg_idx;
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    constexpr uint32_t num_connections = 1;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    // pre-populate packet headers
    auto unicast_scatter_write_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto unicast_sem_inc_route_id = PacketHeaderPool::allocate_header_n(num_connections);

    uint8_t unicast_num_hops[] = {static_cast<uint8_t>(1)};

    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        fabric_connection,
        unicast_scatter_write_route_id,
        unicast_num_hops,
        NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)}),
        page_size * 2);

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        unicast_sem_inc_route_id,
        unicast_num_hops,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                         // ignore
            static_cast<uint32_t>(1),  // increment 1
            false});                   // flush = false

    // pre op barrier
    // unicast to the opposite core in the direction you're sending
    // you need the device you're writing to, to have sent you and inc
    uint64_t pre_op_barrier_semaphore_noc_addr_in_pkt = safe_get_noc_addr(
        pre_op_barrier_semaphore_noc0_x, pre_op_barrier_semaphore_noc0_y, pre_op_barrier_semaphore, 0);
    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        unicast_sem_inc_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{pre_op_barrier_semaphore_noc_addr_in_pkt, 0, false});
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pre_op_barrier_semaphore), 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pre_op_barrier_semaphore), 0);

    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        uint32_t actual_slice_idx;
        if (direction) {
            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
        } else {
            actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
        }

        uint32_t reduced_cb_id = i == 0 ? input_slice_cb_ids[actual_slice_idx] : compute_cb_id;

        uint32_t tiles_read = start_tiles_read;
        uint32_t tiles_to_read = start_tiles_to_read;
        if (i < (ring_size - 1)) {
            while (tiles_read < tiles_to_read) {
                uint64_t intermediate_slice_noc_address_one;
                uint64_t intermediate_slice_noc_address_two;
                switch (actual_slice_idx) {
                    case 0:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_0_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_0_tensor_accesor, tiles_read++, 0);
                        break;
                    case 1:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_1_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_1_tensor_accesor, tiles_read++, 0);
                        break;
                    case 2:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_2_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_2_tensor_accesor, tiles_read++, 0);
                        break;
                    case 3:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_3_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_3_tensor_accesor, tiles_read++, 0);
                        break;
                    case 4:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_4_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_4_tensor_accesor, tiles_read++, 0);
                        break;
                    case 5:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_5_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_5_tensor_accesor, tiles_read++, 0);
                        break;
                    case 6:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_6_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_6_tensor_accesor, tiles_read++, 0);
                        break;
                    case 7:
                        intermediate_slice_noc_address_one = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_7_tensor_accesor, tiles_read++, 0);
                        intermediate_slice_noc_address_two = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_slice_7_tensor_accesor, tiles_read++, 0);
                        break;
                }

                // op hardcoded for each worker handling even multiple of 2 tiles, so always use scatter_write
                cb_wait_front(reduced_cb_id, tile_granularity);
                size_t intermediate_slice_l1_read_addr = get_read_ptr(reduced_cb_id);
                fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                    fabric_connection,
                    unicast_scatter_write_route_id,
                    intermediate_slice_l1_read_addr,
                    NocUnicastScatterCommandHeader(
                        {intermediate_slice_noc_address_one, intermediate_slice_noc_address_two}));

                noc_async_writes_flushed();
                cb_pop_front(reduced_cb_id, tile_granularity);

                // TODO: (GR) fuse
                uint64_t op_semaphore_noc_addr_in_pkt =
                    safe_get_noc_addr(op_semaphore_noc0_x, op_semaphore_noc0_y, op_semaphore, 0);
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    fabric_connection,
                    unicast_sem_inc_route_id,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{op_semaphore_noc_addr_in_pkt, 0, false});
            }

            noc_async_writes_flushed();
        } else {
            while (tiles_read < tiles_to_read) {
                cb_wait_front(reduced_cb_id, tile_granularity);
                size_t output_l1_read_addr = get_read_ptr(reduced_cb_id);
                for (uint32_t j = 0; j < tile_granularity; ++j) {
                    uint64_t output_local_noc_addr = get_noc_addr(tiles_read, output_tensor_accessor);
                    noc_async_write(output_l1_read_addr, output_local_noc_addr, page_size);
                    output_l1_read_addr += page_size;
                    tiles_read++;
                }

                noc_async_write_barrier();
                cb_pop_front(reduced_cb_id, tile_granularity);
            }
        }

        // next slice idx
        if (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }

    close_connections(fabric_connection);

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
