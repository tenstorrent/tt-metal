// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

constexpr bool is_first_chip = get_compile_time_arg_val(0);
constexpr bool is_last_chip = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(3);
constexpr bool direction = get_compile_time_arg_val(4);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t stick_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_outer_dim = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    constexpr auto src_args = TensorAccessorArgs<5>();
    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(src_args, input_tensor_address, stick_size);

    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    pkt_hdr->to_chip_unicast(1);

    fabric_connection.open();

    if (is_first_chip) {
        // Replicate a slice of 1 from input to output
        uint32_t dst_stick_id = 0;
        if (direction) {
            dst_stick_id = (output_outer_dim_size - padding) * num_sticks_per_outer_dim + stick_start_id;
        } else {
            dst_stick_id = stick_start_id;
        }
        for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
            cb_wait_front(cb_output_id, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_output_id);

            for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                uint64_t dst_noc_addr = get_noc_addr(dst_stick_id + pad_id * num_sticks_per_outer_dim, dst_accessor);
                noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            }
            dst_stick_id++;

            noc_async_write_barrier();
            cb_pop_front(cb_output_id, 1);
        }
    }

    if (!is_last_chip) {
        // Read the "end" of each slice into the CB to write to the neighbor
        for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
            uint32_t dst_stick_id = 0;
            if (direction) {
                dst_stick_id = (output_outer_dim_size - (padding - pad_id)) * num_sticks_per_outer_dim + stick_start_id;
            } else {
                dst_stick_id = pad_id * num_sticks_per_outer_dim + stick_start_id;
            }
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                cb_wait_front(cb_output_id, 1);
                uint32_t l1_read_addr = get_read_ptr(cb_output_id);

                uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor, 0, 0);

                pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, stick_size);
                if (direction) {
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
                        l1_read_addr, stick_size);
                    fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                        (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                } else {
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
                        l1_read_addr, stick_size);
                    fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                        (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                }
                noc_async_writes_flushed();

                dst_stick_id++;

                noc_async_write_barrier();
                cb_pop_front(cb_output_id, 1);
            }
        }

        // unicast output ready semaphore
        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
        auto* pkt_hdr_sem_inc = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
        pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});
        // Write the unicast packet
        if (direction) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        } else {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
    }

    // Copy the entire input
    if (direction) {
        for (uint32_t t = 0; t < input_outer_dim_size; t++) {
            uint32_t dst_stick_id = (t + padding_left) * num_sticks_per_outer_dim + stick_start_id;
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                cb_wait_front(cb_output_id, 1);
                uint32_t l1_read_addr = get_read_ptr(cb_output_id);

                uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor);
                noc_async_write(l1_read_addr, dst_noc_addr, stick_size);

                dst_stick_id++;

                noc_async_write_barrier();
                cb_pop_front(cb_output_id, 1);
            }
        }
    }

    fabric_connection.close();

    noc_async_write_barrier();
}
