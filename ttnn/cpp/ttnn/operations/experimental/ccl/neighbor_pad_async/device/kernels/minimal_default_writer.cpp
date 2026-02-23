// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

constexpr bool is_first_chip = get_compile_time_arg_val(0);
constexpr bool is_last_chip = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr bool direction = get_compile_time_arg_val(3);
constexpr bool is_padding_zeros = get_compile_time_arg_val(4);
constexpr uint32_t stick_size = get_compile_time_arg_val(5);
// Output TensorAccessorArgs start at index 6 (variable length)
constexpr auto dst_ct_args = TensorAccessorArgs<6>();
constexpr uint32_t ct_after_dst = dst_ct_args.next_compile_time_args_offset();
constexpr bool use_l1_intermediate = get_compile_time_arg_val(ct_after_dst);
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(ct_after_dst + 1);
constexpr bool handle_incoming_writes = get_compile_time_arg_val(ct_after_dst + 2);
constexpr bool sems_are_program_local = get_compile_time_arg_val(ct_after_dst + 3);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t outer_dim_offset_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    // CreateSemaphore returns an ID; convert to L1 address via get_semaphore().
    // GlobalSemaphore.address() is already an absolute L1 address — no conversion needed.
    if constexpr (sems_are_program_local) {
        out_ready_sem = get_semaphore(out_ready_sem);
        barrier_sem = get_semaphore(barrier_sem);
    }
    const uint32_t target_device_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t opposite_target_device_offset = get_arg_val<uint32_t>(arg_idx++);
    // Phase 2 barrier signal targets (0 for 1D, >0 for 2D)
    const uint32_t num_phase2_signal_targets = get_arg_val<uint32_t>(arg_idx++);
    uint8_t signal_noc_x[2];
    uint8_t signal_noc_y[2];
    uint32_t signal_sem_addr[2];
    for (uint32_t st = 0; st < 2; st++) {
        signal_noc_x[st] = get_arg_val<uint32_t>(arg_idx++);
        signal_noc_y[st] = get_arg_val<uint32_t>(arg_idx++);
        signal_sem_addr[st] = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    }
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    uint32_t read_size = stick_size;
    const auto dst_accessor = TensorAccessor(dst_ct_args, output_tensor_address, stick_size);

    // L1 intermediate: discover the recv CB base address (same on neighbor device due to identical program)
    uint32_t recv_buf_base = 0;
    if constexpr (use_l1_intermediate) {
        recv_buf_base = get_write_ptr(recv_cb_id);
    }

    // pre-populate packet headers
    auto pkt_hdr = PacketHeaderPool::allocate_header();
    pkt_hdr->to_chip_unicast(target_device_offset);
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

    fabric_connection.open();

    // Barrier semaphore
    if (use_barrier_sem) {
        auto pkt_hdr_barrier_sem_inc = PacketHeaderPool::allocate_header();

        if (!is_last_chip) {
            // unicast output ready semaphore
            uint64_t barrier_sem_noc_addr_in_pkt =
                safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
            pkt_hdr_barrier_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                barrier_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
            // Write the unicast packet
            if (direction) {
                if (fabric_connection.has_backward_connection()) {
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    pkt_hdr_barrier_sem_inc->to_chip_unicast(opposite_target_device_offset);
                    fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_barrier_sem_inc, sizeof(PACKET_HEADER_TYPE));
                }
            } else {
                if (fabric_connection.has_forward_connection()) {
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    pkt_hdr_barrier_sem_inc->to_chip_unicast(opposite_target_device_offset);
                    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_barrier_sem_inc, sizeof(PACKET_HEADER_TYPE));
                }
            }
            noc_async_writes_flushed();
        }

        if (!is_last_chip) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 1);
        }
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    uint32_t outer_dim_offset = outer_dim_offset_start_id;
    uint32_t l1_buf_offset = 0;  // L1 intermediate: accumulates across all outer_dims (no reuse)
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        if (is_first_chip) {
            if (!is_padding_zeros) {
                // Replicate a slice of 1 from input to output
                uint32_t dst_stick_id = 0;
                if (direction) {
                    dst_stick_id = (output_halo_dim_size - padding) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    dst_stick_id = stick_start_id;
                }
                dst_stick_id += outer_dim_offset;
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    cb_wait_front(cb_output_id, 1);
                    uint32_t l1_read_addr = get_read_ptr(cb_output_id);

                    for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                        uint64_t dst_noc_addr =
                            get_noc_addr(dst_stick_id + pad_id * num_sticks_per_halo_dim, dst_accessor);
                        noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
                    }
                    dst_stick_id++;

                    noc_async_write_barrier();
                    cb_pop_front(cb_output_id, 1);
                }
            } else {
                uint32_t dst_stick_id = 0;
                if (direction) {
                    dst_stick_id = (output_halo_dim_size - padding) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    dst_stick_id = stick_start_id;
                }
                dst_stick_id += outer_dim_offset;
                cb_wait_front(cb_output_id, 1);
                uint32_t l1_read_addr = get_read_ptr(cb_output_id);
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                        uint64_t dst_noc_addr =
                            get_noc_addr(dst_stick_id + pad_id * num_sticks_per_halo_dim, dst_accessor);
                        noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
                    }
                    dst_stick_id++;

                    noc_async_write_barrier();
                }
                cb_pop_front(cb_output_id, 1);
            }
        }

        if (!is_last_chip) {
            // Read the "end" of each slice into the CB to write to the neighbor
            for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                uint32_t dst_stick_id = 0;
                if (direction) {
                    dst_stick_id =
                        (output_halo_dim_size - (padding - pad_id)) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    dst_stick_id = pad_id * num_sticks_per_halo_dim + stick_start_id;
                }
                dst_stick_id += outer_dim_offset;
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    cb_wait_front(cb_output_id, 1);
                    uint32_t l1_read_addr = get_read_ptr(cb_output_id);

                    uint64_t dst_noc_addr;
                    if constexpr (use_l1_intermediate) {
                        // Target the receiver core's L1 buffer instead of DRAM
                        dst_noc_addr = safe_get_noc_addr(
                            out_ready_sem_noc0_x, out_ready_sem_noc0_y, recv_buf_base + l1_buf_offset, 0);
                        l1_buf_offset += stick_size;
                    } else {
                        dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor, 0, 0);
                    }

                    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, stick_size);
                    if (direction) {
                        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                        fabric_connection.get_backward_connection()
                            .send_payload_without_header_non_blocking_from_address(l1_read_addr, stick_size);
                        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                            (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                    } else {
                        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                        fabric_connection.get_forward_connection()
                            .send_payload_without_header_non_blocking_from_address(l1_read_addr, stick_size);
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
            pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                out_ready_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
            // Write the unicast packet
            if (direction) {
                fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                pkt_hdr_sem_inc->to_chip_unicast(target_device_offset);
                fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr_sem_inc, sizeof(PACKET_HEADER_TYPE));

            } else {
                fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                pkt_hdr_sem_inc->to_chip_unicast(target_device_offset);
                fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr_sem_inc, sizeof(PACKET_HEADER_TYPE));
            }
            noc_async_writes_flushed();
        }

        // No local interior copy in this kernel. Dedicated local-copy kernels handle that work.

        outer_dim_offset += (num_sticks_per_halo_dim * output_halo_dim_size);
    }

    fabric_connection.close();

    // Ensure all DRAM writes are complete before signaling Phase 2.
    noc_async_write_barrier();

    // Signal Phase 2 W fabric reader cores that Phase 1 writes are complete
    for (uint32_t st = 0; st < num_phase2_signal_targets; st++) {
        uint64_t sem_noc_addr = get_noc_addr(signal_noc_x[st], signal_noc_y[st], signal_sem_addr[st]);
        noc_semaphore_inc(sem_noc_addr, 1);
    }
    // Ensure sem inc signals are delivered before kernel exits.
    noc_async_write_barrier();

    // Incoming writes: pop sticks that the paired reader pushed from its L1 recv buffer
    // (fabric-delivered padding from neighbor) and write to output DRAM.
    // Used by both H fabric writers (incoming H halo) and W fabric writers (incoming W padding).
    if constexpr (handle_incoming_writes) {
        if (!is_first_chip) {
            uint32_t inc_offset = outer_dim_offset_start_id;
            for (uint32_t od = 0; od < outer_dim_size; od++) {
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    uint32_t row_offset;
                    if (direction) {
                        row_offset = (output_halo_dim_size - padding + pad_id) * num_sticks_per_halo_dim;
                    } else {
                        row_offset = pad_id * num_sticks_per_halo_dim;
                    }
                    uint32_t dst_stick_id = inc_offset + row_offset + stick_start_id;

                    for (uint32_t iter = 0; iter < num_sticks_to_read; iter++) {
                        cb_wait_front(cb_output_id, 1);
                        uint32_t l1_read_addr = get_read_ptr(cb_output_id);
                        uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor);
                        noc_async_write(l1_read_addr, dst_noc_addr, stick_size);

                        noc_async_write_barrier();
                        cb_pop_front(cb_output_id, 1);
                        dst_stick_id++;
                    }
                }
                inc_offset += num_sticks_per_halo_dim * output_halo_dim_size;
            }

            noc_async_write_barrier();
            // Signal Phase 2 W fabric reader cores that incoming writes are complete
            for (uint32_t st = 0; st < num_phase2_signal_targets; st++) {
                uint64_t sem_noc_addr = get_noc_addr(signal_noc_x[st], signal_noc_y[st], signal_sem_addr[st]);
                noc_semaphore_inc(sem_noc_addr, 1);
            }
            noc_async_write_barrier();
        }
    }
}
