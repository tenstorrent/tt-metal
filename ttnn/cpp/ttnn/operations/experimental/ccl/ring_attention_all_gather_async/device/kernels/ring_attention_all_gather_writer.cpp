// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#ifdef USE_WORKER_MUX
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#endif

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);
constexpr bool dynamic_alternate = get_compile_time_arg_val(8);
constexpr bool fuse_op = get_compile_time_arg_val(9);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(10));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(11);
constexpr uint32_t num_inputs = get_compile_time_arg_val(12);
constexpr bool direction = get_compile_time_arg_val(13);  // 1 is forward, 0 is backward

#ifdef USE_WORKER_MUX
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(21);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(22);
#endif

void kernel_main() {
    constexpr uint32_t page_size_base_idx = 14;
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx, page_size_base_idx);
    arg_idx += num_inputs;
    auto output_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);
    size_t arg_for_fab = arg_idx;

#ifdef USE_WORKER_MUX
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
#else
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
#endif

    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;

    if constexpr (fuse_op) {
#ifndef USE_WORKER_MUX
        arg_idx = arg_for_fab;
#endif
        op_signaler_sender = OpSignaler(arg_idx);
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
    fabric_set_unicast_route<false>(pkt_hdr, 1);

#ifdef USE_WORKER_MUX
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_sender;
    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection = nullptr;
    if (mux_connection_valid) {
        mux_sender = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
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
        fabric_direction_connection = reinterpret_cast<tt::tt_fabric::WorkerToFabricEdmSender*>(&mux_sender);
    }
    if (mux_connection_valid) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
        tt::tt_fabric::fabric_client_connect(mux_sender);
    }
#else
    fabric_connection.open();

    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection =
        fabric_connection.is_logically_connected() ? (direction == 1 ? &fabric_connection.get_backward_connection()
                                                                     : &fabric_connection.get_forward_connection())
                                                   : nullptr;
#endif
    constexpr uint32_t num_targets_in_direction =
        direction == 1 ? num_targets_backward_direction : num_targets_forward_direction;

    uint32_t slice_writes = 0;

    uint32_t row_offset = 0;
    uint32_t tile_id_start = my_chip_id * input_tensor_Wt;
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        /**
         * Write out the local slice to forward and backward devices
         * Note that it is not copied to local output buffer. This is because
         * the fused op (RingJointAttention) reads from the input buffer directly
         * when accessing the local slice. This is a performance optimization
         * to remove startup latency from the fused op.
         */
        uint32_t pages_read_in_row = input_tile_id_start % input_tensor_Wt;
        uint32_t row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
        uint32_t tiles_read = input_tile_id_start;
        uint32_t tiles_to_read = input_tile_id_end;
        uint32_t tile_id_start = my_chip_id * input_tensor_Wt;
        if (gather_dim == 3) {
            tile_id_start = my_chip_id * input_tensor_Wt;
        } else {
            tile_id_start = my_chip_id * input_tensor_Ht * input_tensor_Wt;
        }
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                cb_wait_front(cb_output_id, packet_size_in_pages);
                const size_t l1_read_addr_base = get_read_ptr(cb_output_id);
                size_t l1_read_addr = l1_read_addr_base;

                // for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;

                pages_read_in_row++;
                if (pages_read_in_row >= input_tensor_Wt) {
                    row_offset += output_tensor_Wt;
                    pages_read_in_row = 0;
                }

                if (num_pages_to_read == 2) {
                    uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;

                    if constexpr (num_targets_in_direction) {
                        scatter_fabric_write_unidir(
                            tile_id,
                            second_tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }

                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }
                } else {
                    ASSERT(num_pages_to_read == 1);

                    if constexpr (num_targets_in_direction) {
                        // Has valid targets to send to
                        fabric_write_unidir(
                            tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }
                }

                tiles_read += num_pages_to_read;

                cb_pop_front(cb_output_id, packet_size_in_pages);
            }
            tile_id_start += output_tensor_Wt * output_tensor_Ht;
            tiles_read = input_tile_id_start;
            tiles_to_read = input_tile_id_end;
            pages_read_in_row = input_tile_id_start % input_tensor_Wt;
            row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
        }
    }

    noc_async_write_barrier();
    // increment locally
    if constexpr (fuse_op && direction == 1) {
        /**
         * Synchronize and signal that the local tensor slice is available
         *
         * While the fused op will not wait on this "local write done" increment,
         * the fused op signaler will account for it in future waits.
         */
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    auto* pkt_hdr_sem_inc = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1

    // Write the unicast packet
    if constexpr (num_targets_in_direction) {
        fabric_direction_connection->wait_for_empty_write_slot();
        fabric_set_unicast_route<false>(pkt_hdr_sem_inc, 1);
        fabric_direction_connection->send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }

    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if constexpr (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if constexpr (topology == Topology::Ring) {
        if constexpr (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    while (slice_writes < writes_expected) {
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

        int slice_chip_id;
        uint32_t actual_slice_chip_id;
        if constexpr (direction == 1) {
            slice_chip_id = my_chip_id + slice_writes + 1;
            actual_slice_chip_id = (slice_chip_id >= (int)ring_size) ? slice_chip_id - ring_size : slice_chip_id;
        } else {
            slice_chip_id = my_chip_id - slice_writes - 1;
            actual_slice_chip_id = (slice_chip_id < 0) ? ring_size + slice_chip_id : slice_chip_id;
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            uint32_t tiles_read = input_tile_id_start;
            uint32_t tiles_to_read = input_tile_id_end;
            uint32_t tile_id_start = actual_slice_chip_id * input_tensor_Wt;
            uint32_t row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
            uint32_t pages_read_in_row = (input_tile_id_start % input_tensor_Wt);
            uint32_t slice_Wt = input_tensor_Wt;
            uint32_t stride_Wt = output_tensor_Wt;

            if (gather_dim == 3) {
                tile_id_start = actual_slice_chip_id * input_tensor_Wt;
            } else {
                tile_id_start = actual_slice_chip_id * input_tensor_Ht * input_tensor_Wt;
            }
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    cb_wait_front(cb_output_id, packet_size_in_pages);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    uint32_t first_tile_id = tile_id_start + row_offset + pages_read_in_row;
                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }

                    if (num_pages_to_read == 2) {
                        uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }

                        scatter_fabric_write_unidir(
                            first_tile_id,
                            second_tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    } else {
                        ASSERT(num_pages_to_read == 1);
                        fabric_write_unidir(
                            first_tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }

                    tiles_read += num_pages_to_read;
                    cb_pop_front(cb_output_id, packet_size_in_pages);
                }
                tile_id_start += output_tensor_Wt * output_tensor_Ht;
                tiles_read = input_tile_id_start;
                tiles_to_read = input_tile_id_end;
                row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
                pages_read_in_row = (input_tile_id_start % input_tensor_Wt);
            }
        }

        // 2. unicast output ready semaphore forward
        fabric_direction_connection->wait_for_empty_write_slot();
        fabric_set_unicast_route<false>(pkt_hdr_sem_inc, 1);
        fabric_direction_connection->send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));

        slice_writes++;
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
#ifdef USE_WORKER_MUX
    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(mux_sender);
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
#else
    fabric_connection.close();
#endif
    noc_async_write_barrier();
}
