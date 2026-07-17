// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(2);
constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_inputs = get_compile_time_arg_val(4);
constexpr uint32_t unicast_route_arg0 = get_compile_time_arg_val(5);
constexpr uint32_t unicast_route_arg1 = get_compile_time_arg_val(6);

constexpr uint32_t page_size_base_idx = 7;

void kernel_main() {
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> output_batch_head_stride_pages;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_id_start;
    std::array<uint32_t, num_inputs> input_tile_id_end;
    std::array<uint32_t, num_inputs> input_origin_page;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        output_batch_head_stride_pages[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_start[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_end[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_origin_page[input_idx] = get_arg_val<uint32_t>(arg_idx++);
    }

    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx);
    arg_idx += num_inputs;
    auto output_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);
    size_t fabric_args_idx = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(fabric_args_idx);

    Noc noc_obj;
    CircularBuffer cb_packet_header(reserved_packet_header_cb_id);
    CircularBuffer cb_output(cb_output_id);

    // packet header cb
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);

    // pre-populate packet headers
    constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info = {
        .dst_mesh_id = static_cast<uint16_t>(unicast_route_arg0),
        .dst_chip_id = static_cast<uint16_t>(unicast_route_arg1)};

    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, unicast_route_info);

    fabric_connection.open();

    tt::tt_fabric::WorkerToFabricEdmSender& fabric_direction_connection = fabric_connection.get_forward_connection();
    const uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        // Send the local halo tail directly to the next device's compact output buffer.

        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
            uint32_t tiles_read = input_tile_id_start[input_idx];
            const uint32_t tiles_to_read = input_tile_id_end[input_idx];
            const uint32_t output_batch_head_base = bh_idx * output_batch_head_stride_pages[input_idx];
            while (tiles_read < tiles_to_read) {
                const uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                cb_output.wait_front(packet_size_in_pages);
                const size_t l1_read_addr = cb_output.get_read_ptr();
                const uint32_t tile_id = output_batch_head_base + tiles_read - input_origin_page[input_idx];

                if (num_pages_to_read == 2) {
                    const uint32_t second_tile_id = tile_id + 1;
                    const bool is_last_source_packet = input_idx + 1 == num_inputs &&
                                                       bh_idx + 1 == input_batch_head_count[input_idx] &&
                                                       tiles_read + num_pages_to_read >= tiles_to_read;

                    if (is_last_source_packet) {
                        const uint64_t first_noc_addr = output_addrgens[input_idx].get_noc_addr(tile_id);
                        const uint64_t second_noc_addr = output_addrgens[input_idx].get_noc_addr(second_tile_id);
                        pkt_hdr->to_noc_fused_unicast_scatter_write_atomic_inc(
                            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                                {first_noc_addr, second_noc_addr},
                                out_ready_sem_noc_addr_in_pkt,
                                {static_cast<uint16_t>(output_page_size)},
                                1,
                                true},
                            output_page_size * 2);
                        perform_payload_send(fabric_direction_connection, l1_read_addr, output_page_size * 2, pkt_hdr);
                        noc_async_writes_flushed();
                    } else {
                        scatter_fabric_write_unidir(
                            tile_id,
                            second_tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }
                } else {
                    ASSERT(num_pages_to_read == 1);

                    const bool is_last_source_packet = input_idx + 1 == num_inputs &&
                                                       bh_idx + 1 == input_batch_head_count[input_idx] &&
                                                       tiles_read + num_pages_to_read >= tiles_to_read;
                    if (is_last_source_packet) {
                        tt::tt_fabric::linear::to_noc_fused_unicast_write_atomic_inc(
                            output_page_size,
                            pkt_hdr,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 1, true},
                            tile_id,
                            output_addrgens[input_idx]);
                        perform_payload_send(fabric_direction_connection, l1_read_addr, output_page_size, pkt_hdr);
                        noc_async_writes_flushed();
                    } else {
                        fabric_write_unidir(
                            tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }
                }

                tiles_read += num_pages_to_read;
                cb_output.pop_front(packet_size_in_pages);
            }
        }
    }

    noc_obj.async_write_barrier();
    // Drain in-flight writes BEFORE closing the EDM connections.
    noc_obj.async_atomic_barrier();
    noc_obj.async_write_barrier();

    fabric_connection.close();
}
