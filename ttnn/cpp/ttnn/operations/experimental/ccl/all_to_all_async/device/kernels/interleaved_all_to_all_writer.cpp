// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_ring_id = get_arg(args::my_ring_id);
constexpr uint32_t ring_size = get_arg(args::ring_size);
constexpr uint32_t num_packet_headers_storable = get_arg(args::num_packet_headers_storable);
constexpr uint32_t packet_size_in_pages = get_arg(args::packet_size_in_pages);
constexpr uint32_t tensor0_page_size = get_arg(args::tensor0_page_size);
constexpr uint32_t num_targets_forward_direction = get_arg(args::num_targets_forward_direction);
constexpr uint32_t num_targets_backward_direction = get_arg(args::num_targets_backward_direction);
constexpr bool dynamic_alternate = get_arg(args::dynamic_alternate);
constexpr uint32_t chunk_granularity = get_arg(args::chunk_granularity);
constexpr uint32_t contig_pages_advanced = get_arg(args::contig_pages_advanced);
constexpr uint32_t N_DRAM_BANKS = get_arg(args::N_DRAM_BANKS);

constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;

constexpr uint32_t wait_sem_value = (ring_size - 1);

constexpr uint32_t NUM_SENDERS = ring_size - 1;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t global_semaphore_addr = GlobalSemaphore(global_sem::chunk_ready).address();
    uint32_t out_row_tiles = get_arg(args::out_row_tiles);
    uint32_t out_col_tiles = get_arg(args::out_col_tiles);
    uint32_t out_row_start = get_arg(args::out_row_start);
    uint32_t out_col_start = get_arg(args::out_col_start);
    uint32_t input_shard_row_tiles = get_arg(args::input_shard_row_tiles);
    uint32_t input_shard_col_tiles = get_arg(args::input_shard_col_tiles);
    bool wait_output_semaphore = get_arg(args::wait_output_semaphore);
    bool reset_global_semaphore = get_arg(args::reset_global_semaphore);
    uint32_t receiver_core_x = get_arg(args::receiver_core_x);
    uint32_t receiver_core_y = get_arg(args::receiver_core_y);

    auto fabric_connection = FabricConnectionManager::build_from_optional_bindings<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
        fabric::get_token_if_present<"forward">(), fabric::get_token_if_present<"backward">());

    Noc noc_obj;
    DataflowBuffer dfb_packet_header(dfb::packet_headers);
    DataflowBuffer dfb0(dfb::data);

    uint32_t out_row_end = out_row_start + input_shard_row_tiles;
    uint32_t out_col_end = out_col_start + input_shard_col_tiles;

    // packet header dfb
    dfb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_forward = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);
    dfb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr_backward = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);
    dfb_packet_header.reserve_back(1);
    auto packet_header_buffer_seminc = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);

    auto intermediate_tensor_addrgen = TensorAccessor(tensor::intermediate);
    auto output_tensor_addrgen = TensorAccessor(tensor::output);

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    // Set up seminc packet
    uint64_t output_semaphore_noc_addr_in_pkt =
        safe_get_noc_addr(receiver_core_x, receiver_core_y, global_semaphore_addr, 0);
    volatile PACKET_HEADER_TYPE* pkt_hdr_seminc =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);

    pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        output_semaphore_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1

    volatile PACKET_HEADER_TYPE* cur_pkt_header;

    bool cur_is_forward = num_targets_forward_direction > num_targets_backward_direction;
    uint32_t forward_hops = 1;
    uint32_t backward_hops = 1;
    for (uint32_t i = 0; i < ring_size - 1; ++i) {
        if (forward_hops == num_targets_forward_direction + 1) {
            cur_is_forward = false;
        }
        if (backward_hops == num_targets_backward_direction + 1) {
            cur_is_forward = true;
        }
        uint32_t& cur_hops = cur_is_forward ? forward_hops : backward_hops;
        uint32_t dst_ring_id =
            cur_is_forward ? (my_ring_id + cur_hops) % ring_size : (my_ring_id - cur_hops + ring_size) % ring_size;
        tt::tt_fabric::WorkerToFabricEdmSender& cur_connection =
            cur_is_forward ? fabric_connection.get_forward_connection() : fabric_connection.get_backward_connection();
        cur_pkt_header = cur_is_forward ? pkt_hdr_forward : pkt_hdr_backward;
        fabric_set_unicast_route<false>(cur_pkt_header, cur_hops);

        const uint32_t my_relative_ring_id = (my_ring_id < dst_ring_id) ? my_ring_id : my_ring_id - 1;
        uint32_t packet_id = 0;

        uint32_t prev_chunk_id = 0;

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += packet_size_in_pages) {
                dfb0.wait_front(packet_size_in_pages);
                size_t l1_read_addr = dfb0.get_read_ptr();
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, packet_size_in_pages);

                constexpr uint32_t payload_size_bytes = contig_pages_advanced * tensor0_page_size;

                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    // Calculate the tile id of the first tile in the pair to send. Guaranteed to be in the same bank.
                    uint32_t global_id = my_relative_ring_id + packet_id * NUM_SENDERS;
                    uint32_t first_id =
                        (global_id % N_DRAM_BANKS) + contig_pages_advanced * N_DRAM_BANKS * (global_id / N_DRAM_BANKS);

                    packet_id++;  // increment packet_id for chunk calculation

                    uint32_t current_chunk_id = packet_id / chunk_granularity;
                    if (current_chunk_id != prev_chunk_id) {
                        // Fused payload write with atomic inc
                        perform_atomic_fabric_write(
                            cur_pkt_header,
                            first_id,
                            intermediate_tensor_addrgen,
                            cur_connection,
                            l1_read_addr,
                            payload_size_bytes,
                            output_semaphore_noc_addr_in_pkt,
                            1,
                            false);

                        prev_chunk_id = current_chunk_id;
                    } else {
                        // Unicast payload write
                        tt::tt_fabric::linear::to_noc_unicast_write(
                            payload_size_bytes, cur_pkt_header, first_id, intermediate_tensor_addrgen);
                        perform_payload_send(cur_connection, l1_read_addr, payload_size_bytes, cur_pkt_header);
                    }

                    noc_obj.async_writes_flushed();

                    // Advance local read address
                    l1_read_addr += payload_size_bytes;
                }

                dfb0.pop_front(packet_size_in_pages);
            }
        }

        // Handle final incomplete chunk
        if (packet_id % chunk_granularity != 0) {
            fabric_set_unicast_route<false>(pkt_hdr_seminc, cur_hops);
            cur_connection.wait_for_empty_write_slot();
            cur_connection.send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }

        cur_hops++;
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_obj.async_write_barrier();
}
