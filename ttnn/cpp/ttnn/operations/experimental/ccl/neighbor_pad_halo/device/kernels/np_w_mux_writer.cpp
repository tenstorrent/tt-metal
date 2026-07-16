// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric-MUX W writer for neighbor_pad_halo.
//
// Purpose: the single-worker-per-link W send caps at ~1.4 GB/s/link because the EDM exposes only one
// direct local-worker sender channel per link (fabric.cpp:226). To saturate the eth link (~12.5 GB/s
// Linear) multiple workers must feed it through a MUX core. This kernel is one such worker: it connects
// to a fabric-mux endpoint and routes ALL fabric ops (startup barrier atomic-inc, coalesced W-halo data,
// and the W recv-sem atomic-inc) through the mux connection, mirroring the all_gather worker-mux pattern
// (ttnn/.../all_gather_async/device/kernels/minimal_default_writer.cpp, USE_WORKER_MUX path).
//
// Scope: the coalesced W path across all W devices (edges included — a no-send direction is gated off
// in-kernel via has_send_neighbor). Each (link,direction) is served by N workers + 1 mux core; each
// worker owns a contiguous sub-range of this direction's W rows (split host-side). Bank-major coalescing
// matches np_writer's W-coalesce send (base+r, base+r+8, ... contiguous on one bank). Non-coalesced
// (per-stick) shapes use np_writer instead.

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include <cstdint>

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

// ---- Compile-time args ----
constexpr uint32_t send_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t stick_size = get_compile_time_arg_val(1);
constexpr auto dst_ct_args = TensorAccessorArgs<2>();
constexpr uint32_t ct_after_dst = dst_ct_args.next_compile_time_args_offset();
constexpr uint32_t W_COALESCE = get_compile_time_arg_val(ct_after_dst);
constexpr uint32_t NP_NUM_DRAM_BANKS = 8;
// Mux connection CT args (lockstep with FabricMuxConfig on the host).
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(ct_after_dst + 1);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(ct_after_dst + 2);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(ct_after_dst + 3);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(ct_after_dst + 4);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(ct_after_dst + 5);

void kernel_main() {
    // ---- Common runtime args ----
    const address_t output_tensor_address = get_common_arg_val<address_t>(1);
    const size_t neighbor_sem = get_common_arg_val<uint32_t>(2);

    // ---- Per-core runtime args ----
    uint32_t arg_idx = 0;
    const uint32_t base = get_arg_val<uint32_t>(arg_idx++);            // compact-buffer W-section base for this worker
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);  // this worker's row count
    const uint8_t neighbor_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t neighbor_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    auto unicast_route_info = ccl_routing_utils::line_unicast_route_info_t{
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++))};

    // Mux connection RT args — EXACT layout of ccl::fabric_mux_connection_rt_args (positions 0..16).
    const bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;                           // 0
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++) == 1;                          // 1
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);                                     // 2
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);                                     // 3
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);                   // 4
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);                // 5
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);           // 6
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);                   // 7
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);                   // 8
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);                            // 9
    const uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));         // 10
    const uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));  // 11
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));       // 12
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));           // 13
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));       // 14
    const uint8_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);                         // 15
    const uint8_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);                         // 16

    const auto dst_accessor = TensorAccessor(dst_ct_args, output_tensor_address, stick_size);

    // Nothing to send from an outward-facing edge worker (no neighbor this direction). The WRITER's
    // is_first/is_last args are NOT direction-swapped (unlike the reader's), so the send condition is
    // direction-dependent: forward sends iff !is_last_chip, backward sends iff !is_first_chip.
    const bool has_neighbor = direction ? !is_first_chip : !is_last_chip;

    // ---- Build + connect the mux endpoint ----
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
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
        tt::tt_fabric::fabric_client_connect(mux_connection);
    }

    // Stateful send headers (matches all_gather's mux usage): configure once with set_state, then update
    // only the dst addr + payload size per packet with with_state. num_hops = unicast distance to the
    // immediate W neighbor (dst_chip_id carries the hop count for the 1D line).
    const uint8_t num_hops = static_cast<uint8_t>(unicast_route_info.dst_chip_id);
    auto pkt_hdr = PacketHeaderPool::allocate_header();
    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_hdr, num_hops, nullptr, static_cast<uint16_t>(W_COALESCE * stick_size));
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();
    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_sem, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, outer_dim_size});

    // No separate W startup barrier: H->W ordering is provided entirely by the reader's barrier wait, and
    // the send buffers are fresh in a single standalone dispatch.

    // ---- Coalesced W send through the mux (bank-major; lockstep with np_phase2_w_reader) ----
    if (has_neighbor && mux_connection_valid) {
        for (uint32_t j = 0; j < NP_NUM_DRAM_BANKS; j++) {
            uint32_t r = j;
            while (r < outer_dim_size) {
                uint32_t g = 0;
                for (uint32_t rr = r; g < W_COALESCE && rr < outer_dim_size; rr += NP_NUM_DRAM_BANKS) {
                    g++;
                }
                cb_wait_front(send_cb_id, g);
                const uint32_t l1_read_addr = get_read_ptr(send_cb_id);
                const uint64_t dst_noc_addr = get_noc_addr(base + r, dst_accessor);
                // Tail groups on a bank are shorter than W_COALESCE, so update PayloadSize too.
                fabric_unicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    &mux_connection,
                    pkt_hdr,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr},
                    static_cast<uint16_t>(g * stick_size));
                // Flush the non-blocking mux write before releasing the group: the mux reads the payload
                // out of send_cb asynchronously, so popping first lets the reader overwrite it mid-read.
                noc_async_writes_flushed();
                cb_pop_front(send_cb_id, g);
                r += g * NP_NUM_DRAM_BANKS;
            }
        }
        // ---- Raise the neighbor's W recv sem (deferred single inc of the full row count) ----
        uint64_t nb_recv = safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, neighbor_sem, 0);
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            &mux_connection, pkt_hdr_sem, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{nb_recv, 0});
    }

    // ---- Disconnect + termination handshake (avoid mux-kernel hang) ----
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(mux_connection);
        if (is_termination_master) {
            auto* term_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
            noc_semaphore_wait(term_ptr, num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            uint64_t dest =
                safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
            noc_semaphore_inc(dest, 1);
            noc_async_atomic_barrier();
        }
    }
}
