// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric-MUX H writer for neighbor_pad_halo. Mirrors np_w_mux_writer for the H exchange: N workers per
// (link,direction) feed the H-axis eth link through a mux core to reach link bandwidth, instead of the
// single-worker H send. Uses the same stateful linear API + mux lifecycle as all_gather.
//
// H send pattern (vs W): the paired np_h_reader gathers a full H-halo row (num_sticks_to_read = W_dev
// sticks) into send_cb in dst-bank-major order per row; this writer ships each bank's sticks as
// W_COALESCE-sized 4KB packets, once per padding row, per frame this worker owns. Straight-to-DRAM
// (use_l1=0); the neighbor's np_h_reader waits on h_neighbor_sem.

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
constexpr uint32_t H_COALESCE = get_compile_time_arg_val(ct_after_dst);
constexpr uint32_t NP_NUM_DRAM_BANKS = 8;
// Mux connection CT args (lockstep with FabricMuxConfig on the host).
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(ct_after_dst + 1);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(ct_after_dst + 2);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(ct_after_dst + 3);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(ct_after_dst + 4);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(ct_after_dst + 5);

void kernel_main() {
    // ---- Common runtime args ---- (index 2 = h_neighbor_sem, index 3 = barrier_sem; matches np_writer CRTA)
    const address_t output_tensor_address = get_common_arg_val<address_t>(1);
    const size_t neighbor_sem = get_common_arg_val<uint32_t>(2);
    const size_t barrier_sem = get_common_arg_val<uint32_t>(3);

    // ---- Per-core runtime args ----
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_start = get_arg_val<uint32_t>(arg_idx++);   // this worker's first frame
    const uint32_t outer_dim_count = get_arg_val<uint32_t>(arg_idx++);   // frames this worker owns
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);       // W_dev
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);  // W_dev (row stride)
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);      // H_dev (for eff_offset)
    const uint8_t neighbor_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t neighbor_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    auto unicast_route_info = ccl_routing_utils::line_unicast_route_info_t{
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++))};

    // Mux connection RT args — EXACT layout of ccl::fabric_mux_connection_rt_args (positions 0..16).
    const bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // H->W barrier targets: after H completes, inc barrier_sem (intra-device NOC) on each W-reader core so
    // the W reader clears its H->W barrier wait. barrier_count on the W side = total H mux workers.
    constexpr uint32_t MAX_W_BARRIER_TARGETS = 16;
    const uint32_t num_w_barrier_targets = get_arg_val<uint32_t>(arg_idx++);
    uint8_t w_bar_x[MAX_W_BARRIER_TARGETS];
    uint8_t w_bar_y[MAX_W_BARRIER_TARGETS];
    for (uint32_t t = 0; t < MAX_W_BARRIER_TARGETS; t++) {
        w_bar_x[t] = get_arg_val<uint32_t>(arg_idx++);
        w_bar_y[t] = get_arg_val<uint32_t>(arg_idx++);
    }

    const auto dst_accessor = TensorAccessor(dst_ct_args, output_tensor_address, stick_size);
    const bool has_neighbor = direction ? !is_first_chip : !is_last_chip;

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection;
    if (mux_connection_valid) {
        mux_connection = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
            fabric_mux_x, fabric_mux_y, fabric_mux_channel_id, fabric_mux_num_buffers_per_channel,
            fabric_mux_channel_buffer_size_bytes, fabric_mux_channel_base_address,
            fabric_mux_connection_info_address, fabric_mux_connection_handshake_address,
            fabric_mux_flow_control_address, fabric_mux_buffer_index_address, local_flow_control_address,
            local_teardown_address, local_buffer_index_address);
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
        tt::tt_fabric::fabric_client_connect(mux_connection);
    }

    const uint8_t num_hops = static_cast<uint8_t>(unicast_route_info.dst_chip_id);
    auto pkt_hdr = PacketHeaderPool::allocate_header();
    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_hdr, num_hops, nullptr, static_cast<uint16_t>(H_COALESCE * stick_size));
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();
    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_sem, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, outer_dim_count});

    if (has_neighbor && mux_connection_valid) {
        const uint32_t row_stride = num_sticks_per_halo_dim * output_halo_dim_size;
        for (uint32_t od = 0; od < outer_dim_count; od++) {
            const uint32_t eff_offset = (outer_dim_start + od) * row_stride;
            for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                uint32_t base_row = direction
                                        ? (output_halo_dim_size - (padding - pad_id)) * num_sticks_per_halo_dim +
                                              stick_start_id
                                        : pad_id * num_sticks_per_halo_dim + stick_start_id;
                base_row += eff_offset;
                cb_wait_front(send_cb_id, num_sticks_to_read);
                const uint32_t row_l1 = get_read_ptr(send_cb_id);
                uint32_t m = 0;
                for (uint32_t j = 0; j < NP_NUM_DRAM_BANKS; j++) {
                    for (uint32_t w = j; w < num_sticks_to_read;) {
                        uint32_t g = 0;
                        for (uint32_t ww = w; g < H_COALESCE && ww < num_sticks_to_read; ww += NP_NUM_DRAM_BANKS) {
                            g++;
                        }
                        fabric_unicast_noc_unicast_write_with_state<
                            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                            &mux_connection,
                            pkt_hdr,
                            row_l1 + m * stick_size,
                            tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(base_row + w, dst_accessor)},
                            static_cast<uint16_t>(g * stick_size));
                        m += g;
                        w += g * NP_NUM_DRAM_BANKS;
                    }
                }
                cb_pop_front(send_cb_id, num_sticks_to_read);
            }
        }
        uint64_t nb_recv = safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, neighbor_sem, 0);
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            &mux_connection, pkt_hdr_sem, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{nb_recv, 0});
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    // H->W barrier: every H worker (incl. no-neighbor edges) increments barrier_sem on each W-reader core
    // so the W reader clears its H->W wait after all H workers finish. Intra-device NOC inc (barrier_sem
    // is per-core L1 at a fixed address). barrier_count on the W side = total H mux workers.
    for (uint32_t t = 0; t < num_w_barrier_targets; t++) {
        noc_semaphore_inc(safe_get_noc_addr(w_bar_x[t], w_bar_y[t], barrier_sem, 0), 1);
    }
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
