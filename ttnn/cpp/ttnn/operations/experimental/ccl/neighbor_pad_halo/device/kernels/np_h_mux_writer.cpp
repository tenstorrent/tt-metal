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
constexpr bool is_padding_zeros = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);  // c_in0: reader's is_first local-pad output
constexpr uint32_t send_cb_id = get_compile_time_arg_val(2);    // hsend: coalesced fabric-send ring
constexpr uint32_t stick_size = get_compile_time_arg_val(3);
constexpr auto dst_ct_args = TensorAccessorArgs<4>();
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
    // Compact-buffer H-section base for this worker's first frame (h_top or h_bot section base +
    // outer_dim_start*padding*num_sticks_per_halo_dim). The compact buffer stores H-top and H-bot as
    // SEPARATE sections (stride padding, not output_halo_dim_size) — matches np_writer's H-section override.
    const uint32_t h_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_count = get_arg_val<uint32_t>(arg_idx++);  // frames this worker owns
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);       // W_dev
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);  // W_dev (row stride)
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

    const auto dst_accessor = TensorAccessor(dst_ct_args, output_tensor_address, stick_size);
    // is_first_chip/is_last_chip are direction-adjusted by the factory (match np_h_reader + np_writer):
    // a worker sends iff !is_last_chip; it fills its own outward padding locally iff is_first_chip.
    const bool has_neighbor = !is_last_chip;

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

    const uint8_t num_hops = static_cast<uint8_t>(unicast_route_info.dst_chip_id);
    auto pkt_hdr = PacketHeaderPool::allocate_header();
    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_hdr, num_hops, nullptr, static_cast<uint16_t>(H_COALESCE * stick_size));
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();
    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_sem, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, outer_dim_count});

    // No startup barrier here: np_h_reader signals the H->W barrier after its recv drains
    // (H_SIGNAL_W_RECV), so W corners already wait on real H-recv — a send-side start sync would add nothing.

    // Compact H-section layout: rows are [frame][pad_id][W], stride padding rows per frame. h_base already
    // includes this worker's outer_dim_start offset + the h_top/h_bot section base. Per frame the reader
    // produces (in order): the is_first local-pad row into c_in0 (edge devices only), then the coalesced
    // send rows into hsend. Drain in the same order.
    for (uint32_t od = 0; od < outer_dim_count; od++) {
        if (is_first_chip) {
            // Local outward padding for this device's edge H-halo (no fabric): write c_in0 to the compact
            // H-section. Replicate: reader pushed one stick per W col -> broadcast each to all padding rows.
            // Zeros: reader pushed one stick -> broadcast to all padding rows x W cols.
            const uint32_t frame_base = h_base + od * padding * num_sticks_per_halo_dim;
            if (!is_padding_zeros) {
                for (uint32_t col = 0; col < num_sticks_to_read; col++) {
                    cb_wait_front(cb_output_id, 1);
                    const uint32_t l1 = get_read_ptr(cb_output_id);
                    for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                        noc_async_write(
                            l1,
                            get_noc_addr(frame_base + pad_id * num_sticks_per_halo_dim + col, dst_accessor),
                            stick_size);
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_output_id, 1);
                }
            } else {
                cb_wait_front(cb_output_id, 1);
                const uint32_t l1 = get_read_ptr(cb_output_id);
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    for (uint32_t col = 0; col < num_sticks_to_read; col++) {
                        noc_async_write(
                            l1,
                            get_noc_addr(frame_base + pad_id * num_sticks_per_halo_dim + col, dst_accessor),
                            stick_size);
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_id, 1);
            }
        }
        if (has_neighbor && mux_connection_valid) {
            for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                const uint32_t base_row = h_base + (od * padding + pad_id) * num_sticks_per_halo_dim;
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
                        // The payload copy (send_cb -> mux slot) is non-blocking and pkt_hdr is reused for
                        // the next packet; flush per packet so neither the source row nor the header is
                        // touched while the mux is still reading it (a dropped-packet race, worse at large
                        // page where the copy takes longer). Measured free: the writes have drained by the
                        // next loop iteration, so the poll returns immediately.
                        noc_async_writes_flushed();
                        m += g;
                        w += g * NP_NUM_DRAM_BANKS;
                    }
                }
                cb_pop_front(send_cb_id, num_sticks_to_read);
            }
        }
    }
    if (has_neighbor && mux_connection_valid) {
        // Single deferred recv-sem inc of the full frame count (set_state configured Val=outer_dim_count).
        uint64_t nb_recv = safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, neighbor_sem, 0);
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            &mux_connection, pkt_hdr_sem, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{nb_recv, 0});
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    // H->W barrier is signaled by np_h_reader after recv (see its H_SIGNAL_W_RECV block), not here.

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
