// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
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
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

// Runtime-arg versions of route info readers, local to this kernel.
// Used because consolidated kernels have per-core routing via runtime args.
inline ccl_routing_utils::line_unicast_route_info_t get_line_unicast_route_info_from_rt_args(uint32_t& arg_idx) {
    return {
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++))};
}

inline ccl_routing_utils::line_multicast_route_info_t get_line_multicast_route_info_from_rt_args(uint32_t& arg_idx) {
    return {
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .e_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .w_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .n_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++)),
        .s_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++))};
}

// Compile-time args (uniform across all cores sharing this kernel)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Output TensorAccessorArgs start at index 3 (variable length)
constexpr auto dst_ct_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_dst = dst_ct_args.next_compile_time_args_offset();
constexpr bool use_l1_intermediate [[maybe_unused]] = get_compile_time_arg_val(ct_after_dst);
constexpr uint32_t recv_cb_id [[maybe_unused]] = get_compile_time_arg_val(ct_after_dst + 1);
constexpr bool handle_incoming_writes = get_compile_time_arg_val(ct_after_dst + 2);
constexpr bool is_w_fabric_writer = get_compile_time_arg_val(ct_after_dst + 3);
constexpr uint32_t ring_size = get_compile_time_arg_val(ct_after_dst + 4);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Common runtime args (uniform across all cores, updated between dispatches)
    const address_t input_tensor_address = get_common_arg_val<address_t>(0);
    const address_t output_tensor_address = get_common_arg_val<address_t>(1);
    const size_t neighbor_sem = get_common_arg_val<uint32_t>(2);
    const size_t barrier_sem = get_common_arg_val<uint32_t>(3);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_offset_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t neighbor_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t neighbor_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    // Phase 2 barrier signal targets (0 for 1D, >0 for 2D)
    // Max targets = pad2_num_links * 2 directions (up to 8 W fabric cores)
    constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 8;
    const uint32_t num_phase2_signal_targets = get_arg_val<uint32_t>(arg_idx++);
    uint8_t signal_noc_x[MAX_PHASE2_SIGNAL_TARGETS];
    uint8_t signal_noc_y[MAX_PHASE2_SIGNAL_TARGETS];
    for (uint32_t st = 0; st < MAX_PHASE2_SIGNAL_TARGETS; st++) {
        signal_noc_x[st] = get_arg_val<uint32_t>(arg_idx++);
        signal_noc_y[st] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Per-core direction and routing args (moved from compile-time for kernel consolidation)
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    auto unicast_route_info = get_line_unicast_route_info_from_rt_args(arg_idx);
    auto barrier_multicast_route_info = get_line_multicast_route_info_from_rt_args(arg_idx);

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    // W writer two-pass args: appended after fabric connection args (compile-time guarded).
    uint32_t w2_t_start = 0, w2_t_count = 0, w2_h_pad_top = 0, w2_h_in = 0, w2_h_pad_bot = 0, w2_h_out = 0;
    if constexpr (is_w_fabric_writer) {
        w2_t_start = get_arg_val<uint32_t>(arg_for_fab++);
        w2_t_count = get_arg_val<uint32_t>(arg_for_fab++);
        w2_h_pad_top = get_arg_val<uint32_t>(arg_for_fab++);
        w2_h_in = get_arg_val<uint32_t>(arg_for_fab++);
        w2_h_pad_bot = get_arg_val<uint32_t>(arg_for_fab++);
        w2_h_out = get_arg_val<uint32_t>(arg_for_fab++);
    }

    const auto dst_accessor = TensorAccessor(dst_ct_args, output_tensor_address);

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

    // pre-populate packet headers with proper routing
    auto pkt_hdr = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, unicast_route_info);
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

    // H writers: always open fabric at start (for data transfer in main loop).
    // W writers: open at start only when startup barrier is enabled (defer data transfer until CB ready).
    bool fabric_opened = false;
    if (!is_w_fabric_writer || use_barrier_sem) {
        fabric_connection.open();
        fabric_opened = true;
    }

    // Startup barrier: sync across all devices before sending new fabric data.
    // H writers: multicast to all H-axis devices (same column, consistent harvesting).
    // W writers: 1-hop unicast to immediate W neighbor only.
    //   W-axis devices span different UBBs with potentially different core harvesting,
    //   so multicast (which targets fixed NOC x,y) would hit wrong cores on remote devices.
    //   H all-to-all multicast + W 1-hop unicast transitively synchronizes the full mesh.
    if (use_barrier_sem) {
        if constexpr (!is_w_fabric_writer) {
            // H barrier: multicast to all H-axis devices (same column)
            auto pkt_hdr_barrier_sem_inc = PacketHeaderPool::allocate_header();

            if (!is_last_chip) {
                // Set up multicast routing and atomic inc state
                ccl_routing_utils::fabric_set_line_multicast_route(
                    pkt_hdr_barrier_sem_inc, barrier_multicast_route_info);
                fabric_multicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    pkt_hdr_barrier_sem_inc,
                    static_cast<uint8_t>(barrier_multicast_route_info.start_distance_in_hops),
                    static_cast<uint8_t>(barrier_multicast_route_info.range_hops),
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});

                // Multicast to same-direction cores on all reachable devices
                uint64_t same_dir_noc_addr =
                    safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, barrier_sem, 0);
                if (direction) {
                    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        &fabric_connection.get_backward_connection(),
                        pkt_hdr_barrier_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same_dir_noc_addr, 0});
                } else {
                    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        &fabric_connection.get_forward_connection(),
                        pkt_hdr_barrier_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same_dir_noc_addr, 0});
                }

                // Multicast to opposite-direction cores on all reachable devices
                uint64_t opp_dir_noc_addr = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
                if (direction) {
                    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        &fabric_connection.get_backward_connection(),
                        pkt_hdr_barrier_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opp_dir_noc_addr, 0});
                } else {
                    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        &fabric_connection.get_forward_connection(),
                        pkt_hdr_barrier_sem_inc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opp_dir_noc_addr, 0});
                }
            }

            if constexpr (ring_size > 1) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
            }
        } else {
            // W barrier: 1-hop unicast to immediate W neighbor only.
            // neighbor_sem_noc0_x/y and barrier_sem_noc0_x/y are NOC coords of the local
            // device's worker cores. Since NOC coords are not chip-dependent, these are
            // valid targets on the neighbor device as well.
            if (!is_last_chip) {
                auto pkt_hdr_barrier_sem_inc = PacketHeaderPool::allocate_header();

                // Unicast barrier inc to same-direction W core on immediate neighbor
                uint64_t same_dir_noc_addr =
                    safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, barrier_sem, 0);
                pkt_hdr_barrier_sem_inc->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{same_dir_noc_addr, 1u});
                ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_barrier_sem_inc, unicast_route_info);
                if (direction) {
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_barrier_sem_inc, sizeof(PACKET_HEADER_TYPE));
                } else {
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_barrier_sem_inc, sizeof(PACKET_HEADER_TYPE));
                }

                // Unicast barrier inc to opposite-direction W core on immediate neighbor
                uint64_t opp_dir_noc_addr = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
                pkt_hdr_barrier_sem_inc->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opp_dir_noc_addr, 1u});
                ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_barrier_sem_inc, unicast_route_info);
                if (direction) {
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_barrier_sem_inc, sizeof(PACKET_HEADER_TYPE));
                } else {
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_barrier_sem_inc, sizeof(PACKET_HEADER_TYPE));
                }
            }

            // Wait for 1 barrier inc from each adjacent W device (if it exists)
            uint32_t w_barrier_wait = (is_first_chip ? 0u : 1u) + (is_last_chip ? 0u : 1u);
            if (w_barrier_wait > 0) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), w_barrier_wait);
            }
        }
    }
    // Always zero barrier_sem, even when the startup barrier itself was skipped
    // (persistent output): Phase 2 (H->W) signaling reuses this same address, so a
    // stale non-zero value here would corrupt the Phase 2 handshake.
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);

    // Per-row processing body shared between H writer sequential loop and W writer two-pass loop.
    uint32_t outer_dim_offset = outer_dim_offset_start_id;

    auto process_one_row = [&]() {
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
                    cb_output.wait_front(1);
                    if constexpr (is_w_fabric_writer) {
                        if (!fabric_opened) {
                            fabric_connection.open();
                            fabric_opened = true;
                        }
                    }
                    uint32_t l1_read_addr = cb_output.get_read_ptr();

                    for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                        noc_obj.async_write(
                            CoreLocalMem<uint8_t>(l1_read_addr),
                            dst_accessor,
                            stick_size,
                            {},
                            {.page_id = dst_stick_id + pad_id * num_sticks_per_halo_dim});
                    }
                    dst_stick_id++;

                    noc_obj.async_write_barrier();
                    cb_output.pop_front(1);
                }
            } else {
                uint32_t dst_stick_id = 0;
                if (direction) {
                    dst_stick_id = (output_halo_dim_size - padding) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    dst_stick_id = stick_start_id;
                }
                dst_stick_id += outer_dim_offset;
                cb_output.wait_front(1);
                if constexpr (is_w_fabric_writer) {
                    if (!fabric_opened) {
                        fabric_connection.open();
                        fabric_opened = true;
                    }
                }
                uint32_t l1_read_addr = cb_output.get_read_ptr();
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                        noc_obj.async_write(
                            CoreLocalMem<uint8_t>(l1_read_addr),
                            dst_accessor,
                            stick_size,
                            {},
                            {.page_id = dst_stick_id + pad_id * num_sticks_per_halo_dim});
                    }
                    dst_stick_id++;

                    noc_obj.async_write_barrier();
                }
                cb_output.pop_front(1);
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
                    cb_output.wait_front(1);
                    if constexpr (is_w_fabric_writer) {
                        if (!fabric_opened) {
                            fabric_connection.open();
                            fabric_opened = true;
                        }
                    }
                    uint32_t l1_read_addr = cb_output.get_read_ptr();

                    // Always the neighbor's output DRAM (the persistent ping-pong buffer when
                    // allocated). Remote L1 CBs are not double-buffered: they only exist while
                    // this program is resident, so fabric-writing them after skipping the
                    // startup barrier can land in the neighbor's previous op (conv3d / GN).
                    uint64_t dst_noc_addr = dst_accessor.get_noc_addr(dst_stick_id, 0, 0);

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
                    noc_obj.async_writes_flushed();

                    dst_stick_id++;

                    noc_obj.async_write_barrier();
                    cb_output.pop_front(1);
                }
            }

            // unicast output ready semaphore
            uint64_t neighbor_sem_noc_addr_in_pkt =
                safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, neighbor_sem, 0);
            pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                neighbor_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
            // Write the unicast packet
            if (direction) {
                fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
                fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr_sem_inc, sizeof(PACKET_HEADER_TYPE));

            } else {
                fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
                fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr_sem_inc, sizeof(PACKET_HEADER_TYPE));
            }
            noc_obj.async_writes_flushed();
        }
    };  // end process_one_row

    if constexpr (!is_w_fabric_writer) {
        // H writer: sequential outer_dim loop (unchanged)
        for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
            process_one_row();
            // No local interior copy in this kernel. Dedicated local-copy kernels handle that work.
            outer_dim_offset += (num_sticks_per_halo_dim * output_halo_dim_size);
        }
    } else {
        // W writer two-pass loop.
        // Pass 1: interior rows (h_pad_top <= row < h_pad_top + h_in per T batch).
        // CB is fed by W reader Phase 1 which reads from INPUT DRAM — no barrier needed.
        for (uint32_t t = 0; t < w2_t_count; ++t) {
            for (uint32_t h = 0; h < w2_h_in; ++h) {
                // dst_row is the flat output row index for interior row h in T batch (w2_t_start + t)
                uint32_t dst_row = (w2_t_start + t) * w2_h_out + w2_h_pad_top + h;
                outer_dim_offset = dst_row * output_halo_dim_size;
                process_one_row();
            }
        }
        // Pass 2: H-pad rows (corners). CB is fed by W reader Phase 2 after H barrier.
        for (uint32_t t = 0; t < w2_t_count; ++t) {
            // Top H-pad rows
            for (uint32_t h = 0; h < w2_h_pad_top; ++h) {
                uint32_t dst_row = (w2_t_start + t) * w2_h_out + h;
                outer_dim_offset = dst_row * output_halo_dim_size;
                process_one_row();
            }
            // Bottom H-pad rows
            for (uint32_t h = 0; h < w2_h_pad_bot; ++h) {
                uint32_t dst_row = (w2_t_start + t) * w2_h_out + w2_h_pad_top + w2_h_in + h;
                outer_dim_offset = dst_row * output_halo_dim_size;
                process_one_row();
            }
        }
    }

    // Ensure all DRAM writes from main loop are complete.
    noc_obj.async_write_barrier();

    // Incoming fabric writes land in output DRAM (persistent POB when allocated).
    // The paired H reader waits on neighbor_sem then pushes one CB token so this
    // writer does not signal Phase 2 until those DRAM writes are visible.
    if constexpr (handle_incoming_writes) {
        if (!is_first_chip) {
            cb_output.wait_front(1);
            cb_output.pop_front(1);
        }
    }

    // Close fabric connection.
    if (fabric_opened) {
        noc_obj.async_write_barrier();
        fabric_connection.close();
    }

    // Signal Phase 2 AFTER fabric close and all work is complete.
    // Uses barrier_sem from CRTA[3] — same for all targets.
    noc_obj.async_write_barrier();
    for (uint32_t st = 0; st < num_phase2_signal_targets; st++) {
        uint64_t sem_noc_addr = get_noc_addr(signal_noc_x[st], signal_noc_y[st], barrier_sem);
        noc_semaphore_inc(sem_noc_addr, 1);
    }
    noc_obj.async_atomic_barrier();
}
