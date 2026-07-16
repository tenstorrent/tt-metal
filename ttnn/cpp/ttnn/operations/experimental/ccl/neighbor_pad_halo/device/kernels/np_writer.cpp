// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/np_reorder.hpp"
#include <cstdint>
#include <utility>

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
constexpr bool use_l1_intermediate = get_compile_time_arg_val(ct_after_dst);
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(ct_after_dst + 1);
constexpr bool handle_incoming_writes = get_compile_time_arg_val(ct_after_dst + 2);
constexpr bool is_w_fabric_writer = get_compile_time_arg_val(ct_after_dst + 3);
constexpr uint32_t ring_size = get_compile_time_arg_val(ct_after_dst + 4);
// CB the fabric-send loop drains. H writer: dedicated batched send ring (c_in2). W writer: c_in0.
constexpr uint32_t send_cb_id = get_compile_time_arg_val(ct_after_dst + 5);

// H-writer corner-first send (H-writer path only): raise the recv sem after the corner sticks, before
// the bulk middle, so the neighbor's recv-wait clears after ~pad2 sticks instead of the whole row. Set
// per-shape by the program factory; unused on the W writer.
constexpr bool H_CORNER_FIRST = get_compile_time_arg_val(ct_after_dst + 6);

// W-send bank-major coalesce factor (0 = per-stick). Lockstep with np_phase2_w_reader: a middle W
// device ships N same-dst-bank sticks (base+r, base+r+8, ...) as one N*page fabric write to the
// neighbor's interleaved DRAM. BH: 8 DRAM banks.
constexpr uint32_t W_COALESCE = get_compile_time_arg_val(ct_after_dst + 7);
constexpr uint32_t NP_NUM_DRAM_BANKS = 8;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Common runtime args (uniform across all cores, updated between dispatches). Index 0 (input addr)
    // is part of the shared CRTA layout but unused by the writer (it reads/writes the output buffer).
    const address_t output_tensor_address = get_common_arg_val<address_t>(1);
    const size_t neighbor_sem = get_common_arg_val<uint32_t>(2);
    const size_t barrier_sem = get_common_arg_val<uint32_t>(3);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_offset_start_id = get_arg_val<uint32_t>(arg_idx++);  // this core's first dst stick
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);             // W offset where interior begins
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);        // interior H per device (H_dev)
    const uint32_t output_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);       // halo rows per frame (top+bot)
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);             // frames/rows this core owns
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);                    // halo rows per side this od
    const uint32_t padding_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);  // interior W width to read per row
    // Dst row stride in sticks. Overloaded by path: full output row width (W+pad2_left+pad2_right) on the
    // corners-only H path, but interior W_dev on the compact-buffer paths (see the corners derivation below).
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t neighbor_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t neighbor_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    // Phase 2 barrier signal targets (0 for 1D, >0 for 2D). Under W-mux each (link,dir) fans out to
    // num_w_workers reader cores, so the max is pad2_num_links * 2 * num_w_workers (MAX_PAD2_NUM_LINKS
    // 4 * 2 * 4 workers = 32). Must match MAX_PHASE2_SIGNAL_TARGETS in the program factory.
    constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 32;
    const uint32_t num_phase2_signal_targets = get_arg_val<uint32_t>(arg_idx++);
    uint8_t signal_noc_x[MAX_PHASE2_SIGNAL_TARGETS];
    uint8_t signal_noc_y[MAX_PHASE2_SIGNAL_TARGETS];
    for (uint32_t st = 0; st < MAX_PHASE2_SIGNAL_TARGETS; st++) {
        signal_noc_x[st] = get_arg_val<uint32_t>(arg_idx++);
        signal_noc_y[st] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Per-core direction and routing args passed at runtime (not compile-time) so one kernel binary serves every core
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    auto unicast_route_info = get_line_unicast_route_info_from_rt_args(arg_idx);
    auto barrier_multicast_route_info = get_line_multicast_route_info_from_rt_args(arg_idx);

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    const auto dst_accessor = TensorAccessor(dst_ct_args, output_tensor_address, stick_size);

    // L1 intermediate: discover the recv CB base address (same on neighbor device due to identical program)
    uint32_t recv_buf_base = 0;
    if constexpr (use_l1_intermediate) {
        recv_buf_base = get_write_ptr(recv_cb_id);
    }

    // pre-populate packet headers with proper routing
    auto pkt_hdr = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, unicast_route_info);
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

    // Send one payload stick to dst_noc_addr over the fabric; pkt_hdr is reused per stick.
    auto fabric_send_stick = [&](uint32_t l1_read_addr, uint64_t dst_noc_addr) {
        pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, stick_size);
        auto& conn =
            direction ? fabric_connection.get_backward_connection() : fabric_connection.get_forward_connection();
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, stick_size);
        conn.send_payload_flush_non_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
        noc_async_writes_flushed();
    };

    // Atomic-inc the neighbor's output-ready semaphore over the fabric.
    auto raise_neighbor_sem = [&]() {
        uint64_t sem_noc_addr = safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, neighbor_sem, 0);
        pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, static_cast<uint32_t>(1)});
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
        auto& conn =
            direction ? fabric_connection.get_backward_connection() : fabric_connection.get_forward_connection();
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr_sem_inc, sizeof(PACKET_HEADER_TYPE));
        noc_async_writes_flushed();
    };

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
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    // Corners-only optimization for 2D H writers:
    // Only W-boundary sticks (corners) go to neighbor L1; non-corner sticks go directly to DRAM.
    // Phase 2 W reader only needs corners, so this is safe.
    // Derivation: the output row is [pad2_left | W interior sticks | pad2_right].
    // The factory sets stick_start_id = pad2_left (the W offset where interior data begins),
    // num_sticks_per_halo_dim = W + pad2_left + pad2_right (full output row width),
    // and num_sticks_to_read = W (interior width). So:
    //   pad2_left_sticks  = stick_start_id = pad2_left
    //   pad2_right_sticks = (W + pad2_left + pad2_right) - W - pad2_left = pad2_right
    // These can be different (asymmetric W padding is supported).
    uint32_t pad2_left_sticks = 0;
    uint32_t pad2_right_sticks = 0;
    if constexpr (use_l1_intermediate && !is_w_fabric_writer) {
        // Use padding_left for the corner count (not stick_start_id) so it still works when
        // stick_start_id=0 (the compact-buffer case): stick_start_id=0 but padding_left can be W/2,
        // so pad2_left_sticks == pad2_right_sticks → overlap case → all sticks are corners.
        pad2_left_sticks = (padding_left > 0) ? padding_left : stick_start_id;
        uint32_t w_overhead = num_sticks_per_halo_dim - num_sticks_to_read;
        pad2_right_sticks = (w_overhead >= pad2_left_sticks) ? (w_overhead - pad2_left_sticks) : pad2_left_sticks;
    }
    // A W-boundary (corner) stick is one of the pad2 columns at either row end; the rest are interior.
    auto is_corner_stick = [&](uint32_t iter) {
        return iter < pad2_right_sticks || iter >= (num_sticks_to_read - pad2_left_sticks);
    };

    // W-path interior-first reorder dims (common args [4],[5], W-writer only; the factory always sets them).
    uint32_t w_input_H = 0, w_pad_h = 0, w_h_total = 1, w_row_stride = 0, w_slice_frames = 0;
    bool w_interior_first_p0 = false;
    if constexpr (is_w_fabric_writer) {
        w_input_H = get_common_arg_val<uint32_t>(4);
        w_pad_h = get_common_arg_val<uint32_t>(5);
        w_h_total = w_input_H + 2 * w_pad_h;
        w_row_stride = num_sticks_per_halo_dim * output_halo_dim_size;
        w_slice_frames = (w_h_total > 0) ? (outer_dim_size / w_h_total) : 0;
        // Lockstep with np_phase2_w_reader: reorder interior-first when frames align to the slice.
        w_interior_first_p0 = (outer_dim_size == w_slice_frames * w_h_total);
    }

    // Coalesced W-send (middle device): ship g (<= W_COALESCE) same-dst-bank sticks as one g*page fabric
    // write, lockstep with np_phase2_w_reader's bank-major gather. base = outer_dim_offset_start_id =
    // section_base + w_link_start (compact W base, 8-aligned by factory eligibility), so dst sticks
    // base+r, base+r+8, ..., base+r+8*(g-1) are contiguous on bank (base+r)%8 in interleaved DRAM.
    if constexpr (is_w_fabric_writer && W_COALESCE > 0) {
        if (!is_first_chip && !is_last_chip) {
            if (!fabric_opened) {
                fabric_connection.open();
                fabric_opened = true;
            }
            const uint32_t base = outer_dim_offset_start_id;
            auto& conn =
                direction ? fabric_connection.get_backward_connection() : fabric_connection.get_forward_connection();
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
                    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, g * stick_size);
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, g * stick_size);
                    conn.send_payload_flush_non_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                    noc_async_writes_flushed();
                    cb_pop_front(send_cb_id, g);
                    r += g * NP_NUM_DRAM_BANKS;
                }
            }
            // Raise the neighbor's W recv sem by the full row count (receiver waits >= outer_dim_size).
            {
                uint64_t sem_noc_addr = safe_get_noc_addr(neighbor_sem_noc0_x, neighbor_sem_noc0_y, neighbor_sem, 0);
                pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, outer_dim_size});
                ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr_sem_inc, sizeof(PACKET_HEADER_TYPE));
                noc_async_writes_flushed();
            }
            noc_async_write_barrier();
            fabric_connection.close();
            return;
        }
    }

    uint32_t outer_dim_offset = outer_dim_offset_start_id;
    uint32_t l1_buf_offset = 0;                       // L1 intermediate: accumulates across all outer_dims (no reuse)
    uint32_t inc_offset = outer_dim_offset_start_id;  // recv-commit cursor, advances per committed od
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        // Interior-first reorder: ALL interior rows then ALL corners (lockstep with the W reader).
        uint32_t eff_offset = outer_dim_offset;
        bool w_use_reorder = false;
        if constexpr (is_w_fabric_writer) {
            w_use_reorder = w_interior_first_p0;
        }
        if (w_use_reorder) {
            uint32_t frame_in_slice, hp;
            np_reorder_batch(outer_dim, w_slice_frames, w_input_H, w_pad_h, frame_in_slice, hp);
            const uint32_t reordered = frame_in_slice * w_h_total + hp;
            eff_offset = outer_dim_offset_start_id + reordered * w_row_stride;
        }
        if (is_first_chip) {
            if (!is_padding_zeros) {
                // Replicate a slice of 1 from input to output
                uint32_t dst_stick_id = 0;
                if (direction) {
                    dst_stick_id = (output_halo_dim_size - padding) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    dst_stick_id = stick_start_id;
                }
                dst_stick_id += eff_offset;
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    cb_wait_front(cb_output_id, 1);
                    if constexpr (is_w_fabric_writer) {
                        if (!fabric_opened) {
                            fabric_connection.open();
                            fabric_opened = true;
                        }
                    }
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
                dst_stick_id += eff_offset;
                cb_wait_front(cb_output_id, 1);
                if constexpr (is_w_fabric_writer) {
                    if (!fabric_opened) {
                        fabric_connection.open();
                        fabric_opened = true;
                    }
                }
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
            bool did_corner_first = false;
            // H writer, corners-only, padding==1: the neighbor's H reader recv-commit pulls only the
            // corner sticks; the bulk middle sticks go straight to its halo-buffer DRAM, read later by
            // its W-reader. So send the few corners first and raise the recv sem BEFORE the
            // middle bulk — the neighbor clears its recv-wait after ~pad2 sticks instead of the whole
            // row, taking the full-row send off the H-recv critical path. The single send_cb row holds
            // both passes, so no extra read. padding!=1 needs all pad rows resident before the sem inc
            // (the 2-row send_cb can't guarantee that), so it keeps the in-order path below.
            if constexpr (use_l1_intermediate && !is_w_fabric_writer) {
                if (H_CORNER_FIRST && padding == 1) {
                    uint32_t dst_stick_id =
                        (direction ? (output_halo_dim_size - 1) * num_sticks_per_halo_dim + stick_start_id
                                   : stick_start_id) +
                        eff_offset;
                    cb_wait_front(send_cb_id, num_sticks_to_read);
                    const uint32_t row_base = get_read_ptr(send_cb_id);
                    // Pass 1: corner sticks -> neighbor L1.
                    for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                        if (!is_corner_stick(iter)) {
                            continue;
                        }
                        uint64_t dst_noc_addr = safe_get_noc_addr(
                            neighbor_sem_noc0_x, neighbor_sem_noc0_y, recv_buf_base + l1_buf_offset, 0);
                        l1_buf_offset += stick_size;
                        fabric_send_stick(row_base + iter * stick_size, dst_noc_addr);
                    }
                    noc_async_write_barrier();

                    // Corners delivered: raise the recv sem before the middle bulk so the neighbor's
                    // recv-commit clears now instead of after the whole row.
                    raise_neighbor_sem();

                    // Pass 2: bulk middle sticks -> neighbor DRAM (off the H-recv critical path).
                    for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                        if (!is_corner_stick(iter)) {
                            fabric_send_stick(
                                row_base + iter * stick_size, get_noc_addr(dst_stick_id, dst_accessor, 0, 0));
                        }
                        dst_stick_id++;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(send_cb_id, num_sticks_to_read);
                    did_corner_first = true;
                }
            }
            if (!did_corner_first) {
                // Read the "end" of each slice into the CB to write to the neighbor
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    uint32_t dst_stick_id = 0;
                    if (direction) {
                        dst_stick_id =
                            (output_halo_dim_size - (padding - pad_id)) * num_sticks_per_halo_dim + stick_start_id;
                    } else {
                        dst_stick_id = pad_id * num_sticks_per_halo_dim + stick_start_id;
                    }
                    dst_stick_id += eff_offset;
                    // H-writer bank-major coalesced send (use_l1=0). np_h_reader gathered the row into
                    // send_cb in bank-major order (w=0,8,..; 1,9,..); ship each bank's sticks as
                    // W_COALESCE-sized packets to base_row+w (contiguous on bank (base_row+w)%8, 8-aligned
                    // base_row). W_COALESCE here holds h_coalesce_n (the H writer's factory arg slot).
                    if constexpr (!is_w_fabric_writer && W_COALESCE > 0) {
                        const uint32_t base_row = dst_stick_id;
                        cb_wait_front(send_cb_id, num_sticks_to_read);
                        const uint32_t row_l1 = get_read_ptr(send_cb_id);
                        auto& conn = direction ? fabric_connection.get_backward_connection()
                                               : fabric_connection.get_forward_connection();
                        uint32_t m = 0;
                        for (uint32_t j = 0; j < NP_NUM_DRAM_BANKS; j++) {
                            for (uint32_t w = j; w < num_sticks_to_read;) {
                                uint32_t g = 0;
                                for (uint32_t ww = w; g < W_COALESCE && ww < num_sticks_to_read;
                                     ww += NP_NUM_DRAM_BANKS) {
                                    g++;
                                }
                                const uint64_t dst_noc = get_noc_addr(base_row + w, dst_accessor);
                                pkt_hdr->to_noc_unicast_write(
                                    tt::tt_fabric::NocUnicastCommandHeader{dst_noc}, g * stick_size);
                                conn.wait_for_empty_write_slot();
                                conn.send_payload_without_header_non_blocking_from_address(
                                    row_l1 + m * stick_size, g * stick_size);
                                conn.send_payload_flush_non_blocking_from_address(
                                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                                noc_async_writes_flushed();
                                m += g;
                                w += g * NP_NUM_DRAM_BANKS;
                            }
                        }
                        cb_pop_front(send_cb_id, num_sticks_to_read);
                        continue;  // next pad_id; skip the per-stick loop
                    }
                    for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                        cb_wait_front(send_cb_id, 1);
                        if constexpr (is_w_fabric_writer) {
                            if (!fabric_opened) {
                                fabric_connection.open();
                                fabric_opened = true;
                            }
                        }
                        uint32_t l1_read_addr = get_read_ptr(send_cb_id);

                        uint64_t dst_noc_addr;
                        if constexpr (use_l1_intermediate && !is_w_fabric_writer) {
                            // Corners-only: W-boundary sticks go to L1, rest to DRAM
                            if (is_corner_stick(iter)) {
                                dst_noc_addr = safe_get_noc_addr(
                                    neighbor_sem_noc0_x, neighbor_sem_noc0_y, recv_buf_base + l1_buf_offset, 0);
                                l1_buf_offset += stick_size;
                            } else {
                                // Non-corner: send directly to neighbor's output DRAM
                                dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor, 0, 0);
                            }
                        } else if constexpr (use_l1_intermediate) {
                            // W writer: all sticks to L1
                            dst_noc_addr = safe_get_noc_addr(
                                neighbor_sem_noc0_x, neighbor_sem_noc0_y, recv_buf_base + l1_buf_offset, 0);
                            l1_buf_offset += stick_size;
                        } else {
                            dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor, 0, 0);
                        }

                        fabric_send_stick(l1_read_addr, dst_noc_addr);

                        dst_stick_id++;

                        // fabric_send_stick already flushed the local writes (source read done), so the
                        // CB slot is safe to reuse. The receiver only reads after the post-loop
                        // raise_neighbor_sem, so no per-stick remote-completion barrier is needed here.
                        cb_pop_front(send_cb_id, 1);
                    }
                }

                raise_neighbor_sem();
            }
        }

        // This kernel writes only the halo exchange, never the interior. In padded-output mode the fused
        // scatter writer copies the interior; in compact mode there is no interior copy (output is the
        // halo buffer).

        outer_dim_offset += (num_sticks_per_halo_dim * output_halo_dim_size);

        // Per-batch H-commit: commit this od's incoming H-halo (pushed by the paired reader from the
        // L1 recv buffer as soon as the sender link delivered it), then signal HT/HB per batch so a
        // per-batch consumer could ramp this batch rather than after the whole send pass (inert here,
        // progress==0). One producer per (region,link) -> monotonic.
        if constexpr (handle_incoming_writes) {
            if (!is_first_chip) {
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    uint32_t row_offset;
                    if (direction) {
                        row_offset = (output_halo_dim_size - padding + pad_id) * num_sticks_per_halo_dim;
                    } else {
                        row_offset = pad_id * num_sticks_per_halo_dim;
                    }
                    uint32_t base_dst = inc_offset + row_offset + stick_start_id;

                    if constexpr (use_l1_intermediate && !is_w_fabric_writer) {
                        if (pad2_right_sticks + pad2_left_sticks >= num_sticks_to_read) {
                            // Overlap: all sticks are corners, pop exactly num_sticks_to_read
                            for (uint32_t c = 0; c < num_sticks_to_read; c++) {
                                cb_wait_front(cb_output_id, 1);
                                uint32_t l1_read_addr = get_read_ptr(cb_output_id);
                                uint64_t dst_noc_addr = get_noc_addr(base_dst + c, dst_accessor);
                                noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
                                noc_async_write_barrier();
                                cb_pop_front(cb_output_id, 1);
                            }
                        } else {
                            // Corners-only: pop left corners then right corners from CB
                            for (uint32_t c = 0; c < pad2_right_sticks; c++) {
                                cb_wait_front(cb_output_id, 1);
                                uint32_t l1_read_addr = get_read_ptr(cb_output_id);
                                uint64_t dst_noc_addr = get_noc_addr(base_dst + c, dst_accessor);
                                noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
                                noc_async_write_barrier();
                                cb_pop_front(cb_output_id, 1);
                            }
                            uint32_t right_start = base_dst + (num_sticks_to_read - pad2_left_sticks);
                            for (uint32_t c = 0; c < pad2_left_sticks; c++) {
                                cb_wait_front(cb_output_id, 1);
                                uint32_t l1_read_addr = get_read_ptr(cb_output_id);
                                uint64_t dst_noc_addr = get_noc_addr(right_start + c, dst_accessor);
                                noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
                                noc_async_write_barrier();
                                cb_pop_front(cb_output_id, 1);
                            }
                        }
                    } else {
                        uint32_t dst_stick_id = base_dst;
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
                }
                inc_offset += num_sticks_per_halo_dim * output_halo_dim_size;
            }
        }
    }

    // Ensure all DRAM writes from main loop are complete.
    noc_async_write_barrier();

    // Close fabric connection.
    if (fabric_opened) {
        noc_async_write_barrier();
        fabric_connection.close();
    }

    // Signal Phase 2 W-cores AFTER handle_incoming_writes and fabric close.
    // This guarantees L1-routed corners are committed to DRAM before W-readers start.
    // In 1D mode num_phase2_signal_targets == 0, so this is a no-op.
    noc_async_write_barrier();
    for (uint32_t st = 0; st < num_phase2_signal_targets; st++) {
        uint64_t sem_noc_addr = get_noc_addr(signal_noc_x[st], signal_noc_y[st], barrier_sem);
        noc_semaphore_inc(sem_noc_addr, 1);
    }
    noc_async_atomic_barrier();
}
