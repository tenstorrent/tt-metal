// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

///////////////////////////////////////////////////////////////////////////////
// Tree Reduce-Scatter Writer Kernel
//
// This kernel handles two responsibilities:
// 1. SEND: Read data and send to destination devices via fabric
// 2. STORE: Write reduced results from compute to intermediate or output buffer
//
// The tree algorithm has ceil(log2(ring_size)) steps. At each step S:
// - Communication stride = 2^S (devices communicate with partners stride hops away)
// - Some slices are sent to destination devices (sender role)
// - Some slices have reduced results to store (receiver role)
//
// Multi-worker support:
// - Each worker has a direction (forward=1, backward=0)
// - Forward workers handle slices sent forward and received from forward
// - Backward workers handle slices sent backward and received from backward
// - Multiple workers within same direction split tiles among themselves
//
// REQUIREMENTS:
// - Host must configure forward/backward route_info.distance_in_hops >= ring_size/2
//   to support multi-hop sends at later tree steps
// - Symmetric allocation: all devices must use the same L1 addresses for semaphores
//   and the same worker core coordinates (typical for collective operations)
///////////////////////////////////////////////////////////////////////////////

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include <cstdint>

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t num_steps = get_compile_time_arg_val(2);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);  // reduced data from compute
constexpr uint32_t cb_send_id = get_compile_time_arg_val(4);    // staging CB for send data
constexpr uint32_t tile_granularity = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);

// Tensor shape info
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(7);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(8);
constexpr uint32_t slice_C = get_compile_time_arg_val(9);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(10);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(11);
constexpr uint32_t dim = get_compile_time_arg_val(12);

// Worker direction: 0=backward, 1=forward
constexpr uint32_t direction = get_compile_time_arg_val(13);

#ifdef USE_WORKER_MUX
// Fabric mux compile-time args
constexpr uint32_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(14);
constexpr uint32_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(15);
constexpr uint32_t fabric_mux_status_address = get_compile_time_arg_val(16);
constexpr uint32_t fabric_mux_termination_signal_address = get_compile_time_arg_val(17);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(18);
constexpr uint32_t routing_args_start = 19;
#else
constexpr uint32_t routing_args_start = 14;
#endif

// Routing info (compile-time args for fabric) - only need one direction per worker
constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<routing_args_start>();

// TensorAccessor args come after routing args
constexpr uint32_t tensor_args_start = routing_args_start + ccl_routing_utils::num_line_unicast_args;

///////////////////////////////////////////////////
// TREE ALGORITHM HELPER FUNCTIONS
///////////////////////////////////////////////////

FORCE_INLINE int32_t compute_offset(uint32_t device_id, uint32_t slice_id, uint32_t ring_sz) {
    int32_t raw_offset = static_cast<int32_t>(device_id) - static_cast<int32_t>(slice_id);
    if (raw_offset > static_cast<int32_t>(ring_sz / 2)) {
        raw_offset -= ring_sz;
    } else if (raw_offset < -static_cast<int32_t>((ring_sz - 1) / 2)) {
        raw_offset += ring_sz;
    }
    return raw_offset;
}

FORCE_INLINE int32_t get_left_root(uint32_t ring_sz) { return (ring_sz % 2 == 0) ? 0 : -1; }

// Check if offset is a LEFT side sender at given step
FORCE_INLINE bool is_left_sender(int32_t offset, uint32_t step, uint32_t ring_sz) {
    int32_t left_root = get_left_root(ring_sz);
    if (offset >= left_root) {
        return false;
    }

    int32_t adj = offset - left_root;
    uint32_t stride = 1u << step;
    uint32_t group = stride << 1;
    uint32_t abs_adj = static_cast<uint32_t>(-adj);

    return (abs_adj % group == stride);
}

// Check if offset is a RIGHT side sender at given step
FORCE_INLINE bool is_right_sender(int32_t offset, uint32_t step) {
    if (offset <= 1) {
        return false;
    }

    uint32_t adj = static_cast<uint32_t>(offset - 1);
    uint32_t stride = 1u << step;
    uint32_t group = stride << 1;

    return (adj % group == stride);
}

// Check if offset is a LEFT side receiver at given step
FORCE_INLINE bool is_left_receiver(int32_t offset, uint32_t step, uint32_t ring_sz) {
    int32_t left_root = get_left_root(ring_sz);
    if (offset > left_root) {
        return false;
    }

    int32_t adj = offset - left_root;
    uint32_t stride = 1u << step;
    uint32_t group = stride << 1;
    uint32_t abs_adj = static_cast<uint32_t>(-adj);

    return (abs_adj % group == 0);
}

// Check if offset is a RIGHT side receiver at given step
FORCE_INLINE bool is_right_receiver(int32_t offset, uint32_t step) {
    if (offset < 1) {
        return false;
    }

    uint32_t adj = static_cast<uint32_t>(offset - 1);
    uint32_t stride = 1u << step;
    uint32_t group = stride << 1;

    return (adj % group == 0);
}

// Get send direction: 0=none, 1=backward (to device-1), 2=forward (to device+1)
FORCE_INLINE uint32_t get_send_direction(int32_t offset, uint32_t step, uint32_t num_steps_total, uint32_t ring_sz) {
    bool is_final = (step == num_steps_total - 1);

    if (is_final) {
        if (offset == 1) {
            return 1;  // backward
        }
        return 0;
    }

    if (is_left_sender(offset, step, ring_sz)) {
        return 2;  // forward
    }

    if (is_right_sender(offset, step)) {
        return 1;  // backward
    }

    return 0;
}

// Get receive direction: 0=none, 1=backward (from device-1), 2=forward (from device+1)
FORCE_INLINE uint32_t get_receive_direction(int32_t offset, uint32_t step, uint32_t num_steps_total, uint32_t ring_sz) {
    bool is_final = (step == num_steps_total - 1);

    if (is_final && offset == 0) {
        return 2;  // forward (from +1)
    }

    if (is_left_receiver(offset, step, ring_sz)) {
        return 1;  // backward
    }

    if (is_right_receiver(offset, step)) {
        return 2;  // forward
    }

    return 0;
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);

    // Semaphore addresses on destination devices (symmetric allocation)
    const uint8_t dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
    size_t dest_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // destination's semaphore

    uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tile_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t end_tile_offset = get_arg_val<uint32_t>(arg_idx++);

#ifdef USE_WORKER_MUX
    // Mux connection runtime args
    [[maybe_unused]] bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++);
    bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_sync_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_fabric_mux_status_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_teardown_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
#endif

    ///////////////////////////////////////////////////
    // TENSOR ACCESSOR SETUP
    ///////////////////////////////////////////////////

    constexpr auto input_tensor_args = TensorAccessorArgs<tensor_args_start>();
    constexpr uint32_t input_ct_offset = input_tensor_args.num_compile_time_args();
    auto input_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, page_size);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<tensor_args_start + input_ct_offset>();
    constexpr uint32_t intermediate_ct_offset = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_tensor_address, page_size);

    constexpr auto output_tensor_args =
        TensorAccessorArgs<tensor_args_start + input_ct_offset + intermediate_ct_offset>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_tensor_address, page_size);

    ///////////////////////////////////////////////////
    // FABRIC CONNECTION SETUP
    ///////////////////////////////////////////////////

#ifdef USE_WORKER_MUX
    auto mux_connection_handle = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
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

    // Wait for fabric mux to be ready to accept connections
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    tt::tt_fabric::fabric_client_connect(mux_connection_handle);
#else
    size_t fab_arg_idx = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(fab_arg_idx);

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }
#endif

    // Allocate packet headers
    auto pkt_hdr_data = PacketHeaderPool::allocate_header();
    auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();

    // Set up routing
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_data, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);

    ///////////////////////////////////////////////////
    // DERIVED CONSTANTS
    ///////////////////////////////////////////////////

    constexpr uint32_t slice_num_pages = slice_C * slice_Ht * slice_Wt;
    constexpr uint32_t batch_num_pages = ring_size * slice_num_pages;
    constexpr uint32_t channel_num_pages = slice_Ht * slice_Wt;
    constexpr uint32_t tiles_per_channel = (dim == 1) ? slice_num_pages : channel_num_pages;
    constexpr uint32_t num_channel_iters = (dim == 1) ? 1 : slice_C;

    // Direction mask: forward=2, backward=1
    constexpr uint32_t my_send_direction = direction ? 2 : 1;
    constexpr uint32_t my_receive_direction = direction ? 2 : 1;

    // Intermediate buffer layout helper
    auto get_intermediate_tile_id = [](uint32_t buffer_idx, uint32_t slice_idx, uint32_t tile_offset) -> uint32_t {
        return buffer_idx * ring_size * slice_num_pages + slice_idx * slice_num_pages + tile_offset;
    };

    // Input tensor tile ID helper
    auto get_input_tile_id =
        [](uint32_t slice_idx, uint32_t batch_idx, uint32_t channel_idx, uint32_t tile_offset) -> uint32_t {
        if constexpr (dim == 3) {
            return batch_idx * batch_num_pages + channel_idx * channel_num_pages + slice_idx * slice_Wt + tile_offset;
        } else if constexpr (dim == 2) {
            return batch_idx * batch_num_pages + channel_idx * channel_num_pages + slice_idx * slice_Ht * slice_Wt +
                   tile_offset;
        } else if constexpr (dim == 1) {
            return batch_idx * batch_num_pages + slice_idx * slice_C * slice_Ht * slice_Wt + tile_offset;
        } else {
            return 0;
        }
    };

    uint32_t chunk_count = 0;

    ///////////////////////////////////////////////////
    // MAIN LOOP
    ///////////////////////////////////////////////////

    for (uint32_t batch = 0; batch < input_tensor_B; ++batch) {
        for (uint32_t step = 0; step < num_steps; ++step) {
            uint32_t recv_buffer_idx = step % 2;
            uint32_t accum_buffer_idx = (step > 0) ? ((step - 1) % 2) : 0;

            // ========== CONFIGURE PACKET HEADERS FOR THIS STEP ==========
            // Tree algorithm hop counts:
            // - Steps 0 to num_steps-2: stride = 2^step (tree levels collapse inward)
            // - Final step (num_steps-1): stride = 1 (tree roots +1/-1 send to owner 0)
            uint32_t stride = (step == num_steps - 1) ? 1u : (1u << step);

            // Configure packet headers with correct hop count
            fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                pkt_hdr_data, static_cast<uint8_t>(stride), nullptr, page_size);

            fabric_unicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                pkt_hdr_seminc, static_cast<uint8_t>(stride), tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});

            // ========== PHASE 1: SEND DATA ==========
            for (uint32_t slice = 0; slice < ring_size; ++slice) {
                int32_t offset = compute_offset(my_chip_id, slice, ring_size);
                uint32_t send_dir = get_send_direction(offset, step, num_steps, ring_size);

                // Only process if this slice's send direction matches our worker direction
                if (send_dir != my_send_direction) {
                    continue;
                }

#ifdef USE_WORKER_MUX
                auto* fabric_conn = &mux_connection_handle;
#else
                auto* fabric_conn = direction ? &fabric_connection.get_forward_connection()
                                              : &fabric_connection.get_backward_connection();
#endif

                // Process tiles for this slice within our assigned range
                for (uint32_t c = 0; c < num_channel_iters; ++c) {
                    uint32_t channel_tile_base = c * tiles_per_channel;

                    // Compute tile range for this channel
                    uint32_t tile_start, tile_end;
                    if constexpr (dim == 1) {
                        tile_start = start_tile_offset;
                        tile_end = end_tile_offset;
                    } else {
                        uint32_t global_tile_start = c * tiles_per_channel;
                        uint32_t global_tile_end = (c + 1) * tiles_per_channel;

                        if (end_tile_offset <= global_tile_start || start_tile_offset >= global_tile_end) {
                            continue;
                        }

                        tile_start =
                            (start_tile_offset > global_tile_start) ? (start_tile_offset - global_tile_start) : 0;
                        tile_end = (end_tile_offset < global_tile_end) ? (end_tile_offset - global_tile_start)
                                                                       : tiles_per_channel;
                    }

                    for (uint32_t tile_offset = tile_start; tile_offset < tile_end; tile_offset += tile_granularity) {
                        uint32_t tiles_this_chunk = std::min(tile_granularity, tile_end - tile_offset);

                        // Read data into send CB
                        cb_reserve_back(cb_send_id, tiles_this_chunk);
                        uint32_t l1_addr = get_write_ptr(cb_send_id);

                        for (uint32_t t = 0; t < tiles_this_chunk; ++t) {
                            uint64_t src_noc_addr;
                            if (step == 0) {
                                uint32_t tile_id = get_input_tile_id(slice, batch, c, tile_offset + t);
                                src_noc_addr = get_noc_addr(tile_id, input_addrgen);
                            } else {
                                uint32_t tile_id = get_intermediate_tile_id(
                                    accum_buffer_idx, slice, channel_tile_base + tile_offset + t);
                                src_noc_addr = get_noc_addr(tile_id, intermediate_addrgen);
                            }
                            noc_async_read(src_noc_addr, l1_addr, page_size);
                            l1_addr += page_size;
                        }
                        noc_async_read_barrier();

                        // Send via fabric
                        l1_addr = get_write_ptr(cb_send_id);
                        for (uint32_t t = 0; t < tiles_this_chunk; ++t) {
                            uint32_t dest_tile_id =
                                get_intermediate_tile_id(recv_buffer_idx, slice, channel_tile_base + tile_offset + t);
                            uint64_t dest_noc_addr = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                intermediate_addrgen, dest_tile_id, 0);

                            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                                fabric_conn, pkt_hdr_data, l1_addr, NocUnicastCommandHeader{dest_noc_addr});

                            l1_addr += page_size;
                        }
                        noc_async_writes_flushed();

                        cb_pop_front(cb_send_id, tiles_this_chunk);

                        // Signal semaphore based on chunks_per_sync
                        chunk_count++;
                        if (chunk_count % chunks_per_sync == 0) {
                            uint64_t sem_noc_addr = safe_get_noc_addr(dest_noc_x, dest_noc_y, dest_sem_addr, 0);
                            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                                fabric_conn,
                                pkt_hdr_seminc,
                                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
                        }
                    }
                }

                // Final semaphore signal for this slice if not already sent
                if (chunk_count % chunks_per_sync != 0) {
                    uint64_t sem_noc_addr = safe_get_noc_addr(dest_noc_x, dest_noc_y, dest_sem_addr, 0);
#ifdef USE_WORKER_MUX
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        &mux_connection_handle,
                        pkt_hdr_seminc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
#else
                    auto* fabric_conn_sem = direction ? &fabric_connection.get_forward_connection()
                                                      : &fabric_connection.get_backward_connection();
                    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                        fabric_conn_sem,
                        pkt_hdr_seminc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
#endif
                }
            }

            noc_async_writes_flushed();

            // ========== PHASE 2: STORE REDUCED RESULTS ==========
            for (uint32_t slice = 0; slice < ring_size; ++slice) {
                int32_t offset = compute_offset(my_chip_id, slice, ring_size);
                uint32_t recv_dir = get_receive_direction(offset, step, num_steps, ring_size);

                // Only process if this slice's receive direction matches our worker direction
                if (recv_dir != my_receive_direction) {
                    continue;
                }

                bool is_final_for_slice = (step == num_steps - 1) && (offset == 0);

                // Process tiles within our assigned range
                for (uint32_t c = 0; c < num_channel_iters; ++c) {
                    uint32_t channel_tile_base = c * tiles_per_channel;

                    // Compute tile range for this channel
                    uint32_t tile_start, tile_end;
                    if constexpr (dim == 1) {
                        tile_start = start_tile_offset;
                        tile_end = end_tile_offset;
                    } else {
                        uint32_t global_tile_start = c * tiles_per_channel;
                        uint32_t global_tile_end = (c + 1) * tiles_per_channel;

                        if (end_tile_offset <= global_tile_start || start_tile_offset >= global_tile_end) {
                            continue;
                        }

                        tile_start =
                            (start_tile_offset > global_tile_start) ? (start_tile_offset - global_tile_start) : 0;
                        tile_end = (end_tile_offset < global_tile_end) ? (end_tile_offset - global_tile_start)
                                                                       : tiles_per_channel;
                    }

                    for (uint32_t tile_offset = tile_start; tile_offset < tile_end; tile_offset += tile_granularity) {
                        uint32_t tiles_this_chunk = std::min(tile_granularity, tile_end - tile_offset);

                        cb_wait_front(cb_output_id, tiles_this_chunk);
                        uint32_t l1_addr = get_read_ptr(cb_output_id);

                        for (uint32_t t = 0; t < tiles_this_chunk; ++t) {
                            uint64_t dest_noc_addr;
                            if (is_final_for_slice) {
                                uint32_t output_tile_id = batch * slice_num_pages + channel_tile_base + tile_offset + t;
                                dest_noc_addr = get_noc_addr(output_tile_id, output_addrgen);
                            } else {
                                uint32_t tile_id = get_intermediate_tile_id(
                                    recv_buffer_idx, slice, channel_tile_base + tile_offset + t);
                                dest_noc_addr = get_noc_addr(tile_id, intermediate_addrgen);
                            }
                            noc_async_write(l1_addr, dest_noc_addr, page_size);
                            l1_addr += page_size;
                        }
                        noc_async_write_barrier();

                        cb_pop_front(cb_output_id, tiles_this_chunk);
                    }
                }
            }
        }
    }

    // Cleanup
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#ifdef USE_WORKER_MUX
    tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);
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
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
#endif
}
