// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "matmul_dataflow_common.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(9);
constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
constexpr uint32_t in0_tile_size = get_compile_time_arg_val(11);
constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
constexpr uint32_t is_output_writer = get_compile_time_arg_val(14);
constexpr uint32_t is_injector_core = get_compile_time_arg_val(15);
constexpr uint32_t num_devices = get_compile_time_arg_val(16);
constexpr uint32_t my_rank = get_compile_time_arg_val(17);
constexpr uint32_t in3_tile_size = get_compile_time_arg_val(18);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(19);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(20);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(21);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(22));
constexpr bool is_linear = (topology == Topology::Linear);
constexpr uint32_t N_chunks = get_compile_time_arg_val(23);
constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(24);
constexpr uint32_t K_tiles_per_device = get_compile_time_arg_val(25);
constexpr uint32_t K_block_tail_tiles = get_compile_time_arg_val(26);

#ifdef USE_MUX
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(27);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(28);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(29);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(30);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(31);

constexpr uint32_t mux_arg_count = 32;

constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info_forward =
    ccl_routing_utils::get_line_unicast_route_info_from_args<mux_arg_count>();

constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info_backward =
    ccl_routing_utils::get_line_unicast_route_info_from_args<
        mux_arg_count + ccl_routing_utils::num_line_unicast_args>();

constexpr uint32_t ct_arg_count = mux_arg_count + 2 * ccl_routing_utils::num_line_unicast_args;
#else
constexpr uint32_t ct_arg_count = 27;
#endif

namespace detail {

bool valid_targets_forward(const bool direction) {
    if constexpr (num_targets_forward_direction) {
        return (direction == 0);
    } else {
        return false;
    }
}

bool valid_targets_backward(const bool direction) {
    if constexpr (num_targets_backward_direction) {
        return (direction == 1);
    } else {
        return false;
    }
}

bool valid_targets(const bool direction) {
    if constexpr (num_targets_backward_direction + num_targets_forward_direction == 0) {
        return false;
    } else {
        return (valid_targets_forward(direction) || valid_targets_backward(direction));
    }
}
}  // namespace detail

void kernel_main() {
    // Load common runtime args (same for all cores, updated in override_runtime_arguments)
    uint32_t cargidx = 0;
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t in2_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t in3_addr = get_common_arg_val<uint32_t>(cargidx++);
    size_t out_ready_sem_backward = get_common_arg_val<uint32_t>(cargidx++);
    size_t out_ready_sem_forward = get_common_arg_val<uint32_t>(cargidx++);

#ifdef FUSE_TERNARY
    const uint32_t ternary_a_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t ternary_b_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t broadcast_ternary_b = get_common_arg_val<uint32_t>(cargidx++);
#endif  // FUSE_TERNARY

    // Output tensor addresses from common args
    const uint32_t out_addr_common_arg_start = cargidx;

    // Load per-core runtime args
    uint32_t argidx = 0;
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_receiver_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_valid_semaphore_id = get_arg_val<uint32_t>(argidx++);
    Semaphore<> in0_sender_sem(in0_sender_semaphore_id);
    Semaphore<> in0_receiver_sem(in0_receiver_semaphore_id);
    Semaphore<> in0_valid_sem(in0_valid_semaphore_id);
    uint32_t in0_valid_semaphore_addr = get_semaphore(in0_valid_semaphore_id);
    uint32_t in0_receiver_semaphore_addr = get_semaphore(in0_receiver_semaphore_id);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_injector_noc0_x = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_injector_noc0_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_core_order_index = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_core_order_size = get_arg_val<uint32_t>(argidx++);

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<ct_arg_count>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);

    // Always create tuple of output accessors (size = N_chunks) - addresses from common args
    constexpr uint32_t out_tensor_args_cta_offset = in0_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple =
        make_tensor_accessor_tuple_uniform_page_size_common(outputs_args, out_addr_common_arg_start, out_tile_size);

#ifdef USE_MUX
    uint32_t backward_in0_core_order_index = in0_core_order_size - 2;
    uint32_t forward_in0_core_order_index = in0_core_order_size - 1;

    // Each fabric-sender core only parses + connects the SINGLE direction it actually uses.
    // The program factory pushes RT args for exactly one direction per core, so argidx alignment
    // stays correct. The unused-direction mux/handle is default-initialized (connection_valid=false)
    // so close_mux below skips it.
    MuxConnection<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes> mux_backward{};
    MuxConnection<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes> mux_forward{};
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle_backward =
        nullptr;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle_forward = nullptr;
    if (in0_core_order_index == backward_in0_core_order_index) {
        mux_backward =
            parse_mux_connection_args<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes>(
                argidx, in0_core_order_index, backward_in0_core_order_index);
        mux_connection_handle_backward = mux_backward.build_and_connect(fabric_mux_status_address);
    } else if (in0_core_order_index == forward_in0_core_order_index) {
        mux_forward =
            parse_mux_connection_args<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes>(
                argidx, in0_core_order_index, forward_in0_core_order_index);
        mux_connection_handle_forward = mux_forward.build_and_connect(fabric_mux_status_address);
    }
#endif

#ifdef FUSE_BIAS
    constexpr uint32_t in2_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in2_args = TensorAccessorArgs<in2_args_cta_offset>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr);
#endif

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    // K_blocks_per_device uses div_up semantics: when K_block_tiles does not evenly divide
    // K_tiles_per_device, the last block per device is a "tail" block of K_block_tail_tiles
    // (< K_block_tiles). When it divides cleanly, K_block_tail_tiles == K_block_tiles.
    constexpr uint32_t K_blocks_per_device = (K_tiles_per_device + K_block_tiles - 1) / K_block_tiles;
    constexpr uint32_t K_num_blocks = K_blocks_per_device * num_devices;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    // SwiGLU emits one output tile per interleaved gate/up pair -> output N is half the
    // matmul (weight) N. Weight-space n ranges are halved at each write call site.
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
#endif

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_4;
#endif

    Noc noc_obj;
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_out(cb_id_out);
#ifdef FUSE_BIAS
    CircularBuffer cb_in2(cb_id_in2);
#endif

#ifdef READ_FROM_LOCAL_INPUT
#ifdef FUSE_BIAS
    constexpr auto in3_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
#else
    constexpr uint32_t in3_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in3_args = TensorAccessorArgs<in3_args_cta_offset>();
#endif
    const auto in3_reader = TensorAccessor(in3_args, in3_addr);
#endif

#ifdef FUSE_TERNARY
#ifdef READ_FROM_LOCAL_INPUT
    constexpr auto ternary_a_args = TensorAccessorArgs<in3_args.next_compile_time_args_offset()>();
#else
#ifdef FUSE_BIAS
    constexpr auto ternary_a_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
#else
    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto ternary_a_args = TensorAccessorArgs<ternary_a_args_cta_offset>();
#endif
#endif
    constexpr auto ternary_b_args = TensorAccessorArgs<ternary_a_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_ternary_a = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_ternary_b = tt::CBIndex::c_6;
    CircularBuffer cb_ternary_a(cb_id_ternary_a);
    CircularBuffer cb_ternary_b(cb_id_ternary_b);

    constexpr uint32_t ternary_a_tile_size = get_tile_size(cb_id_ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(cb_id_ternary_b);

    const auto ternary_a_reader = TensorAccessor(ternary_a_args, ternary_a_addr);
    const auto ternary_b_reader = TensorAccessor(ternary_b_args, ternary_b_addr);
#endif

    in0_valid_sem.set(VALID);

    const uint64_t in0_receiver_semaphore_noc_addr =
        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);
    // all gather
    volatile tt_l1_ptr uint32_t* out_ready_sem_backward_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_backward);
    volatile tt_l1_ptr uint32_t* out_ready_sem_forward_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_forward);
    uint32_t sem_target_backward = 0;
    uint32_t sem_target_forward = 0;
    uint64_t out_ready_sem_noc_addr_backward_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_backward, 0);
    uint64_t out_ready_sem_noc_addr_forward_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_forward, 0);
    uint64_t out_ready_sem_injector_noc_addr_backward_in_pkt =
        safe_get_noc_addr(out_ready_sem_injector_noc0_x, out_ready_sem_injector_noc0_y, out_ready_sem_backward, 0);
    uint64_t out_ready_sem_injector_noc_addr_forward_in_pkt =
        safe_get_noc_addr(out_ready_sem_injector_noc0_x, out_ready_sem_injector_noc0_y, out_ready_sem_forward, 0);

#ifdef USE_MUX
    auto pkt_hdrs_backward = allocate_and_init_packet_headers(
        detail::valid_targets(1),
        unicast_route_info_backward,
        in0_reader,
        num_tiles_to_write_per_packet,
        in3_tile_size);

    auto pkt_hdrs_forward = allocate_and_init_packet_headers(
        detail::valid_targets(0), unicast_route_info_forward, in0_reader, num_tiles_to_write_per_packet, in3_tile_size);
#endif

    /**
     * This is a Row-Major output block ordering.
     * It enables reuse of the last in0 block when striding the output block N dimension.
     */

    bool k_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
        uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;

        // When striding M block, in0 gets no reuse
        reuse_block = false;
        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
#ifdef FUSE_SWIGLU
                        cb_out.wait_front(out_block_num_tiles_swiglu);
                        uint32_t out_read_ptr_swiglu = cb_out.get_read_ptr();
                        write_block_sync<M_block_tiles, out_N_block_tiles>(
                            noc_obj,
                            std::get<0>(outputs_tuple),
                            out_shape_swiglu,
                            out_read_ptr_swiglu,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile / 2,
                            defer_write_n_tile_end / 2);
                        cb_out.pop_front(out_block_num_tiles_swiglu);
#else
                        cb_out.wait_front(out_block_num_tiles);
                        uint32_t out_read_ptr = cb_out.get_read_ptr();

                        // write_block_sync_split is more generic (support multiple output tensors)
                        // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync should be faster
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, N_block_tiles>(
                                noc_obj,
                                std::get<0>(outputs_tuple),
                                out_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        } else {
                            write_block_sync_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                                noc_obj,
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        }
                        cb_out.pop_front(out_block_num_tiles);
#endif  // FUSE_SWIGLU
                    }
                }
                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
                    continue;
                }
                cb_in0.reserve_back(in0_block_num_tiles);

                uint32_t in0_start_address = cb_in0.get_write_ptr();

                uint32_t k_block_left_tile = 0;
                uint32_t k_block_right_tile = 0;
                uint32_t actual_k_block = k_forward ? k_block_iter : (K_num_blocks - 1 - k_block_iter);
                uint32_t device_k_block_iter = actual_k_block % K_blocks_per_device;
                bool is_tail_k_block = (device_k_block_iter == K_blocks_per_device - 1);
                uint32_t current_K_block_tiles = is_tail_k_block ? K_block_tail_tiles : K_block_tiles;
                bool k_block_odd = (actual_k_block % K_blocks_per_device) & 1;
                uint32_t k_left_tiles, k_right_tiles;
                if constexpr (is_linear) {
                    // Linear: full block from one direction. Tail block has current_K_block_tiles
                    // valid tiles + (K_block_tiles - current_K_block_tiles) zero-fill tiles to
                    // preserve the K_block_tiles L1 row stride in the CB layout.
                    k_left_tiles = current_K_block_tiles;
                    k_right_tiles = K_block_tiles - current_K_block_tiles;
                } else {
                    // Ring: bidirectional half-block. Requires K_block_tiles | K_tiles_per_device
                    // (no tail support yet — enforced upstream for Ring).
                    k_left_tiles = k_block_odd ? (K_block_tiles - (K_block_tiles / 2)) : (K_block_tiles / 2);
                    k_right_tiles = k_block_odd ? (K_block_tiles / 2) : (K_block_tiles - k_left_tiles);
                }
                compute_actual_k_block<is_linear>(
                    k_block_iter,
                    K_num_blocks,
                    my_rank,
                    K_blocks_per_device,
                    K_block_tiles,
                    K_tiles_per_device,
                    num_devices,
                    k_forward,
                    n_block_iter == 0,
                    out_ready_sem_forward_addr_ptr,
                    out_ready_sem_backward_addr_ptr,
                    sem_target_forward,
                    sem_target_backward,
                    is_injector_core,
                    1,
                    k_left_tiles,
                    k_block_left_tile,
                    k_block_right_tile);
                if constexpr (is_linear) {
                    // For Linear tail, the right half is pure zero-fill padding (no real tiles).
                    // Point past the logical K end so read_in0_block_sync's `j < logical_d1` check
                    // triggers fill_zeros_async for every right-half tile.
                    if (is_tail_k_block) {
                        k_block_right_tile = K_tiles;
                    }
                }
                if (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        noc_obj,
                        in0_reader,
                        in0_shape,
                        cb_in0,
                        in0_tile_size,
#ifdef READ_FROM_LOCAL_INPUT
                        in3_reader,
                        my_rank * K_tiles_per_device,
                        ((my_rank + 1) * K_tiles_per_device) - 1,
                        K_tiles_per_device,
#endif
                        m_tile,
                        m_tile_end,
                        k_block_left_tile,
                        k_block_left_tile + k_left_tiles,
                        k_left_tiles,
                        k_block_right_tile,
                        k_block_right_tile + k_right_tiles,
                        k_right_tiles);
                } else {
                    // Get from previous device
                    in0_receiver_sem.set(INVALID);
                    in0_sender_sem.up(noc_obj, in0_sender_noc_x, in0_sender_noc_y, 1);
                    in0_receiver_sem.wait(VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_in0.push_back(in0_block_num_tiles);
                if (!is_sink_core) {
                    in0_sender_sem.wait(1);
                    in0_sender_sem.set(0);

                    /**
                     * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                     * padded tiles. Use `current_block_bytes`.
                     */
                    noc_obj.async_write(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        UnicastEndpoint{},
                        current_block_bytes,
                        {},
                        {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});

#ifdef ARCH_BLACKHOLE
                    noc_obj.async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                }
#ifdef USE_MUX
                if (n_block_iter == 0) {
                    if constexpr (is_linear) {
                        // Linear uni-ring: every (non-skipped) iter, every device sends ONE full
                        // K-block to its predecessor in the virtual ring. Dev 0
                        // (num_targets_backward_direction == 0) long-sends to Dev N-1 via
                        // mux_backward (forward direction, N-1 hops; the routing was overridden in
                        // the program factory to set distance_in_hops = N-1). Other devices
                        // short-send to my_rank-1 via mux_forward (backward direction, 1 hop). All
                        // sends signal out_ready_semaphore_forward at the receiver.
                        //
                        // Skip the last K_blocks_per_device iters: by then every block has already
                        // reached every device, so this final relay lap is redundant -- it would
                        // re-deliver each device's own data back to it and fire sem increments the
                        // receiver never waits on (which is why the receiver no longer needs to
                        // compensate sem_target; see compute_actual_k_block). Mirrors the Ring skip.
                        if constexpr (num_targets_backward_direction == 0) {
                            // Dev 0 (chain head): long send via mux_backward + pkt_hdrs_forward
                            if constexpr (num_targets_forward_direction > 0) {
                                if (in0_core_order_index >= backward_in0_core_order_index &&
                                    in0_core_order_index < forward_in0_core_order_index &&
                                    k_block_iter < (K_num_blocks - K_blocks_per_device)) {
                                    forward_half_block_to_fabric_neighbor(
                                        noc_obj,
                                        m_tile,
                                        k_block_left_tile,
                                        current_M_block_tiles,
                                        k_left_tiles,
                                        k_right_tiles,
                                        num_tiles_to_write_per_packet,
                                        in0_start_address,
                                        K_tiles,
                                        in0_reader,
                                        mux_connection_handle_backward,
                                        pkt_hdrs_forward,
                                        in0_tile_size,
                                        out_ready_sem_injector_noc_addr_forward_in_pkt,
                                        true,
                                        M_tiles,
                                        true);
                                }
                            }
                        } else {
                            // Dev k > 0: short send via mux_forward + pkt_hdrs_backward
                            if (in0_core_order_index >= forward_in0_core_order_index &&
                                k_block_iter < (K_num_blocks - K_blocks_per_device)) {
                                forward_half_block_to_fabric_neighbor(
                                    noc_obj,
                                    m_tile,
                                    k_block_left_tile,
                                    current_M_block_tiles,
                                    k_left_tiles,
                                    k_right_tiles,
                                    num_tiles_to_write_per_packet,
                                    in0_start_address,
                                    K_tiles,
                                    in0_reader,
                                    mux_connection_handle_forward,
                                    pkt_hdrs_backward,
                                    in0_tile_size,
                                    out_ready_sem_injector_noc_addr_forward_in_pkt,
                                    true,
                                    M_tiles,
                                    true);
                            }
                        }
                    } else {
                        // Ring: relay each K-block both directions every iter. Skip the last
                        // K_blocks_per_device iterations — those K-blocks have already reached
                        // all devices via wrap-around relay.
                        if (k_block_iter < (K_num_blocks - (K_num_blocks / num_devices))) {
                            if constexpr (num_targets_backward_direction > 0) {
                                if (in0_core_order_index >= forward_in0_core_order_index) {
                                    forward_half_block_to_fabric_neighbor(
                                        noc_obj,
                                        m_tile,
                                        k_block_left_tile,
                                        current_M_block_tiles,
                                        k_left_tiles,
                                        k_right_tiles,
                                        num_tiles_to_write_per_packet,
                                        in0_start_address,
                                        K_tiles,
                                        in0_reader,
                                        mux_connection_handle_forward,
                                        pkt_hdrs_backward,
                                        in0_tile_size,
                                        out_ready_sem_injector_noc_addr_forward_in_pkt,
                                        true,
                                        M_tiles,
                                        true);
                                }
                            }
                            if constexpr (num_targets_forward_direction > 0) {
                                if (in0_core_order_index >= backward_in0_core_order_index &&
                                    in0_core_order_index < forward_in0_core_order_index) {
                                    forward_half_block_to_fabric_neighbor(
                                        noc_obj,
                                        m_tile,
                                        k_block_right_tile,
                                        current_M_block_tiles,
                                        k_left_tiles,
                                        k_right_tiles,
                                        num_tiles_to_write_per_packet,
                                        in0_start_address,
                                        K_tiles,
                                        in0_reader,
                                        mux_connection_handle_backward,
                                        pkt_hdrs_forward,
                                        in0_tile_size,
                                        out_ready_sem_injector_noc_addr_backward_in_pkt,
                                        false,
                                        M_tiles,
                                        true);
                                }
                            }
                        }
                    }
                }
#endif  // USE_MUX
            }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_in2.reserve_back(N_block_tiles);

                uint32_t l1_write_addr_in2 = cb_in2.get_write_ptr();
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc_obj.async_read(
                        in2_reader,
                        CoreLocalMem<uint8_t>(l1_write_addr_in2),
                        in2_tile_size,
                        {.page_id = n_tile_id},
                        {});
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc_obj.async_read_barrier();

                cb_in2.push_back(N_block_tiles);
            }
#endif

#ifdef FUSE_TERNARY
            if constexpr (!is_output_writer) {
                read_ternary_blocks_sync<M_block_tiles, N_block_tiles>(
                    noc_obj,
                    ternary_a_reader,
                    ternary_b_reader,
                    out_shape,
                    cb_ternary_a,
                    cb_ternary_b,
                    ternary_a_tile_size,
                    ternary_b_tile_size,
                    broadcast_ternary_b,
                    m_tile,
                    m_tile_end,
                    n_tile,
                    n_tile_end);
            }
#endif

            if constexpr (!is_linear) {
                k_forward = !k_forward;
            }
            // Ring reuses in0_cb across N strides because k_forward toggles, so n=X's last
            // actual_k_block equals n=(X+1)'s first. Linear keeps k_forward=true: reusing would
            // feed n=X's K-block (K_num_blocks-1) as if it were K-block 0 — fresh read required.
            if constexpr (!is_linear) {
                reuse_block = true;
            }

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !((m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1)));
            defer_write = defer_write && !is_injector_core;

            if (!defer_write) {
                if constexpr (is_output_writer) {
#ifdef FUSE_SWIGLU
                    write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                        noc_obj,
                        std::get<0>(outputs_tuple),
                        out_shape_swiglu,
                        cb_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile / 2,
                        n_tile_end / 2);
#else
                    // write_block_sync_granular_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            noc_obj,
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            noc_obj,
                            outputs_tuple,
                            out0_shape,
                            cb_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    }
#endif  // FUSE_SWIGLU
                }
            }
        }
    }

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();

#ifdef USE_MUX
    if (mux_backward.connection_valid) {
        close_mux(
            noc_obj,
            mux_connection_handle_backward,
            mux_backward.is_termination_master,
            mux_backward.termination_sync_address,
            mux_backward.num_mux_clients,
            mux_backward.fabric_mux_x,
            mux_backward.fabric_mux_y,
            fabric_mux_termination_signal_address,
            mux_backward.termination_master_noc_x,
            mux_backward.termination_master_noc_y);
    }
    if (mux_forward.connection_valid) {
        close_mux(
            noc_obj,
            mux_connection_handle_forward,
            mux_forward.is_termination_master,
            mux_forward.termination_sync_address,
            mux_forward.num_mux_clients,
            mux_forward.fabric_mux_x,
            mux_forward.fabric_mux_y,
            fabric_mux_termination_signal_address,
            mux_forward.termination_master_noc_x,
            mux_forward.termination_master_noc_y);
    }
#endif  // USE_MUX

    noc_obj.async_write_barrier();

    noc_semaphore_set(out_ready_sem_backward_addr_ptr, 0);
    noc_semaphore_set(out_ready_sem_forward_addr_ptr, 0);
}
