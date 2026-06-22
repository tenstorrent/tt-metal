// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

using ttnn::ccl::Topology;

#ifdef FSDP_FUSED
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#endif

void kernel_main() {
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
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(14);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(15);
    constexpr uint32_t num_devices = get_compile_time_arg_val(16);
    constexpr uint32_t my_rank = get_compile_time_arg_val(17);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(19);
    constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(20));
    constexpr bool is_linear = (topology == Topology::Linear);
    constexpr uint32_t K_tiles_per_device = get_compile_time_arg_val(21);
    constexpr uint32_t K_block_tail_tiles = get_compile_time_arg_val(22);

#ifdef FSDP_FUSED
    // FSDP CT args follow target's base args (topology=20, K_tiles_per_device=21, K_block_tail_tiles=22).
    constexpr uint32_t fsdp_ring_size = get_compile_time_arg_val(23);
    constexpr uint32_t fsdp_ring_index = get_compile_time_arg_val(24);
    constexpr uint32_t fsdp_num_targets_forward = get_compile_time_arg_val(25);
    constexpr uint32_t fsdp_num_targets_backward = get_compile_time_arg_val(26);
    constexpr Topology fsdp_topology = static_cast<Topology>(get_compile_time_arg_val(27));

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(28);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(29);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(30);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(31);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(32);

    constexpr uint32_t fsdp_mux_arg_count = 33;

    constexpr ccl_routing_utils::line_unicast_route_info_t fsdp_unicast_route_info_forward =
        ccl_routing_utils::get_line_unicast_route_info_from_args<fsdp_mux_arg_count>();
    constexpr ccl_routing_utils::line_unicast_route_info_t fsdp_unicast_route_info_backward =
        ccl_routing_utils::get_line_unicast_route_info_from_args<
            fsdp_mux_arg_count + ccl_routing_utils::num_line_unicast_args>();

    constexpr uint32_t ct_arg_count = fsdp_mux_arg_count + 2 * ccl_routing_utils::num_line_unicast_args;
#else
    constexpr uint32_t ct_arg_count = 23;
#endif

    // Load common runtime args (same for all cores, updated in override_runtime_arguments)
    uint32_t cargidx = 0;
    const uint32_t in1_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t in2_addr = get_common_arg_val<uint32_t>(cargidx++);

#ifdef FUSE_TERNARY
    const uint32_t ternary_a_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t ternary_b_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t broadcast_ternary_b = get_common_arg_val<uint32_t>(cargidx++);
#endif  // FUSE_TERNARY

#ifdef FSDP_FUSED
    // FSDP common args: local_weight (FSDP-sharded source) + the two FSDP semaphores that remote
    // fabric senders atomic-inc into. in1_addr above already points to the PWB (gathered weight).
    const uint32_t local_weight_addr = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t fsdp_sem_backward = get_common_arg_val<uint32_t>(cargidx++);
    const uint32_t fsdp_sem_forward = get_common_arg_val<uint32_t>(cargidx++);
#endif

    // Output tensor addresses from common args
    const uint32_t out_addr_common_arg_start = cargidx;

    // Load per-core runtime args
    uint32_t argidx = 0;
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    uint32_t in1_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
    uint32_t in1_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
    uint32_t in1_valid_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);

#ifdef FSDP_FUSED
    const uint32_t my_virtual_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t my_virtual_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t injector_virtual_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t injector_virtual_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_core_order_index = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_core_order_size = get_arg_val<uint32_t>(argidx++);
    // Chain-order indices of this row's backward/forward fabric senders (cores co-located with their
    // mux columns). Computed host-side so the relay gate is a per-row column match, not a fixed tail.
    const uint32_t fsdp_backward_in1_core_order_index = get_arg_val<uint32_t>(argidx++);
    const uint32_t fsdp_forward_in1_core_order_index = get_arg_val<uint32_t>(argidx++);
#endif

    // Tensor accessor for input tensor (PWB when FSDP_FUSED, else FSDP-sharded local weight = full weight when not
    // FSDP).
    constexpr auto in1_args = TensorAccessorArgs<ct_arg_count>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr);

    // Always create tuple of output accessors (size = N_chunks) - addresses from common args
    constexpr uint32_t out_tensor_args_cta_offset = in1_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple =
        make_tensor_accessor_tuple_uniform_page_size_common(outputs_args, out_addr_common_arg_start, out_tile_size);
#ifdef FUSE_BIAS
    constexpr uint32_t in2_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in2_args = TensorAccessorArgs<in2_args_cta_offset>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr);
#endif

#ifdef FSDP_FUSED
    // FSDP-sharded local weight accessor (appended by append_accessors after bias, matching its order).
#ifdef FUSE_BIAS
    constexpr auto local_weight_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
#else
    constexpr uint32_t local_weight_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto local_weight_args = TensorAccessorArgs<local_weight_args_cta_offset>();
#endif
    const auto local_weight_reader = TensorAccessor(local_weight_args, local_weight_addr);
#endif

#ifdef FUSE_TERNARY
#ifdef FSDP_FUSED
    constexpr auto ternary_a_args = TensorAccessorArgs<local_weight_args.next_compile_time_args_offset()>();
#elif defined(FUSE_BIAS)
    constexpr auto ternary_a_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
#else
    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto ternary_a_args = TensorAccessorArgs<ternary_a_args_cta_offset>();
#endif
    constexpr auto ternary_b_args = TensorAccessorArgs<ternary_a_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_ternary_a = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_ternary_b = tt::CBIndex::c_6;

    constexpr uint32_t ternary_a_tile_size = get_tile_size(cb_id_ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(cb_id_ternary_b);

    const auto ternary_a_reader = TensorAccessor(ternary_a_args, ternary_a_addr);
    const auto ternary_b_reader = TensorAccessor(ternary_b_args, ternary_b_addr);
#endif

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    // K_blocks_per_device uses div_up semantics: when K_block_tiles does not evenly divide
    // K_tiles_per_device, the last block per device is a "tail" block of K_block_tail_tiles
    // (< K_block_tiles). When it divides cleanly, K_block_tail_tiles == K_block_tiles.
    constexpr uint32_t K_blocks_per_device = (K_tiles_per_device + K_block_tiles - 1) / K_block_tiles;
    constexpr uint32_t K_num_blocks = K_blocks_per_device * num_devices;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_4;
#endif

    volatile tt_l1_ptr uint32_t* in1_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_valid_semaphore_addr);
    *(in1_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in1_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in1_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_semaphore_addr);
    const uint64_t in1_sender_semaphore_noc_addr =
        get_noc_addr(in1_sender_noc_x, in1_sender_noc_y, in1_sender_semaphore_addr);

    const uint64_t in1_receiver_semaphore_noc_addr =
        get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, in1_receiver_semaphore_addr);

    const uint64_t in1_unicast_data_base_addr = get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, 0);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

#ifdef FSDP_FUSED
    // FSDP-local K-tile range in global K-tile units. Each device owns this slice of the K dim;
    // half-blocks whose K-tiles fall in this range are read from local_weight_reader; everything
    // else is read from in1_reader (the PWB, populated via fabric).
    constexpr uint32_t K_blocks_per_fsdp = K_num_blocks / fsdp_ring_size;
    constexpr uint32_t fsdp_K_local_start_tile = fsdp_ring_index * K_blocks_per_fsdp * K_block_tiles;
    constexpr uint32_t fsdp_K_local_end_tile = (fsdp_ring_index + 1) * K_blocks_per_fsdp * K_block_tiles;
    // PWB N width in tiles — per-device N. PWB shape is [K_full, N_local] with N_local = N_tiles.
    constexpr uint32_t pwb_N_Wt = N_tiles;

    // Mux connection setup. Each fabric-sender core only parses + connects the single direction
    // it actually uses; the program factory pushes RT args for exactly one direction per sender.
    // Non-sender cores get no mux RT args at all (so their argidx never advances into a mux slot).
    // The two sender indices are read from RT args above (column-matched senders, not the tail).
    MuxConnection<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes> fsdp_mux_backward{};
    MuxConnection<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes> fsdp_mux_forward{};
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* fsdp_mux_handle_backward = nullptr;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* fsdp_mux_handle_forward = nullptr;
    if (in1_core_order_index == fsdp_backward_in1_core_order_index) {
        fsdp_mux_backward =
            parse_mux_connection_args<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes>(
                argidx, in1_core_order_index, fsdp_backward_in1_core_order_index);
        fsdp_mux_handle_backward = fsdp_mux_backward.build_and_connect(fabric_mux_status_address);
    } else if (in1_core_order_index == fsdp_forward_in1_core_order_index) {
        fsdp_mux_forward =
            parse_mux_connection_args<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes>(
                argidx, in1_core_order_index, fsdp_forward_in1_core_order_index);
        fsdp_mux_handle_forward = fsdp_mux_forward.build_and_connect(fabric_mux_status_address);
    }

    // FSDP semaphores (forward + backward counters incremented by remote fabric sends).
    volatile tt_l1_ptr uint32_t* fsdp_sem_forward_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fsdp_sem_forward);
    volatile tt_l1_ptr uint32_t* fsdp_sem_backward_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fsdp_sem_backward);
    uint32_t fsdp_sem_target_forward = 0;
    uint32_t fsdp_sem_target_backward = 0;

    // Target NoC addresses for remote atomic-incs. When this device's fabric sender ships its
    // local K-half to a neighbor, the fabric also delivers an atomic-inc to that neighbor's
    // injector-core semaphore (NOT to ours — those addresses are sent inside the fabric packet).
    uint64_t fsdp_sem_noc_addr_forward_in_pkt =
        safe_get_noc_addr(injector_virtual_x, injector_virtual_y, fsdp_sem_forward, 0);
    uint64_t fsdp_sem_noc_addr_backward_in_pkt =
        safe_get_noc_addr(injector_virtual_x, injector_virtual_y, fsdp_sem_backward, 0);

    // Packet headers for fabric send, mirroring in0's uni-ring direction mapping:
    //   - fsdp Dev 0 (no backward neighbor) long-sends via the size-2 core's forward-routing mux
    //     (handle_backward) using the FORWARD route (N-1 hops) -> needs pkt_hdrs_forward on size-2.
    //   - fsdp Dev k short-sends via the size-1 core's backward-routing mux (handle_forward) using
    //     the BACKWARD route (1 hop) -> needs pkt_hdrs_backward on size-1.
    PacketHeaders fsdp_pkt_hdrs_backward{};
    PacketHeaders fsdp_pkt_hdrs_forward{};
    if (in1_core_order_index == fsdp_backward_in1_core_order_index) {
        fsdp_pkt_hdrs_forward = allocate_and_init_packet_headers(
            fsdp_num_targets_forward > 0, fsdp_unicast_route_info_forward, in1_reader, 1, in1_tile_size);
    } else if (in1_core_order_index == fsdp_forward_in1_core_order_index) {
        fsdp_pkt_hdrs_backward = allocate_and_init_packet_headers(
            fsdp_num_targets_backward > 0, fsdp_unicast_route_info_backward, in1_reader, 1, in1_tile_size);
    }
#endif

    bool k_forward = true;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);

        k_forward = true;

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            // Relay must NOT forward padding N-tiles (n >= N_tiles): the relay dst_tile is
            // k * pwb_N_Wt + n, so a padding column (n >= pwb_N_Wt == N_tiles) wraps into the
            // NEXT K-row's low N-tiles and corrupts them with stale CB data. Padding is always
            // the trailing tiles of the last row's N-stripe, so clamp the relayed count to real N.
            uint32_t relay_N_block_tiles = (n_tile < N_tiles) ? std::min(current_N_block_tiles, N_tiles - n_tile) : 0;

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        cb_wait_front(cb_id_out, out_block_num_tiles);
                        uint32_t out_read_ptr = get_read_ptr(cb_id_out);
                        // write_block_sync_split is more generic (support multiple output tensors)
                        // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync should be faster
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, N_block_tiles>(
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
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        }
                        cb_pop_front(cb_id_out, out_block_num_tiles);
                    }
                }

                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

                uint32_t in1_start_address = get_write_ptr(cb_id_in1);
#ifdef FSDP_FUSED
                // Remembered for fabric send (the NOC-chain block below increments in1_start_address).
                uint32_t in1_send_l1_addr = in1_start_address;
#endif
                // Computed for ALL cores: injector uses for reads, last-two cores use for fabric send.
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
                    // Ring: bidirectional half-block. Requires K_block_tiles | K_tiles_per_device.
                    k_left_tiles = k_block_odd ? (K_block_tiles - (K_block_tiles / 2)) : (K_block_tiles / 2);
                    k_right_tiles = k_block_odd ? (K_block_tiles / 2) : (K_block_tiles - k_left_tiles);
                }
                {
                    compute_actual_k_block<is_linear>(
                        k_block_iter,
                        K_num_blocks,
                        my_rank,
                        K_blocks_per_device,
                        K_block_tiles,
                        K_tiles_per_device,
                        num_devices,
                        k_forward,
#if defined(IS_IN0) || defined(IS_IN1)
                        // in1 reuses each weight stripe across M, so the gather wait fires on the
                        // first m-pass (in0's mirror uses n==0). The skewed sharding makes the
                        // consume order == the uni-ring receive order, so the standard wait gates
                        // correctly with no offset.
                        /*wait_for_forwarded_data=*/(m_block_iter == 0),
                        fsdp_sem_forward_ptr,
                        fsdp_sem_backward_ptr,
                        fsdp_sem_target_forward,
                        fsdp_sem_target_backward,
                        is_injector_core,
                        /*core_order_size=*/1,
#endif
                        k_left_tiles,
                        k_block_left_tile,
                        k_block_right_tile);
                    if constexpr (is_linear) {
                        // For Linear tail, the right half is pure zero-fill padding. Point past the
                        // logical K end so read_in1_block_sync's `i < logical_d0` check triggers
                        // fill_zeros_async for every right-half tile (in1's K dim is d0).
                        if (is_tail_k_block) {
                            k_block_right_tile = K_tiles;
                        }
                    }
                }
                if constexpr (is_injector_core) {
#ifdef FSDP_FUSED
                    // Skewed-shard mirror of in0: the K-block's stripe == my_rank is this device's
                    // local weight slice (device_iter 0) — read from local weight DRAM; any other
                    // stripe is a gathered remote stripe in PWB. The per-stripe arrival wait already
                    // happened inside compute_actual_k_block (Linear wait_for_forwarded_data), so the
                    // PWB data is guaranteed present here. Linear right-half is zero-fill padding
                    // (k_block_right_tile == K_tiles), filled with zeros.
                    const uint32_t local_k_start = my_rank * K_tiles_per_device;
                    const uint32_t local_k_end = local_k_start + K_tiles_per_device;
                    {
                        Noc noc;
                        CircularBuffer cb(cb_id_in1);
                        uint32_t write_ptr = in1_start_address;
                        // Left half (real data: local weight DRAM if this is our stripe, else PWB).
                        for (uint32_t k = 0; k < k_left_tiles; k++) {
                            uint32_t global_k_tile = k_block_left_tile + k;
                            bool local = (global_k_tile >= local_k_start) && (global_k_tile < local_k_end);
                            for (uint32_t n = n_tile; n < n_tile_end; n++) {
                                if (n >= N_tiles) {
                                    write_ptr += in1_tile_size;
                                    continue;
                                }
                                if (local) {
                                    noc_async_read_tile(
                                        (global_k_tile - local_k_start) * N_tiles + n, local_weight_reader, write_ptr);
                                } else {
                                    noc_async_read_tile(global_k_tile * pwb_N_Wt + n, in1_reader, write_ptr);
                                }
                                write_ptr += in1_tile_size;
                            }
                            write_ptr += (N_block_tiles - (n_tile_end - n_tile)) * in1_tile_size;
                        }
                        // Right half (Linear: zero-fill; Ring: real backward data).
                        for (uint32_t k = 0; k < k_right_tiles; k++) {
                            uint32_t global_k_tile = k_block_right_tile + k;
                            bool local = (global_k_tile >= local_k_start) && (global_k_tile < local_k_end);
                            for (uint32_t n = n_tile; n < n_tile_end; n++) {
                                if (n >= N_tiles) {
                                    write_ptr += in1_tile_size;
                                    continue;
                                }
                                if (global_k_tile >= K_tiles) {
                                    fill_zeros_async(noc, cb, in1_tile_size, write_ptr - in1_start_address);
                                } else if (local) {
                                    noc_async_read_tile(
                                        (global_k_tile - local_k_start) * N_tiles + n, local_weight_reader, write_ptr);
                                } else {
                                    noc_async_read_tile(global_k_tile * pwb_N_Wt + n, in1_reader, write_ptr);
                                }
                                write_ptr += in1_tile_size;
                            }
                            write_ptr += (N_block_tiles - (n_tile_end - n_tile)) * in1_tile_size;
                        }
                        noc_async_read_barrier();
                    }
#else
                    read_in1_block_sync<K_block_tiles, N_block_tiles>(
                        in1_reader,
                        in1_shape,
                        cb_id_in1,
                        in1_tile_size,
                        k_block_left_tile,
                        k_block_left_tile + k_left_tiles,
                        k_block_right_tile,
                        k_block_right_tile + k_right_tiles,
                        n_tile,
                        n_tile_end);
#endif
                } else {
                    noc_semaphore_set(in1_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in1_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in1_receiver_semaphore_addr_ptr, VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in1, in1_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in1_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);

                    /**
                     * in1 is K_block_tiles x N_block_tiles. When N block is partial, we don't need to write the
                     * padded tiles. For each tile in the K block, write only the non-padded N tiles. Use
                     * `current_N_tiles_bytes`.
                     */
                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        uint64_t in1_unicast_data_addr = in1_unicast_data_base_addr | in1_start_address;
                        noc_async_write(in1_start_address, in1_unicast_data_addr, current_N_tiles_bytes);
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                }
#ifdef FSDP_FUSED
                // Fabric relay (uni-ring, mirrors in0): on the first m-pass, relay the just-consumed
                // full K-block to the FSDP predecessor's PWB. Skewed (a+b) sharding makes our consume
                // order identical to the predecessor's receive order, so relay-what-you-consume needs
                // no reordering. Dev 0 (fsdp_num_targets_backward == 0) long-sends to Dev N-1 via the
                // forward-routing mux (fsdp_mux_handle_backward, N-1 hops) + fsdp_pkt_hdrs_forward;
                // Dev k short-sends to k-1 via the backward-routing mux (fsdp_mux_handle_forward,
                // 1 hop) + fsdp_pkt_hdrs_backward. Both signal the receiver's forward semaphore. Only
                // the real (left) half is relayed; Linear-tail padding is re-zero-filled by the
                // predecessor's own consume.
                if constexpr (is_linear) {
                    if (m_block_iter == 0) {
                        if constexpr (fsdp_num_targets_backward == 0) {
                            // fsdp Dev 0 (chain head): long send via mux_backward + pkt_hdrs_forward
                            if constexpr (fsdp_num_targets_forward > 0) {
                                if (in1_core_order_index == fsdp_backward_in1_core_order_index &&
                                    k_block_iter < (K_num_blocks - K_blocks_per_device)) {
                                    forward_in1_half_block_to_fabric_neighbor(
                                        k_block_left_tile,
                                        n_tile,
                                        k_left_tiles,
                                        k_right_tiles,
                                        relay_N_block_tiles,
                                        N_block_tiles,
                                        1,
                                        in1_send_l1_addr,
                                        pwb_N_Wt,
                                        in1_reader,
                                        fsdp_mux_handle_backward,
                                        fsdp_pkt_hdrs_forward,
                                        in1_tile_size,
                                        fsdp_sem_noc_addr_forward_in_pkt,
                                        /*write_left_half=*/true,
                                        /*do_write=*/true);
                                }
                            }
                        } else {
                            // fsdp Dev k > 0: short send via mux_forward + pkt_hdrs_backward
                            if (in1_core_order_index == fsdp_forward_in1_core_order_index &&
                                k_block_iter < (K_num_blocks - K_blocks_per_device)) {
                                forward_in1_half_block_to_fabric_neighbor(
                                    k_block_left_tile,
                                    n_tile,
                                    k_left_tiles,
                                    k_right_tiles,
                                    relay_N_block_tiles,
                                    N_block_tiles,
                                    1,
                                    in1_send_l1_addr,
                                    pwb_N_Wt,
                                    in1_reader,
                                    fsdp_mux_handle_forward,
                                    fsdp_pkt_hdrs_backward,
                                    in1_tile_size,
                                    fsdp_sem_noc_addr_forward_in_pkt,
                                    /*write_left_half=*/true,
                                    /*do_write=*/true);
                            }
                        }
                    }
                }
#endif
            }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_reserve_back(cb_id_in2, N_block_tiles);

                uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc_async_read_page(n_tile_id, in2_reader, l1_write_addr_in2);
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id_in2, N_block_tiles);
            }
#endif

#ifdef FUSE_TERNARY
            if constexpr (!is_output_writer) {
                read_ternary_blocks_sync<M_block_tiles, N_block_tiles>(
                    ternary_a_reader,
                    ternary_b_reader,
                    out_shape,
                    cb_id_ternary_a,
                    cb_id_ternary_b,
                    ternary_a_tile_size,
                    ternary_b_tile_size,
                    broadcast_ternary_b,
                    m_tile,
                    m_tile_end,
                    n_tile,
                    n_tile_end);
            }
#endif  // FUSE_TERNARY

            if constexpr (!is_linear) {
                k_forward = !k_forward;
            }
            // We have an output block to write out

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
                    // write_block_sync_granular_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();

#ifdef FSDP_FUSED
    if (fsdp_mux_backward.connection_valid) {
        close_mux(
            fsdp_mux_handle_backward,
            fsdp_mux_backward.is_termination_master,
            fsdp_mux_backward.termination_sync_address,
            fsdp_mux_backward.num_mux_clients,
            fsdp_mux_backward.fabric_mux_x,
            fsdp_mux_backward.fabric_mux_y,
            fabric_mux_termination_signal_address,
            fsdp_mux_backward.termination_master_noc_x,
            fsdp_mux_backward.termination_master_noc_y);
    }
    if (fsdp_mux_forward.connection_valid) {
        close_mux(
            fsdp_mux_handle_forward,
            fsdp_mux_forward.is_termination_master,
            fsdp_mux_forward.termination_sync_address,
            fsdp_mux_forward.num_mux_clients,
            fsdp_mux_forward.fabric_mux_x,
            fsdp_mux_forward.fabric_mux_y,
            fabric_mux_termination_signal_address,
            fsdp_mux_forward.termination_master_noc_x,
            fsdp_mux_forward.termination_master_noc_y);
    }

    noc_async_write_barrier();

    // Reset FSDP semaphores so the next op-launch starts clean (matches the in0 pattern).
    noc_semaphore_set(fsdp_sem_forward_ptr, 0);
    noc_semaphore_set(fsdp_sem_backward_ptr, 0);
#endif
}
