// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

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
uint32_t in0_sender_backward_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
uint32_t in0_sender_forward_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
uint32_t in0_receiver_backward_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
uint32_t in0_receiver_forward_semaphore_addr = get_semaphore(get_compile_time_arg_val(17));
uint32_t in0_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(18));
constexpr uint32_t is_output_writer = get_compile_time_arg_val(19);
constexpr uint32_t is_injector_core_backward = get_compile_time_arg_val(20);
constexpr uint32_t is_injector_core_forward = get_compile_time_arg_val(21);
constexpr uint32_t num_devices = get_compile_time_arg_val(22);
constexpr uint32_t my_rank = get_compile_time_arg_val(23);
constexpr uint32_t in3_tile_size = get_compile_time_arg_val(24);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(25);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(26);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(27);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(28));
 
#ifdef USE_MUX
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(29);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(30);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(31);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(32);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(33);

constexpr uint32_t mux_arg_count = 34;

constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
  ccl_routing_utils::get_line_unicast_route_info_from_args<mux_arg_count>();

constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
  ccl_routing_utils::get_line_unicast_route_info_from_args<
  mux_arg_count + ccl_routing_utils::num_line_unicast_args>();

constexpr uint32_t ct_arg_count = mux_arg_count + 2 * ccl_routing_utils::num_line_unicast_args;

const auto& unicast_route_info =
  is_injector_core_backward ? forward_unicast_route_info : backward_unicast_route_info;
#else
constexpr uint32_t ct_arg_count = 29;
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
    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in3_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_backward_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_backward_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_forward_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_forward_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_backward_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_backward_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_forward_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_forward_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(argidx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(argidx++);

#ifdef USE_MUX
    bool mux_connection_valid = get_arg_val<uint32_t>(argidx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(argidx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(argidx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(argidx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(argidx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(argidx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(argidx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(argidx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(argidx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(argidx++);

    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(argidx++));

    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(argidx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(argidx++);
#endif

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<ct_arg_count>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, in0_tile_size);
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out_reader = TensorAccessor(out_args, out_addr, out_tile_size);
#ifdef FUSE_BIAS
    constexpr auto in2_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr, in2_tile_size);
#endif

#ifdef USE_MUX
    // Setup mux connection
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle;
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
        mux_connection_handle = &mux_connection;
    } else {
        mux_connection_handle = nullptr;
    }

    if (mux_connection_valid) {
        // need to wait for fabric mux to be ready to accept connections
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
        tt::tt_fabric::fabric_client_connect(*mux_connection_handle);
    }
#endif

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_4;
#endif

#ifdef READ_FROM_LOCAL_INPUT
#ifdef FUSE_BIAS
    constexpr auto in3_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
#else
    constexpr auto in3_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif
    const auto in3_reader = TensorAccessor(in3_args, in3_addr, in3_tile_size);
#endif

    volatile tt_l1_ptr uint32_t* in0_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_valid_semaphore_addr);
    *(in0_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in0_receiver_backward_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_backward_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in0_receiver_forward_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_forward_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in0_sender_backward_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_backward_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in0_sender_forward_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_forward_semaphore_addr);
    const uint64_t in0_sender_backward_semaphore_noc_addr =
        get_noc_addr(in0_sender_backward_noc_x, in0_sender_backward_noc_y, in0_sender_backward_semaphore_addr);
    const uint64_t in0_sender_forward_semaphore_noc_addr =
        get_noc_addr(in0_sender_forward_noc_x, in0_sender_forward_noc_y, in0_sender_forward_semaphore_addr);

    const uint64_t in0_receiver_backward_semaphore_noc_addr =
        get_noc_addr(in0_dest_backward_noc_x, in0_dest_backward_noc_y, in0_receiver_backward_semaphore_addr);
    const uint64_t in0_receiver_forward_semaphore_noc_addr =
        get_noc_addr(in0_dest_forward_noc_x, in0_dest_forward_noc_y, in0_receiver_forward_semaphore_addr);

#ifdef USE_MUX
    // all gather
    volatile tt_l1_ptr uint32_t* out_ready_sem_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem);
    uint32_t sem_target = 0;
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    uint32_t slices_expected = 0;
    uint32_t writes_expected = 0;
    if (topology == Topology::Linear) {
        if (is_injector_core_forward) {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_backward_direction ? num_targets_forward_direction : 0;
        } else {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_forward_direction ? num_targets_backward_direction : 0;
        }
    } else if (topology == Topology::Ring) {
        if (is_injector_core_forward) {
            slices_expected = num_targets_backward_direction;
        } else {
            slices_expected = num_targets_forward_direction;
        }
        writes_expected = slices_expected - 1;
    }

    // pre-populate packet headers
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();
    // only initialize if we're actually going to send something over fabric
    if (detail::valid_targets(is_injector_core_forward)) {
        static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported");
        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
        uint16_t chunk_sizes[3] = {out_tile_size, out_tile_size, out_tile_size};
        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, num_tiles_to_write_per_packet),
            in3_tile_size * num_tiles_to_write_per_packet);

        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, in3_tile_size);

        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_sem_inc,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0,  // ignore
                static_cast<uint32_t>(1)});
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
    }
#endif

    /**
     * This is a Serpentine (Boustrophedon) output block ordering.
     * It enables reuse of one of the input blocks for the last output block.
     * Starting at output block (0,0), go east until the end, then south one block, then west until the end, then south
     * one block, and repeat. At the same time, alternate between K striding forwards or backwards in order to enable
     * reuse.
     */

    bool k_forward = true;
    bool n_forward = true;
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
        uint32_t slices_received = 0;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = n_forward ? N_start_tile + n_block_iter * N_block_tiles
                                        : N_start_tile + (N_blocks_per_core - 1 - n_block_iter) * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        cb_wait_front(cb_id_out, out_block_num_tiles);
                        uint32_t out_read_ptr = get_read_ptr(cb_id_out);
                        write_block_sync<M_block_tiles, N_block_tiles>(
                            out_reader,
                            out_shape,
                            out_read_ptr,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile,
                            defer_write_n_tile_end);
                        cb_pop_front(cb_id_out, out_block_num_tiles);
                    }
                }
                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
                    continue;
                }
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);

                bool use_backward_injector_core = is_backward_k_block_iter(k_block_iter, K_num_blocks / num_devices);
                bool is_injector_core =
                    is_injector(use_backward_injector_core, is_injector_core_backward, is_injector_core_forward);
                uint32_t in0_start_address = get_write_ptr(cb_id_in0);

                uint32_t k_block = 0;
#ifdef USE_MUX
                k_block = compute_actual_k_block(
                    k_block_iter,
                    K_num_blocks,
                    my_rank,
                    K_num_blocks / num_devices,
                    num_devices,
                    k_forward,
                    n_block_iter == 0,
                    out_ready_sem_addr_ptr,
                    sem_target,
                    slices_received);
#endif
                if (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        in0_start_address,
                        in0_tile_size,
#ifdef READ_FROM_LOCAL_INPUT
                        in3_reader,
                        my_rank * (padded_K_tiles / num_devices),
                        (my_rank + 1) * (padded_K_tiles / num_devices) - 1,
                        padded_K_tiles / num_devices,
#endif
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                    // Get from previous device
                    noc_semaphore_set(
                        use_backward_injector_core ? in0_receiver_backward_semaphore_addr_ptr
                                                   : in0_receiver_forward_semaphore_addr_ptr,
                        INVALID);
                    noc_semaphore_inc(
                        use_backward_injector_core ? in0_sender_backward_semaphore_noc_addr
                                                   : in0_sender_forward_semaphore_noc_addr,
                        1);
                    noc_semaphore_wait(
                        use_backward_injector_core ? in0_receiver_backward_semaphore_addr_ptr
                                                   : in0_receiver_forward_semaphore_addr_ptr,
                        VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in0, in0_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(
                        use_backward_injector_core ? in0_sender_backward_semaphore_addr_ptr
                                                   : in0_sender_forward_semaphore_addr_ptr,
                        1);
                    noc_semaphore_set(
                        use_backward_injector_core ? in0_sender_backward_semaphore_addr_ptr
                                                   : in0_sender_forward_semaphore_addr_ptr,
                        0);

                    uint64_t in0_unicast_data_addr =
                        use_backward_injector_core
                            ? get_noc_addr(in0_dest_backward_noc_x, in0_dest_backward_noc_y, in0_start_address)
                            : get_noc_addr(in0_dest_forward_noc_x, in0_dest_forward_noc_y, in0_start_address);

                    /**
                     * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                     * padded tiles. Use `current_block_bytes`.
                     */
                    noc_async_write(in0_start_address, in0_unicast_data_addr, current_block_bytes);

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(
                        in0_valid_semaphore_addr,
                        use_backward_injector_core ? in0_receiver_backward_semaphore_noc_addr
                                                   : in0_receiver_forward_semaphore_noc_addr);
                }

#ifdef USE_MUX
                if (n_block_iter == 0) {
                    // If not the last block
                    if (slices_received <= writes_expected) {
                        // If backward, send forward
                        // If forward, send backward
                        forward_tile_to_fabric_neighbor(
                            m_tile,
                            k_block * K_block_tiles,
                            current_M_block_tiles,
                            K_block_tiles,
                            num_tiles_to_write_per_packet,
                            in0_start_address,
                            padded_K_tiles,
                            in0_reader,
                            mux_connection_handle,
                            pkt_scatter_hdr,
                            pkt_unicast_hdr,
                            pkt_hdr_sem_inc,
                            in0_tile_size,
                            out_ready_sem_noc_addr_in_pkt);
                    }
                }
#endif
            }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_reserve_back(cb_id_in2, N_block_tiles);

                uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc_async_read_tile(n_tile_id, in2_reader, l1_write_addr_in2);
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id_in2, N_block_tiles);
            }
#endif

            k_forward = !k_forward;
            // We get reuse on in0 when striding N block
            reuse_block = true;

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !((m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1)));
            defer_write = defer_write && !(is_injector_core_backward || is_injector_core_forward);

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    write_block_sync_granular<M_block_tiles, N_block_tiles>(
                        out_reader, out_shape, cb_id_out, out_tile_size, m_tile, m_tile_end, n_tile, n_tile_end);
                }
            }
        }
        n_forward = !n_forward;
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();

#ifdef USE_MUX
    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

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
    }

    noc_async_write_barrier();

    noc_semaphore_set(out_ready_sem_addr_ptr, 0);
#endif
}
