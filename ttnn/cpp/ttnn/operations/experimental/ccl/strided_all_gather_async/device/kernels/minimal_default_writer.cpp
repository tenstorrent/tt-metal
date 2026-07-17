// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#if defined(USE_MUX_V2)
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#endif
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/strided_all_gather_common.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t max_tiles_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr bool fuse_op = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);  // 1 is forward, 0 is backward
constexpr uint32_t ag_worker_cores = get_compile_time_arg_val(9);
constexpr uint32_t ag_worker_id = get_compile_time_arg_val(10);
constexpr bool is_termination_master = get_compile_time_arg_val(11);
constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(12);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(13);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(14);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(15);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(16);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(21);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(22);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(23);

constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<24>();

inline constexpr uint32_t sharded_args_start_idx = 24 + ccl_routing_utils::num_line_unicast_args;

// Streaming matmul signal: each chunk's M-rows are delivered as this many row-bands, with one
// matmul-aggregator inc per band. Default 1 = single inc per chunk (legacy). Must match the value
// the matmul in0 kernel and the aggregator are built with.
#ifndef IN0_SUB_CHUNKS
#define IN0_SUB_CHUNKS 1
#endif

// Mirrors the out_ready_sem gate inside write_chunk(): this writer forwards a chunk over fabric to a
// remote device only when its direction has a target. When true, the writer also owns signaling the
// remote device's matmul (semaphores[dir]) for each chunk it delivers (see writer_signals_mm).
inline constexpr bool has_remote_target =
    (direction == 1 && num_targets_backward_direction) || (direction == 0 && num_targets_forward_direction);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_batches = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_worker_tile_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    const uint8_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    uint32_t mm_block_wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_block_ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_cores_y = get_arg_val<uint32_t>(arg_idx++);
    bool read_local_slice_from_input = (bool)get_arg_val<uint32_t>(arg_idx++);

    uint32_t device_k_block_counts[ring_size];
    uint32_t device_max_chunks = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_chunk_widths[ring_size][device_max_chunks];
    for (uint32_t d = 0; d < ring_size; d++) {
        device_k_block_counts[d] = get_arg_val<uint32_t>(arg_idx++);
        uint32_t device_chunk_count = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t c = 0; c < device_chunk_count; c++) {
            device_chunk_widths[d][c] = get_arg_val<uint32_t>(arg_idx++);
        }
    }

    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
#if defined(USE_MUX_V2)
    // V2 is fully runtime-arg driven; build_from_args consumes the 11 client-connection args the
    // host appends via FabricMuxV2Config::append_client_connection_rt_args. No worker-side
    // termination protocol: the mux self-drains once every client closes.
    // build_from_args takes size_t&; bridge through a size_t since arg_idx is uint32_t here.
#define EAGER_STAGING false
    size_t mux_arg_idx = arg_idx;
    auto mux_connection = tt::tt_fabric::FabricMuxV2Sender<EAGER_STAGING>::build_from_args(mux_arg_idx);
    arg_idx = static_cast<uint32_t>(mux_arg_idx);
#else
    uint32_t termination_sync_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_sync_address = get_semaphore(termination_sync_id);
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mux_clients = get_arg_val<uint32_t>(arg_idx++);
#endif

    constexpr auto output_tensor_args = TensorAccessorArgs<sharded_args_start_idx>();
    const auto output_addrgen = TensorAccessor(output_tensor_args, output_address);

    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;
    bool writer_signals_mm = false;
    uint32_t agg_core_noc_x = 0;
    uint32_t agg_core_noc_y = 0;
    uint32_t agg_per_worker_sem_addr = 0;
    if constexpr (fuse_op) {
        op_signaler_sender = OpSignaler(arg_idx);
        writer_signals_mm = get_arg_val<uint32_t>(arg_idx++) == 1;
        // Option W: this worker signals its own per-worker semaphore on the direction's aggregation
        // core (a single remote core), which collects all N workers and signals the matmul once.
        agg_core_noc_x = get_arg_val<uint32_t>(arg_idx++);
        agg_core_noc_y = get_arg_val<uint32_t>(arg_idx++);
        // GlobalSemaphore L1 address (already resolved host-side); do NOT get_semaphore() it.
        agg_per_worker_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    }

#if !defined(USE_MUX_V2)
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
    }
#endif

    // pre-populate packet headers
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

    if (mux_connection_valid) {
#if defined(USE_MUX_V2)
        // open() gates on the mux reporting READY_FOR_TRAFFIC internally, then opens the connection.
        mux_connection.open();
#else
        tt::tt_fabric::fabric_client_connect(*mux_connection_handle);
#endif
    }

    auto page_size = tt::tt_fabric::linear::addrgen_detail::get_page_size(output_addrgen);
    // Template holds the max-arity payload/chunk layout; each write overrides DstAddrs plus the actual
    // ChunkSizes/PayloadSize, since a packet may carry fewer tiles than max_tiles_per_packet (tail/holes).
    uint64_t scatter_dummy_addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
    uint16_t scatter_init_chunk_sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {
        static_cast<uint16_t>(page_size), static_cast<uint16_t>(page_size), static_cast<uint16_t>(page_size)};
    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        pkt_scatter_hdr,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        NocUnicastScatterCommandHeader(scatter_dummy_addrs, scatter_init_chunk_sizes, max_tiles_per_packet),
        page_size * max_tiles_per_packet);

    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, output_page_size);

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_sem_inc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint16_t>(1)});  // increment 1

    ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);

    // Dedicated header for signaling the remote device's matmul directly (writer_signals_mm). Only
    // allocated when enabled, to avoid consuming a packet-header pool slot on the default path.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_mm_sem_inc = nullptr;
    if (fuse_op && writer_signals_mm && has_remote_target) {
        pkt_hdr_mm_sem_inc = PacketHeaderPool::allocate_header();
        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_mm_sem_inc,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0,                           // ignore (set per-core below)
                static_cast<uint16_t>(1)});  // increment 1
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_mm_sem_inc, unicast_route_info);
    }
    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    uint32_t writes_expected = 0;
    if (topology == Topology::Linear) {
        if (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if (topology == Topology::Ring) {
        if (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    Noc noc_obj;

    // On an even ring the diametric device's slice is the last one relayed and would otherwise
    // traverse the full half-ring on one link while the other idles. Relay its first half on
    // direction 0 and its second half on direction 1 so both links share the final hop; direction 1
    // gains one relay to carry that second half. Gated to the reader-signaled path: the
    // writer-signals aggregator expects whole per-sender slices per direction.
    bool split_forwarding_enabled = (topology == Topology::Ring) && (ring_size % 2 == 0) && (ring_size > 2);
    if (split_forwarding_enabled && direction == 1) {
        writes_expected++;
    }

    uint32_t batch_output_tile_offset = output_worker_tile_offset;
    uint32_t global_tile_index = 0;
    uint32_t output_tiles_per_batch = output_tensor_Wt * output_tensor_Ht;

    uint32_t padded_M_tiles = round_up(input_tensor_Ht, mm_cores_y);
    uint32_t M_tiles_per_core = padded_M_tiles / mm_cores_y;
    uint32_t M_blocks_per_core = div_up(M_tiles_per_core, mm_block_ht);

    // Write out the local slice to both DRAM and forward and backward
    for (uint32_t b_idx = 0; b_idx < num_batches; b_idx++) {
        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            // Send out local
            uint32_t input_chunk_start_tile = global_tile_index;
            for (uint32_t chunk_idx = 0; chunk_idx < device_k_block_counts[my_chip_id]; chunk_idx++) {
                DeviceZoneScopedN("AG-LOCAL-SEND");
                uint32_t actual_chunk_w = device_chunk_widths[my_chip_id][chunk_idx];
                uint32_t actual_chunk_h = next_mm_aligned_chunk_height(
                    input_chunk_start_tile, M_tiles_per_core, input_tensor_Wt, mm_block_ht);
                uint32_t tiles_in_current_chunk = actual_chunk_w * actual_chunk_h * mm_cores_y;
                // write_chunk fires the per-worker aggregator inc once per row-band (IN0_SUB_CHUNKS
                // total per chunk), replacing the single post-chunk inc that used to live here.
                bool mm_sig_local = fuse_op && writer_signals_mm && has_remote_target;
                uint64_t agg_sem_noc_addr_local =
                    mm_sig_local ? safe_get_noc_addr(
                                       (uint8_t)agg_core_noc_x, (uint8_t)agg_core_noc_y, agg_per_worker_sem_addr, 0)
                                 : 0;
                write_chunk(
                    input_chunk_start_tile,
                    batch_output_tile_offset,
                    cb_output_id,
                    tiles_in_current_chunk,
                    actual_chunk_w,
                    actual_chunk_h,
                    padded_M_tiles / mm_cores_y,
                    max_tiles_per_packet,
                    ag_worker_id,
                    ag_worker_cores,
                    output_addrgen,
                    output_page_size,
                    input_tensor_Wt,
                    input_tensor_Ht,
                    output_tensor_Wt,
                    my_chip_id,
                    mux_connection,
                    pkt_scatter_hdr,
                    pkt_unicast_hdr,
                    pkt_hdr_sem_inc,
                    out_ready_sem_noc_addr_in_pkt,
                    direction,
                    num_targets_forward_direction,
                    num_targets_backward_direction,
                    true && !read_local_slice_from_input,
                    mm_cores_y,
                    IN0_SUB_CHUNKS,
                    mm_sig_local,
                    pkt_hdr_mm_sem_inc,
                    agg_sem_noc_addr_local);
                if (fuse_op && direction == 1 && !read_local_slice_from_input) {
                    DeviceZoneScopedN("AG-MM-SIGNAL");
                    // Synchronize and signal that the local tensor slice is available
                    op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
                }
            }

            // Forward chunks
            uint32_t slice_writes = 0;
            while (slice_writes < writes_expected) {
                uint32_t actual_sender_chip_id = get_sender_id(direction, my_chip_id, slice_writes, ring_size);
                uint32_t num_chunks = device_k_block_counts[actual_sender_chip_id];
                bool is_split_forwarded_slice =
                    split_forwarding_enabled && (slice_writes == writes_expected - 1) && (num_chunks >= 2);
                uint32_t first_half_chunks = num_chunks / 2;
                input_chunk_start_tile = global_tile_index;
                for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                    uint32_t actual_chunk_w = device_chunk_widths[actual_sender_chip_id][chunk_idx];
                    uint32_t actual_chunk_h = next_mm_aligned_chunk_height(
                        input_chunk_start_tile, M_tiles_per_core, input_tensor_Wt, mm_block_ht);

                    // Direction 0 relays the first half of a split slice, direction 1 the second;
                    // each walks past the half it does not relay.
                    bool relay_this_chunk =
                        !is_split_forwarded_slice ||
                        (direction == 0 ? (chunk_idx < first_half_chunks) : (chunk_idx >= first_half_chunks));
                    if (!relay_this_chunk) {
                        advance_chunk_start_tile(
                            input_chunk_start_tile, actual_chunk_w, actual_chunk_h, input_tensor_Wt);
                        continue;
                    }

                    DeviceZoneScopedN("AG-FWD-SEND");
                    uint32_t tiles_in_current_chunk = actual_chunk_w * actual_chunk_h * mm_cores_y;

                    // Same per-band aggregator inc as the local send, fired inside write_chunk.
                    bool mm_sig_fwd = fuse_op && writer_signals_mm && has_remote_target;
                    uint64_t agg_sem_noc_addr_fwd =
                        mm_sig_fwd ? safe_get_noc_addr(
                                         (uint8_t)agg_core_noc_x, (uint8_t)agg_core_noc_y, agg_per_worker_sem_addr, 0)
                                   : 0;
                    write_chunk(
                        input_chunk_start_tile,
                        batch_output_tile_offset,
                        cb_output_id,
                        tiles_in_current_chunk,
                        actual_chunk_w,
                        actual_chunk_h,
                        padded_M_tiles / mm_cores_y,
                        max_tiles_per_packet,
                        ag_worker_id,
                        ag_worker_cores,
                        output_addrgen,
                        output_page_size,
                        input_tensor_Wt,
                        input_tensor_Ht,
                        output_tensor_Wt,
                        actual_sender_chip_id,
                        mux_connection,
                        pkt_scatter_hdr,
                        pkt_unicast_hdr,
                        pkt_hdr_sem_inc,
                        out_ready_sem_noc_addr_in_pkt,
                        direction,
                        num_targets_forward_direction,
                        num_targets_backward_direction,
                        false,
                        mm_cores_y,
                        IN0_SUB_CHUNKS,
                        mm_sig_fwd,
                        pkt_hdr_mm_sem_inc,
                        agg_sem_noc_addr_fwd);
                }
                slice_writes++;
            }
            global_tile_index = input_chunk_start_tile;
        }
        batch_output_tile_offset += output_tiles_per_batch;
    }

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();

    if (mux_connection_valid) {
#if defined(USE_MUX_V2)
        // close() requests teardown and blocks on the manager's ack. The mux self-terminates once
        // all clients have closed, so there is no termination-master election on the worker side.
        mux_connection.close();
#else
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

        if constexpr (is_termination_master) {
            Semaphore<> termination_sync(termination_sync_id);
            termination_sync.wait(num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            uint64_t dest_addr =
                safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
            noc_semaphore_inc(dest_addr, 1);
            noc_obj.async_atomic_barrier();
        }
#endif
    }
    noc_obj.async_write_barrier();
}
