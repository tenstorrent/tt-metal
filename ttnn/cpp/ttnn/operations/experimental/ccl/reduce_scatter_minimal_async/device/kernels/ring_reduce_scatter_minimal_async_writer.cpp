// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_named_compile_time_arg_val("my_chip_id");
constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
constexpr uint32_t cb_compute_output_id = get_named_compile_time_arg_val("cb_compute_output_id");
constexpr uint32_t cb_reader_output_id = get_named_compile_time_arg_val("cb_reader_output_id");
constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
constexpr uint32_t num_tiles_to_write_per_packet = get_named_compile_time_arg_val("num_tiles_to_write_per_packet");
constexpr uint32_t output_batch_num_pages = get_named_compile_time_arg_val("output_batch_num_pages");
constexpr uint32_t input_channel_num_pages = get_named_compile_time_arg_val("input_channel_num_pages");
constexpr uint32_t output_channel_num_pages = get_named_compile_time_arg_val("output_channel_num_pages");
constexpr uint32_t input_tensor_B = get_named_compile_time_arg_val("input_tensor_B");
constexpr uint32_t input_tensor_Wt = get_named_compile_time_arg_val("input_tensor_Wt");
constexpr uint32_t slice_C = get_named_compile_time_arg_val("slice_C");
constexpr uint32_t slice_Ht = get_named_compile_time_arg_val("slice_Ht");
constexpr uint32_t slice_Wt = get_named_compile_time_arg_val("slice_Wt");
constexpr uint32_t dim = get_named_compile_time_arg_val("dim");
#ifdef USE_WORKER_MUX
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(0);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(1);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(2);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(3);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(4);

constexpr uint32_t num_ct_args = 5;
#else
constexpr uint32_t num_ct_args = 0;
#endif

// Routing info uses positional args after fabric mux args
constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<num_ct_args>();
constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        num_ct_args + ccl_routing_utils::num_line_unicast_args>();

constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<
        num_ct_args + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        num_ct_args + 2 * ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t interm_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t this_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t this_core_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 1 is forward, 0 is backward
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
#ifdef USE_WORKER_MUX
    const bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
#endif

    const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

    constexpr auto interm_tensor_args = TensorAccessorArgs<ct_idx>();
    auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);

    constexpr auto output_tensor_args = TensorAccessorArgs<interm_tensor_args.next_compile_time_args_offset()>();
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

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

    // need to wait for fabric mux to be ready to accept connections
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
#else
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
#endif
    // pre-populate packet headers
    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();
    auto pkt_hdr_mcastseminc = PacketHeaderPool::allocate_header();
    // Fused write + atomic-inc headers, used to fold a chunk's semaphore increment into its final
    // data packet (unicast for a 1-tile tail, scatter for a 2-tile tail).
    auto pkt_hdr_fused_unicast = PacketHeaderPool::allocate_header();
    auto pkt_hdr_fused_scatter = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fused_unicast, unicast_route_info);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fused_scatter, unicast_route_info);

#ifdef USE_WORKER_MUX
    tt::tt_fabric::fabric_client_connect(mux_connection_handle);
    auto* fabric_direction_connection = &mux_connection_handle;
#else
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    auto* fabric_direction_connection =
        direction ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection();
#endif
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_mcastseminc,
        static_cast<uint8_t>(multicast_route_info.start_distance_in_hops),
        static_cast<uint8_t>(multicast_route_info.range_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1
    if (use_barrier_sem) {
        // multicast to entire ring of workers for both this dir and opposite dir
        ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_mcastseminc, multicast_route_info);

        uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(this_core_x, this_core_y, barrier_sem, 0);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_mcastseminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

        barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(opposite_core_x, opposite_core_y, barrier_sem, 0);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_mcastseminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 2 * (ring_size - 1));
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported");
    uint64_t remote_noc_addrs[4] = {0, 0, 0, 0};
    uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        pkt_scatter_hdr,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        NocUnicastScatterCommandHeader(remote_noc_addrs, chunk_sizes, num_tiles_to_write_per_packet),
        page_size * num_tiles_to_write_per_packet);

    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        pkt_hdr_seminc,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                           // ignore
            static_cast<uint32_t>(1)});  // increment 1

    // Fused-packet state: payload size, increment value (1) and flush are constant across the run;
    // only the write and semaphore destination addresses are patched per packet via with_state.
    fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
        UnicastFusedAtomicIncUpdateMask::PayloadSize | UnicastFusedAtomicIncUpdateMask::Val |
        UnicastFusedAtomicIncUpdateMask::Flush>(
        pkt_hdr_fused_unicast,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
            0,                          // write dst (patched per packet)
            0,                          // semaphore dst (patched per packet)
            static_cast<uint32_t>(1)},  // increment 1
        page_size);

    fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state<
        UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize |
        UnicastFusedScatterWriteAtomicIncUpdateMask::WriteChunkSizes |
        UnicastFusedScatterWriteAtomicIncUpdateMask::Val | UnicastFusedScatterWriteAtomicIncUpdateMask::Flush>(
        pkt_hdr_fused_scatter,
        static_cast<uint8_t>(unicast_route_info.distance_in_hops),
        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
            {0, 0},                              // write dsts (patched per packet)
            0,                                   // semaphore dst (patched per packet)
            {static_cast<uint16_t>(page_size)},  // first chunk size (second is implicit)
            static_cast<uint16_t>(1)},           // increment 1
        static_cast<uint16_t>(page_size * 2));

    // Relevant for 2nd-last iter:
    // In 2nd-last iter we send the full tensor slice. But in preparation for the last iter where each dir
    // processes half tensor slice, in 2nd-last iter we send sem increments to both forward and backward workers.
    // For example, if we send 2 even chunks and 2 odd chunks, we need to send 2 sem incrs to forward worker
    // and 2 sem incrs to backward worker.
    uint64_t this_core_sem_noc_addr = safe_get_noc_addr(this_core_x, this_core_y, out_ready_sem, 0);
    uint64_t opposite_core_sem_noc_addr = safe_get_noc_addr(opposite_core_x, opposite_core_y, out_ready_sem, 0);
    uint64_t even_core_sem_noc_addr = direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;
    uint64_t odd_core_sem_noc_addr = !direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;

    // ---- Fabric send helpers (capture the per-direction connection and pre-configured headers) ----

    // Emit a standalone atomic increment to a remote worker's out_ready_sem.
    auto send_seminc = [&](uint64_t sem_noc_addr) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_direction_connection,
            pkt_hdr_seminc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0});
        noc_async_writes_flushed();
    };

    // Write one packet worth of tiles (addresses staged in remote_noc_addrs) to the remote tensor.
    // When fuse_seminc is set, this chunk's semaphore increment is folded onto the packet and the
    // function returns true. The fused fabric ops carry at most a 2-tile scatter write plus the
    // semaphore chunk, so packets wider than 2 tiles cannot fuse (caller falls back to send_seminc).
    auto send_write_packet =
        [&](size_t l1_read_addr, uint32_t num_tiles, bool fuse_seminc, uint64_t sem_noc_addr) -> bool {
        if (fuse_seminc && num_tiles <= 2) {
            if (num_tiles == 2) {
                fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state<
                    UnicastFusedScatterWriteAtomicIncUpdateMask::WriteDstAddrs |
                    UnicastFusedScatterWriteAtomicIncUpdateMask::SemaphoreDstAddr>(
                    fabric_direction_connection,
                    pkt_hdr_fused_scatter,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                        {remote_noc_addrs[0], remote_noc_addrs[1]},
                        sem_noc_addr,
                        {static_cast<uint16_t>(page_size)},
                        static_cast<uint16_t>(1)});
            } else {
                fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                    UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                    fabric_direction_connection,
                    pkt_hdr_fused_unicast,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        remote_noc_addrs[0], sem_noc_addr, static_cast<uint32_t>(1)});
            }
            return true;
        }
        if (num_tiles > 1) {
            fabric_unicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_direction_connection,
                pkt_scatter_hdr,
                l1_read_addr,
                NocUnicastScatterCommandHeader(remote_noc_addrs, chunk_sizes, num_tiles),
                page_size * num_tiles);
        } else {
            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_unicast_hdr,
                l1_read_addr,
                NocUnicastCommandHeader{remote_noc_addrs[0]});
        }
        return false;
    };

    for (uint32_t b = 0; b < input_tensor_B; ++b) {
        constexpr uint32_t ring_size_by_2 = ring_size / 2;
        int slice_idx = my_chip_id + ring_size_by_2;  // start with slice belonging to device half-way across in ring
        uint32_t num_iters = ring_size_by_2 + 1;
        for (uint32_t i = 0; i < num_iters; ++i) {
            // State machine for control variables
            bool even_chunks, odd_chunks, reduce_even_chunks, reduce_odd_chunks, write_to_remote, write_to_interm,
                separate_even_odd_sems;
            if (i == 0) {
                even_chunks = direction;     // process the even chunks (half the tensor slice)
                odd_chunks = !direction;     // process the odd chunks (other half of tensor slice)
                reduce_even_chunks = false;  // grab output from compute or reader
                reduce_odd_chunks = false;   // grab output from compute or reader
                write_to_remote = true;      // write to remote device or local device
                write_to_interm = true;      // write to interm_tensor or output_tensor
                separate_even_odd_sems =
                    false;  // 2nd-last iter: send sem incrs separately for even & odd chunks to diff workers
            } else if (i == ring_size_by_2) {
                even_chunks = direction;
                odd_chunks = !direction;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                write_to_remote = false;
                write_to_interm = false;
                separate_even_odd_sems = false;
            } else if (i == 1 || i == ring_size_by_2 - 1) {  // these two cases can coincide (ring_size = 4)
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = (i == 1) ? direction : even_chunks;
                reduce_odd_chunks = (i == 1) ? !direction : odd_chunks;
                write_to_remote = true;
                write_to_interm = (i == ring_size_by_2 - 1) ? direction : true;
                separate_even_odd_sems = (i == ring_size_by_2 - 1);
            } else {
                even_chunks = true;
                odd_chunks = true;
                reduce_even_chunks = even_chunks;
                reduce_odd_chunks = odd_chunks;
                write_to_remote = true;
                write_to_interm = true;
                separate_even_odd_sems = false;
            }

            // below code does 'slice_idx = slice_idx % ring_size'
            if (slice_idx < 0) {
                slice_idx += ring_size;
            } else if (slice_idx >= (int)ring_size) {
                slice_idx = (uint32_t)slice_idx - ring_size;
            }

            // address incrementer for interm_tensor
            uint32_t interm_tile_id_start;
            if constexpr (dim == 3) {
                interm_tile_id_start = slice_idx * slice_Wt;
            } else if constexpr (dim == 2) {
                interm_tile_id_start = slice_idx * slice_Ht * slice_Wt;
            } else if constexpr (dim == 1) {
                interm_tile_id_start = slice_idx * slice_C * slice_Ht * slice_Wt;
            } else {
                ASSERT(false);
            }
            uint32_t interm_pages_read_in_row = start_pages_read_in_row;
            uint32_t interm_row_offset = start_row_offset;
            auto get_next_interm_tile_id = [&]() -> uint32_t {
                uint32_t tile_id = interm_tile_id_start + interm_row_offset + interm_pages_read_in_row;
                ++interm_pages_read_in_row;
                if (interm_pages_read_in_row == slice_Wt) {
                    interm_row_offset += input_tensor_Wt;
                    interm_pages_read_in_row -= slice_Wt;
                }
                return tile_id;
            };

            // address incrementer for output_tensor
            uint32_t output_tile_id_start = b * output_batch_num_pages;
            uint32_t output_tiles_read = start_tiles_read;
            auto get_next_output_tile_id = [&]() -> uint32_t { return output_tile_id_start + (output_tiles_read++); };

            uint32_t chunk_count = 0;
            uint32_t even_chunk_count = 0;
            uint32_t odd_chunk_count = 0;
            for (uint32_t c = 0; c < slice_C; ++c) {
                // reset addr counters
                interm_pages_read_in_row = start_pages_read_in_row;
                interm_row_offset = start_row_offset;
                output_tiles_read = start_tiles_read;
                uint32_t tiles_read = start_tiles_read;
                uint32_t total_tiles_to_read = start_tiles_to_read;

                while (tiles_read < total_tiles_to_read) {
                    const auto [is_even_chunk, tiles_to_read] =
                        reduce_scatter_common::chunk_ring_parity<tile_granularity>(tiles_read, total_tiles_to_read);

                    if ((is_even_chunk && !even_chunks) || (!is_even_chunk && !odd_chunks) || tiles_to_read == 0) {
                        // Skip this chunk
                        tiles_read += tiles_to_read;
                        for (uint32_t k = 0; k < tiles_to_read; ++k) {
                            get_next_interm_tile_id();
                            get_next_output_tile_id();
                        }
                    } else {
                        const bool reduce_interm =
                            (is_even_chunk && reduce_even_chunks) || (!is_even_chunk && reduce_odd_chunks);
                        const uint32_t cb_out =
                            reduce_interm ? cb_compute_output_id : cb_reader_output_id;  // from compute or reader

                        if (write_to_remote) {
                            // Pick the semaphore this chunk signals and the counter that paces it. In
                            // separate-sem mode even/odd chunks signal different workers; otherwise every
                            // chunk signals this worker's peer. The counter is advanced after the writes, so
                            // fuse_seminc predicts against counter + 1: true means this chunk's final packet
                            // reaches chunks_per_sync and can carry the increment itself.
                            uint32_t& sync_counter = separate_even_odd_sems
                                                         ? (is_even_chunk ? even_chunk_count : odd_chunk_count)
                                                         : chunk_count;
                            const uint64_t sem_noc_addr =
                                separate_even_odd_sems
                                    ? (is_even_chunk ? even_core_sem_noc_addr : odd_core_sem_noc_addr)
                                    : this_core_sem_noc_addr;
                            const bool fuse_seminc = (sync_counter + 1 == chunks_per_sync);
                            bool seminc_fused = false;

                            // Write tiles to remote tensor over Fabric
                            cb_wait_front(cb_out, tile_granularity);
                            size_t l1_read_addr = get_read_ptr(cb_out);
                            for (uint32_t j = 0; j < tiles_to_read; j += num_tiles_to_write_per_packet) {
                                uint32_t tiles_to_put_in_current_packet =
                                    std::min(tiles_to_read - j, num_tiles_to_write_per_packet);

                                for (uint32_t k = 0; k < tiles_to_put_in_current_packet; ++k) {
                                    auto interm_tile_id = get_next_interm_tile_id();
                                    auto output_tile_id = get_next_output_tile_id();
                                    if (write_to_interm) {
                                        remote_noc_addrs[k] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                            interm_tensor_accessor, interm_tile_id, 0);
                                    } else {
                                        remote_noc_addrs[k] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                            output_tensor_accessor, output_tile_id, 0);
                                    }
                                }

                                const bool last_packet = (j + num_tiles_to_write_per_packet >= tiles_to_read);
                                seminc_fused |= send_write_packet(
                                    l1_read_addr,
                                    tiles_to_put_in_current_packet,
                                    fuse_seminc && last_packet,
                                    sem_noc_addr);
                                noc_async_writes_flushed();
                                l1_read_addr += page_size * tiles_to_put_in_current_packet;
                                tiles_read += tiles_to_put_in_current_packet;
                            }
                            cb_pop_front(cb_out, tile_granularity);

                            // Advance this chunk's sync counter; emit the increment now unless it was already
                            // fused onto the final data packet above.
                            if (++sync_counter == chunks_per_sync) {
                                sync_counter = 0;
                                if (!seminc_fused) {
                                    send_seminc(sem_noc_addr);
                                }
                            }
                        } else {
                            // Write tiles to local tensor
                            cb_wait_front(cb_out, tile_granularity);
                            size_t l1_read_addr = get_read_ptr(cb_out);
                            for (uint32_t j = 0; j < tiles_to_read; ++j) {
                                auto interm_tile_id = get_next_interm_tile_id();
                                auto output_tile_id = get_next_output_tile_id();
                                uint64_t local_noc_addr;
                                if (write_to_interm) {
                                    local_noc_addr = interm_tensor_accessor.get_noc_addr(interm_tile_id);
                                } else {
                                    local_noc_addr = output_tensor_accessor.get_noc_addr(output_tile_id);
                                }
                                noc_async_write(l1_read_addr, local_noc_addr, page_size);
                                l1_read_addr += page_size;
                                tiles_read++;
                            }
                            noc_async_write_barrier();
                            cb_pop_front(cb_out, tile_granularity);
                        }  // if remote or local
                    }  // if skip or process
                }  // while total_tiles_to_read

                interm_tile_id_start += input_channel_num_pages;
                output_tile_id_start += output_channel_num_pages;
            }  // for slice_C

            // Flush any residual chunks whose counter never reached chunks_per_sync inside the loop.
            if (write_to_remote) {
                if (separate_even_odd_sems) {
                    if (even_chunks && even_chunk_count != 0) {
                        send_seminc(even_core_sem_noc_addr);
                    }
                    if (odd_chunks && odd_chunk_count != 0) {
                        send_seminc(odd_core_sem_noc_addr);
                    }
                } else {
                    if (chunk_count != 0) {
                        send_seminc(this_core_sem_noc_addr);
                    }
                }
            }

            // Next slice idx
            slice_idx = direction ? (slice_idx - 1) : (slice_idx + 1);
        }

        // Batch-ready barrier: a global all-workers sync so the next batch cannot clobber the reused
        // intermediate scratch or out_ready_sem while this batch is still being consumed. Skipped on the
        // final batch — there is no next batch to protect, and the reader gates receive-side completion.
        // input_tensor_B is a compile-time constant, so this whole block compiles away for B == 1.
        if (b + 1 < input_tensor_B) {
            // multicast to entire ring of workers for both this dir and opposite dir
            uint64_t batch_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(this_core_x, this_core_y, batch_ready_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_mcastseminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
            noc_async_writes_flushed();

            batch_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(opposite_core_x, opposite_core_y, batch_ready_sem, 0);
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_direction_connection,
                pkt_hdr_mcastseminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
            noc_async_writes_flushed();

            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 2 * (ring_size - 1));
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);  // reset before next batch
        }
    }

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

    noc_async_write_barrier();
}
