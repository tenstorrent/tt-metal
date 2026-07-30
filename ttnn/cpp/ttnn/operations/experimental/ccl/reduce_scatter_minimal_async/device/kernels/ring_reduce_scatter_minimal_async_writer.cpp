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
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;
using namespace dataflow_kernel_lib::ccl;  // FabricStreamSender / MuxConn / the armed channels
namespace sched = ttnn::ccl::schedule;     // the ring schedule shared with the reader + compute kernel

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
    // Build the worker-mux egress through the helper: MuxConn reads this exact runtime-arg block
    // (same 17 fields, same order) advancing the cursor, then waits for the mux endpoint to be
    // ready. arg_idx is uint32_t here and the helper takes a size_t& cursor, so bridge and sync back.
    // The connection is wrapped in a FabricStreamSender below, so arm/issue/teardown are identical
    // for the mux and direct paths.
    size_t conn_arg_idx = arg_idx;
    MuxConn<fabric_mux_num_buffers_per_channel> mux_conn(
        conn_arg_idx,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_status_address,
        fabric_mux_termination_signal_address,
        num_mux_clients);
    arg_idx = conn_arg_idx;
#endif

    const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

    constexpr auto interm_tensor_args = TensorAccessorArgs<ct_idx>();
    auto interm_tensor_accessor = TensorAccessor(interm_tensor_args, interm_tensor_address);

    constexpr auto output_tensor_args = TensorAccessorArgs<interm_tensor_args.next_compile_time_args_offset()>();
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    // Wrap the egress in the helper. Both policies expose the same open()/arm_*/close() surface, so
    // the per-send #ifdef, the four hand-allocated packet headers, the route programming and the
    // four set_state calls below all collapse into open() + four arm_*() calls.
#ifdef USE_WORKER_MUX
    FabricStreamSender<MuxConn<fabric_mux_num_buffers_per_channel>> sender(mux_conn, /*alignment=*/1);
#else
    // direction: 1 is forward, 0 is backward (this op's convention — note it is the OPPOSITE of
    // all_gather_async's writer, which treats 0 as forward).
    size_t conn_arg_idx = arg_idx;
    FabricStreamSender<> sender(conn_arg_idx, /*is_forward=*/direction, /*alignment=*/1);
    arg_idx = conn_arg_idx;
#endif

    // open(route) binds this stream's unicast route once; every unicast arm_* below reuses it, so an
    // unrouted send cannot be expressed.
    auto stream = sender.open(unicast_route_info);

    static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported");
    uint64_t remote_noc_addrs[4] = {0, 0, 0, 0};
    // Arm once, issue many. Each channel draws its own pooled header, so all four coexist with no
    // ordering constraint. Arming is a local header program (no fabric I/O), so it is unconditional
    // even where the issues below are gated.
    auto scatter = stream.arm_scatter_write(page_size, num_tiles_to_write_per_packet);
    auto writer = stream.arm_unicast_write(page_size);
    auto counter = stream.arm_inc(1);
    // The multicast barrier/batch-ready channel carries its own MULTICAST route, distinct from the
    // stream's unicast one. NOTE: arm_multicast_inc programs both the multicast send state AND the
    // line-multicast route, whereas the pre-migration code set the route only inside the
    // `use_barrier_sem` branch while reusing the same header later for the batch-ready increment.
    // Always setting it matches the use_barrier_sem==true behaviour on every path.
    auto barrier = stream.arm_multicast_inc(multicast_route_info, 1);

    if (use_barrier_sem) {
        // multicast to entire ring of workers for both this dir and opposite dir
        uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(this_core_x, this_core_y, barrier_sem, 0);
        barrier.multicast_inc(barrier_sem_noc_addr_in_pkt);

        barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(opposite_core_x, opposite_core_y, barrier_sem, 0);
        barrier.multicast_inc(barrier_sem_noc_addr_in_pkt);

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 2 * (ring_size - 1));
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    // Relevant for 2nd-last iter:
    // In 2nd-last iter we send the full tensor slice. But in preparation for the last iter where each dir
    // processes half tensor slice, in 2nd-last iter we send sem increments to both forward and backward workers.
    // For example, if we send 2 even chunks and 2 odd chunks, we need to send 2 sem incrs to forward worker
    // and 2 sem incrs to backward worker.
    uint64_t this_core_sem_noc_addr = safe_get_noc_addr(this_core_x, this_core_y, out_ready_sem, 0);
    uint64_t opposite_core_sem_noc_addr = safe_get_noc_addr(opposite_core_x, opposite_core_y, out_ready_sem, 0);
    uint64_t even_core_sem_noc_addr = direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;
    uint64_t odd_core_sem_noc_addr = !direction ? this_core_sem_noc_addr : opposite_core_sem_noc_addr;

    // The ring schedule — slice walk, per-step flags, even/odd chunk split and the tile-id walkers —
    // comes from the shared header, so this writer, the reader and the compute kernel are driven by
    // ONE definition instead of three hand-maintained copies. The writer reads the same flag struct
    // the other two do, plus the three write_*/separate_even_odd_sems fields only it needs.
    static_assert(
        sched::is_supported_scatter_dim(dim), "ring reduce-scatter supports dim 1, 2 or 3 (dim 0 is dim_zero_*)");
    sched::RingRsSchedule schedule(
        ring_size, input_tensor_B, slice_C, tile_granularity, start_tiles_read, start_tiles_to_read, direction);
    sched::SliceRowWalker interm_walker(slice_Wt, input_tensor_Wt);
    sched::SequentialTileWalker output_walker;

    while (schedule.next_batch()) {
        const uint32_t b = schedule.batch_idx();
        // Per-batch: every batch restarts the ring walk at the same first slice.
        sched::RingSliceCursor slice_cursor(my_chip_id, ring_size, direction);
        while (schedule.next_step()) {
            const auto& flags = schedule.flags();
            const bool write_to_remote = flags.write_to_remote;
            const bool write_to_interm = flags.write_to_interm;
            const bool separate_even_odd_sems = flags.separate_even_odd_sems;
            const uint32_t slice_idx = slice_cursor.wrap();

            // address incrementers for interm_tensor and output_tensor
            interm_walker.set_base(sched::slice_tile_offset(dim, slice_idx, slice_C, slice_Ht, slice_Wt));
            output_walker.set_base(b * output_batch_num_pages);

            uint32_t chunk_count = 0;
            uint32_t even_chunk_count = 0;
            uint32_t odd_chunk_count = 0;
            while (schedule.next_channel()) {
                // reset addr counters
                interm_walker.reset_offsets(start_pages_read_in_row, start_row_offset);
                output_walker.reset_offsets(start_tiles_read);

                while (schedule.next_chunk()) {
                    const uint32_t tiles_to_read = schedule.tiles_this_chunk();
                    const bool is_even_chunk = schedule.is_even_chunk();

                    if (schedule.skip()) {
                        // Not this worker's parity this step: keep the walkers in step with the
                        // schedule and move on.
                        interm_walker.advance(tiles_to_read);
                        output_walker.advance(tiles_to_read);
                        continue;
                    }

                    const uint32_t cb_out =
                        schedule.reduce_interm() ? cb_compute_output_id : cb_reader_output_id;  // compute or reader

                    if (write_to_remote) {
                        // Write tiles to remote tensor over Fabric
                        cb_wait_front(cb_out, tile_granularity);
                        size_t l1_read_addr = get_read_ptr(cb_out);
                        for (uint32_t j = 0; j < tiles_to_read; j += num_tiles_to_write_per_packet) {
                            uint32_t tiles_to_put_in_current_packet =
                                std::min(tiles_to_read - j, num_tiles_to_write_per_packet);

                            for (uint32_t k = 0; k < tiles_to_put_in_current_packet; ++k) {
                                auto interm_tile_id = interm_walker.next();
                                auto output_tile_id = output_walker.next();
                                if (write_to_interm) {
                                    remote_noc_addrs[k] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                        interm_tensor_accessor, interm_tile_id, 0);
                                } else {
                                    remote_noc_addrs[k] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                        output_tensor_accessor, output_tile_id, 0);
                                }
                            }

                            if (tiles_to_put_in_current_packet > 1) {
                                scatter.write_scatter(remote_noc_addrs, tiles_to_put_in_current_packet, l1_read_addr);
                            } else {
                                writer.write(remote_noc_addrs[0], l1_read_addr);
                            }
                            noc_async_writes_flushed();
                            l1_read_addr += page_size * tiles_to_put_in_current_packet;
                        }
                        cb_pop_front(cb_out, tile_granularity);

                        // Send semaphore increment to remote worker core
                        ++chunk_count;
                        even_chunk_count += is_even_chunk;
                        odd_chunk_count += !is_even_chunk;
                        if (separate_even_odd_sems) {
                            if (is_even_chunk && even_chunk_count == chunks_per_sync) {
                                even_chunk_count = 0;
                                counter.inc(even_core_sem_noc_addr);
                                noc_async_writes_flushed();
                            } else if (!is_even_chunk && odd_chunk_count == chunks_per_sync) {
                                odd_chunk_count = 0;
                                counter.inc(odd_core_sem_noc_addr);
                                noc_async_writes_flushed();
                            }
                        } else {
                            if (chunk_count == chunks_per_sync) {
                                chunk_count = 0;
                                counter.inc(this_core_sem_noc_addr);
                                noc_async_writes_flushed();
                            }
                        }
                    } else {
                        // Write tiles to local tensor
                        cb_wait_front(cb_out, tile_granularity);
                        size_t l1_read_addr = get_read_ptr(cb_out);
                        for (uint32_t j = 0; j < tiles_to_read; ++j) {
                            auto interm_tile_id = interm_walker.next();
                            auto output_tile_id = output_walker.next();
                            uint64_t local_noc_addr;
                            if (write_to_interm) {
                                local_noc_addr = interm_tensor_accessor.get_noc_addr(interm_tile_id);
                            } else {
                                local_noc_addr = output_tensor_accessor.get_noc_addr(output_tile_id);
                            }
                            noc_async_write(l1_read_addr, local_noc_addr, page_size);
                            l1_read_addr += page_size;
                        }
                        noc_async_write_barrier();
                        cb_pop_front(cb_out, tile_granularity);
                    }  // if remote or local
                }  // while chunks

                interm_walker.bump_base(input_channel_num_pages);
                output_walker.bump_base(output_channel_num_pages);
            }  // while channels

            // Send semaphore increment to remote worker core (cleanup, when chunks_per_sync doesn't evenly divide
            // total_tiles_to_read)
            if (write_to_remote) {
                if (separate_even_odd_sems) {
                    if (flags.even_chunks && even_chunk_count != 0) {
                        counter.inc(even_core_sem_noc_addr);
                        noc_async_writes_flushed();
                    }
                    if (flags.odd_chunks && odd_chunk_count != 0) {
                        counter.inc(odd_core_sem_noc_addr);
                        noc_async_writes_flushed();
                    }
                } else {
                    if (chunk_count != 0) {
                        counter.inc(this_core_sem_noc_addr);
                        noc_async_writes_flushed();
                    }
                }
            }

            // Next slice idx
            slice_cursor.advance();
        }

        // Batch ready semaphore - multicast to entire ring of workers for both this dir and opposite dir
        uint64_t batch_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(this_core_x, this_core_y, batch_ready_sem, 0);
        barrier.multicast_inc(batch_ready_sem_noc_addr_in_pkt);
        noc_async_writes_flushed();

        batch_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(opposite_core_x, opposite_core_y, batch_ready_sem, 0);
        barrier.multicast_inc(batch_ready_sem_noc_addr_in_pkt);
        noc_async_writes_flushed();

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 2 * (ring_size - 1));
        noc_semaphore_set(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);  // reset semaphore before next batch
    }

    // close() drains (write + atomic barriers) then tears the egress down behind the uniform helper
    // teardown: the direct policy closes the connection, the mux policy disconnects and runs the
    // termination-master handshake (master waits for all clients, then signals the mux endpoint).
    // The stream dtor would also close — idempotent.
    stream.close();

    noc_async_write_barrier();
}
