// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace dataflow_kernel_lib::ccl;  // FabricStreamSender / MuxConn / the armed channels
namespace sched = ttnn::ccl::schedule;     // the dim-zero ring schedule shared with the reader + compute kernel

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
constexpr uint32_t output_num_pages = get_named_compile_time_arg_val("output_num_pages");
constexpr uint32_t batch_num_pages = get_named_compile_time_arg_val("batch_num_pages");
constexpr uint32_t slice_B = get_named_compile_time_arg_val("slice_B");

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
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 1 is forward, 0 is backward
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

#ifdef USE_WORKER_MUX
    // Build the worker-mux egress through the helper: MuxConn reads this exact runtime-arg block
    // (same 17 fields, same order), then waits for the mux endpoint to be ready.
    MuxConn<fabric_mux_num_buffers_per_channel> mux_conn(
        arg_cursor(arg_idx),
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_status_address,
        fabric_mux_termination_signal_address,
        num_mux_clients);
#endif

    const auto& unicast_route_info = (direction == 1) ? forward_unicast_route_info : backward_unicast_route_info;
    const auto& multicast_route_info = (direction == 1) ? forward_multicast_route_info : backward_multicast_route_info;

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx>();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_address);

    constexpr auto output_tensor_args = TensorAccessorArgs<intermediate_tensor_args.next_compile_time_args_offset()>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address);

    // Wrap the egress in the helper. Both policies expose the same open()/arm_*/close() surface.
#ifdef USE_WORKER_MUX
    FabricStreamSender<MuxConn<fabric_mux_num_buffers_per_channel>> sender(mux_conn, /*alignment=*/1);
#else
    FabricStreamSender<> sender(arg_cursor(arg_idx), /*is_forward=*/direction, /*alignment=*/1);
#endif

    // open(route) binds this stream's unicast route once; every unicast arm_* below reuses it.
    auto stream = sender.open(unicast_route_info);

    // Arm once, issue many. Each channel draws its own pooled header. NOTE: the multicast
    // barrier/batch-ready channel always programs its multicast route, where the pre-migration
    // kernel set it only under use_barrier_sem while reusing the same header for the batch-ready
    // increment — which sent the batch-ready multicast on an UNROUTED header whenever
    // use_barrier_sem was false. Always setting it matches the use_barrier_sem==true behaviour.
    auto scatter = stream.arm_scatter_write(page_size, num_tiles_to_write_per_packet);
    auto writer = stream.arm_unicast_write(page_size);
    auto counter = stream.arm_inc(1);
    auto barrier = stream.arm_inc(multicast_route_info, 1);

    if (use_barrier_sem) {
        // multicast to entire ring of workers going in the same direction
        barrier.inc(safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0));

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    const uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    // The dim-zero ring schedule — the neighbour-first slice walk, the interleaved own/other chunk
    // pairing, and the chunks-per-sync signal cadence — comes from the shared header; this writer's
    // counter.inc()s pair with the next chip's reader waits through the same SyncCadence.
    auto slice_cursor = sched::RingSliceCursor::starting_at(
        sched::ring_neighbour_first_slice(my_chip_id, direction), ring_size, direction);
    sched::DimZeroChunkWalk walk(slice_B, tile_granularity, start_tiles_read, start_tiles_to_read, direction);
    sched::SyncCadence cadence(chunks_per_sync);

    static_assert(num_tiles_to_write_per_packet <= 2, "dim-zero ring writer packs at most 2 tiles per packet");

    for (uint32_t i = 0; i < ring_size; ++i) {
        // If not the last slice, write what's on cb_output_id forward
        const uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;
        const uint32_t actual_slice_idx = slice_cursor.wrap();

        if (i < (ring_size - 1)) {
            uint32_t intermediate_tile_id_start = actual_slice_idx * output_num_pages;

            cadence.reset();
            walk.reset();
            while (walk.next_batch()) {
                while (walk.next_chunk()) {
                    const uint32_t tiles_this_chunk = walk.tiles_this_chunk();

                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    for (uint32_t j = 0; j < tiles_this_chunk; j += num_tiles_to_write_per_packet) {
                        const uint32_t tiles_to_put_in_current_packet =
                            std::min(tiles_this_chunk - j, num_tiles_to_write_per_packet);
                        const uint32_t first_tile_id = intermediate_tile_id_start + walk.position() + j;

                        if (tiles_to_put_in_current_packet == 2) {
                            const uint64_t remote_noc_addrs[2] = {
                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                    intermediate_addrgen, first_tile_id, 0),
                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                    intermediate_addrgen, first_tile_id + 1, 0)};
                            scatter.write_scatter(remote_noc_addrs, 2, l1_read_addr);
                            l1_read_addr += page_size * 2;
                        } else {
                            writer.write_page(l1_read_addr, first_tile_id, intermediate_addrgen);
                            l1_read_addr += page_size;
                        }
                        noc_async_writes_flushed();
                    }
                    cb_pop_front(cb_output_id, tile_granularity);

                    cadence.advance();
                    if (cadence.signal_due()) {
                        // 2. unicast output ready semaphore
                        counter.inc(out_ready_sem_noc_addr_in_pkt);
                    }
                }
                intermediate_tile_id_start += batch_num_pages;
            }

            if (cadence.tail_due()) {
                // 2. unicast output ready semaphore
                counter.inc(out_ready_sem_noc_addr_in_pkt);
            }
            noc_async_writes_flushed();
        } else {
            // Otherwise, on the last slice, write it to output buffer
            uint32_t output_tile_id_start = 0;
            walk.reset();
            while (walk.next_batch()) {
                while (walk.next_chunk()) {
                    const uint32_t tiles_this_chunk = walk.tiles_this_chunk();

                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    for (uint32_t j = 0; j < tiles_this_chunk; ++j) {
                        const uint32_t output_tile_id = output_tile_id_start + walk.position() + j;
                        const uint64_t local_noc_addr = output_addrgen.get_noc_addr(output_tile_id);
                        noc_async_write(l1_read_addr, local_noc_addr, page_size);
                        l1_read_addr += page_size;
                    }

                    noc_async_write_barrier();
                    cb_pop_front(cb_output_id, tile_granularity);
                }
                output_tile_id_start += batch_num_pages;
            }
        }

        slice_cursor.advance();
    }

    // Batch ready semaphore — multicast to the ring of workers going in the same direction.
    barrier.inc(safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0));
    noc_async_writes_flushed();

    // Reset the global semaphore
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ring_size - 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);

    // close() drains (write + atomic barriers) then tears the egress down behind the uniform helper
    // teardown (direct: close the connection; mux: disconnect + termination-master handshake).
    stream.close();

    noc_async_write_barrier();
}
