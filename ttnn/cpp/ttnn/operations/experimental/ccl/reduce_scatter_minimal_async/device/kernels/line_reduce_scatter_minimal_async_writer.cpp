// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace dataflow_kernel_lib::ccl;  // FabricStreamSender / MuxConn / the armed channels
namespace sched = ttnn::ccl::schedule;     // the line schedule shared with the reader + compute kernel

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(2);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(5);
constexpr uint32_t input_num_pages = get_compile_time_arg_val(6);
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t output_batch_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t output_channel_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(11);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(12);
constexpr uint32_t slice_C = get_compile_time_arg_val(13);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(14);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(15);
constexpr uint32_t dim = get_compile_time_arg_val(16);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(17);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(21);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(22);
constexpr uint32_t barrier_target_count = get_compile_time_arg_val(23);

constexpr uint32_t num_ct_args = 24;

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
    Semaphore<> fwd_bwd_sem(get_arg_val<uint32_t>(arg_idx++));
    uint32_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool is_forward = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_device_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_targets_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const bool do_final_reduction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    // Build the worker-mux egress through the helper: MuxConn reads this exact runtime-arg block
    // (same 17 fields, same order) advancing the cursor, then waits for the mux endpoint to be
    // ready — exactly the hand-rolled build + wait_for_fabric_endpoint_ready it replaces. A worker
    // with no link in its direction has valid==false and never issues (every send below is gated on
    // num_targets_in_direction).
    MuxConn<fabric_mux_num_buffers_per_channel> mux_conn(
        arg_cursor(arg_idx),
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_status_address,
        fabric_mux_termination_signal_address,
        num_mux_clients);

    const auto& unicast_route_info = (is_forward) ? forward_unicast_route_info : backward_unicast_route_info;
    // Upstream replaced the multicast startup barrier with a neighbour unicast handshake
    // (reshaped-mesh support + the folded-FABRIC_2D barrier fix), so no multicast route is issued
    // from this kernel any more; the CT route info is still parsed to keep the arg layout stable.
    [[maybe_unused]] const auto& multicast_route_info =
        (is_forward) ? forward_multicast_route_info : backward_multicast_route_info;

    // SPLIT OPEN: start the mux connection handshake now, overlap it with the address-generator
    // construction below (the sharded variants read whole mapping tables from runtime args), and
    // finish it just before the first issue. This is the hand-rolled
    // fabric_client_connect_start/_finish overlap this kernel carried, expressed through the helper.
    FabricStreamSender<MuxConn<fabric_mux_num_buffers_per_channel>> sender(mux_conn, /*alignment=*/1);
    sender.open_start();

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

#ifdef INTERMEDIATE_IS_SHARDED
    constexpr uint32_t ct_offset = 7;

    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx),       // Memory layout
        get_compile_time_arg_val(ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(ct_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_addrgen = {
        .bank_base_address = intermediate_address, .shard_array = intermediate_mapping_table};

    arg_idx += intermediate_rt_increment;

#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_address);
#endif

#ifdef OUTPUT_IS_SHARDED
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx + ct_offset),       // Memory layout
        get_compile_time_arg_val(ct_idx + ct_offset + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + ct_offset + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + ct_offset + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(ct_idx + ct_offset + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(ct_idx + ct_offset + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + ct_offset + 6)>;  // pages_per_shard_y

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address);
#endif

    // Complete the handshake and bind this direction's unicast route; every unicast arm below
    // reuses it. Must precede the first issue (the barrier multicast may issue right away).
    auto stream = sender.open_finish(unicast_route_info);

    // Arm once, issue many. Each channel draws its own pooled header; arming is a local header
    // program (no fabric I/O), so it is unconditional even though every ISSUE below is gated on
    // num_targets_in_direction. The startup barrier now rides the UNICAST inc channel: upstream
    // replaced the two-target multicast barrier with a single neighbour unicast handshake, which is
    // the same header/route the counting incs use — so one armed unicast channel serves both, as
    // upstream reuses one pkt_hdr_seminc for both.
    auto scatter = stream.arm_scatter_write(page_size, contig_pages_advanced);
    auto writer = stream.arm_unicast_write(page_size);
    auto counter = stream.arm_inc(1);

    if (use_barrier_sem) {
        if (num_targets_in_direction) {
            // Use neighbor unicast instead of multicast to support reshaped 'logical linear' mesh
            // devices: increment only the worker going in the opposite direction.
            counter.inc(safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0));

            // Exactly one increment arrives: the two workers sharing a link (Dk forward and Dk+1
            // backward) handshake with each other, so the target is structurally 1 and does not
            // depend on barrier_target_count. A worker with no link in its direction neither
            // signals nor waits, so the wait lives inside this guard.
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 1);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        }
    }

    Noc noc_obj;
    CircularBuffer cb_compute_output(cb_compute_output_id);
    CircularBuffer cb_reader_output(cb_reader_output_id);

    const uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    /**
     * Intermediate buffer is double-sized (shape [2, *input_shape]) to accommodate forward and
     * backward. BWD indexes into second half of intermediate buffer.
     */
    const uint32_t intermediate_full_offset = is_forward ? 0 : input_num_pages;

    // The line schedule — slice sequence (forward counts DOWN from ring_size-1, backward UP from
    // 0), channel/chunk boundaries, and the chunks-per-sync signal cadence — comes from the shared
    // header, so this writer, the reader and the compute kernel walk ONE definition. The reader's
    // out_ready waits pair with this writer's counter.inc()s through the same SyncCadence.
    sched::LineChannelWalk walk(slice_C, tile_granularity, start_tiles_read, start_tiles_to_read);
    sched::SyncCadence cadence(chunks_per_sync);
    sched::SliceRowWalker interm_walker(slice_Wt, input_tensor_Wt);
    sched::SequentialTileWalker output_walker;

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        // Per-batch: the line walk restarts at the far end of this direction.
        sched::LineSliceCursor slice_cursor(is_forward, ring_size);
        const uint32_t batch_offset = input_batch_num_pages * b;

        for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
            CircularBuffer& cb_output = is_first_device_in_direction ? cb_reader_output : cb_compute_output;
            cadence.reset();
            interm_walker.set_base(
                sched::slice_tile_offset(dim, slice_cursor.slice(), slice_C, slice_Ht, slice_Wt) + batch_offset +
                intermediate_full_offset);

            walk.reset();
            while (walk.next_channel()) {
                interm_walker.reset_offsets(start_pages_read_in_row, start_row_offset);

                // Write to remote intermediate buffer
                while (walk.next_chunk()) {
                    const uint32_t num_pages_to_read = walk.tiles_this_chunk();

                    cb_output.wait_front(tile_granularity);
                    size_t l1_read_addr = cb_output.get_read_ptr();
                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        const uint32_t num_pages_to_write = std::min(contig_pages_advanced, num_pages_to_read - j);

                        if (num_pages_to_write == 1) {
                            writer.write_page(l1_read_addr, interm_walker.next(), intermediate_addrgen);
                            l1_read_addr += page_size;
                        } else if (num_pages_to_write == 2) {
                            const uint64_t remote_noc_addrs[2] = {
                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                    intermediate_addrgen, interm_walker.next(), 0),
                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                    intermediate_addrgen, interm_walker.next(), 0)};
                            scatter.write_scatter(remote_noc_addrs, 2, l1_read_addr);
                            l1_read_addr += page_size * 2;
                        } else {
                            ASSERT(false);
                        }
                        noc_obj.async_writes_flushed();
                    }
                    cb_output.pop_front(tile_granularity);

                    cadence.advance();
                    if (cadence.signal_due()) {
                        // 2. unicast output ready semaphore
                        counter.inc(out_ready_sem_noc_addr_in_pkt);
                    }
                }
                interm_walker.bump_base(input_channel_num_pages);
            }

            if (cadence.tail_due()) {
                // 2. unicast output ready semaphore
                counter.inc(out_ready_sem_noc_addr_in_pkt);
            }

            slice_cursor.advance();
        }

        // Do write of final reduction and sync local FWD/BWD cores
        if (do_final_reduction) {
            // If both directions land a final reduction on this core pair, the FORWARD side hands
            // each finished chunk to the backward reader (which accumulates onto the output) —
            // one definition of the handshake, shared with the reader.
            const bool hands_off = sched::line_rs_forward_hands_off(sync_with_other_direction, is_forward);

            // Write output
            output_walker.set_base(b * output_batch_num_pages);
            walk.reset();
            while (walk.next_channel()) {
                output_walker.reset_offsets(start_tiles_read);
                while (walk.next_chunk()) {
                    const uint32_t num_pages_to_read = walk.tiles_this_chunk();

                    cb_compute_output.wait_front(tile_granularity);
                    uint32_t l1_read_addr = cb_compute_output.get_read_ptr();
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        const uint64_t noc_write_addr = output_addrgen.get_noc_addr(output_walker.next());
                        noc_async_write(l1_read_addr, noc_write_addr, page_size);
                        l1_read_addr += page_size;
                    }

                    if (hands_off) {
                        noc_obj.async_write_barrier();
                    } else {
                        noc_obj.async_writes_flushed();
                    }
                    cb_compute_output.pop_front(tile_granularity);
                    if (hands_off) {
                        // Tell local backwards reader that it can proceed
                        fwd_bwd_sem.up(noc_obj, opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, 1);
                    }
                }
                output_walker.bump_base(output_channel_num_pages);
            }
            noc_obj.async_write_barrier();
        }
    }

    // close() drains (write + atomic barriers) then runs the mux teardown handshake behind the
    // uniform helper teardown (disconnect; termination master waits for all clients then signals
    // the mux endpoint to terminate). The stream dtor would also close — idempotent.
    stream.close();

    noc_obj.async_write_barrier();
}
