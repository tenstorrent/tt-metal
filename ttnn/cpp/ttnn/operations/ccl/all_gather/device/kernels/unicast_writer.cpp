// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>

#include "gather_walk.hpp"
#include "gather_noc.hpp"
#include "gather_args.hpp"
#include "gather_fabric.hpp"

using address_t = uint32_t;

// Store-and-forward AllGather: every fabric send is a single 1-hop unicast to the neighbor (HopSink in
// gather_fabric.hpp). Runs on any effectively-1D topology (both Fabric 1D and 2D).
//
// Store-and-forward writer: CB consumer, owns all fabric. It only ever sends (never waits on a semaphore --
// its sole backpressure is wait_front). Each iteration drains the CB and unicasts the stripe one hop to the
// neighbor's output (same address); iteration 0 also writes this device's local data into local output.
// Maintains the downstream reader's data_valid (= chunks delivered; see the protocol note in
// unicast_reader.cpp), and sends its one-shot "alive" barrier inc up front.
void kernel_main() {
    namespace ct = gather_args::unicast_writer::ct;
    namespace rt = gather_args::unicast_writer::rt;

    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t chunk_size = get_compile_time_arg_val(ct::chunk_size);
    constexpr uint32_t out_cpp = get_compile_time_arg_val(ct::out_cpp);
    constexpr uint32_t cps = get_compile_time_arg_val(ct::cps);
    constexpr uint32_t num_devices = get_compile_time_arg_val(ct::num_devices);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(ct::cb_id);
    constexpr uint32_t cb_entry_bytes = get_compile_time_arg_val(ct::cb_entry_bytes);
    constexpr uint32_t packet_size = get_compile_time_arg_val(ct::packet_size);
    constexpr uint32_t data_valid_granularity = get_compile_time_arg_val(ct::data_valid_granularity);
    constexpr uint32_t k = get_compile_time_arg_val(ct::k);
    constexpr uint32_t walk_stride = get_compile_time_arg_val(ct::walk_stride);
    constexpr uint32_t walk_xfer = get_compile_time_arg_val(ct::walk_xfer);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(ct::slot_bytes);
    constexpr RunSource out_runs = run_source_from_bits(get_compile_time_arg_val(ct::out_runs));
    constexpr auto output_tensor_args = TensorAccessorArgs<ct::tensor_args>();

    check_walk_ct<k, /*in_cpp=*/k, out_cpp, cps, chunk_size, slot_bytes, cb_entry_bytes, walk_xfer, packet_size>();
    constexpr uint32_t chunks_per_entry = cb_entry_bytes / slot_bytes;

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    const address_t output_tensor_address = get_arg_val<address_t>(rt::output_addr);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(rt::initial_stripe);
    const uint32_t stripe_step = get_arg_val<uint32_t>(rt::stripe_step);
    const uint32_t num_iters = get_arg_val<uint32_t>(rt::num_iters);
    const uint32_t slice_first = get_arg_val<uint32_t>(rt::slice_first);
    const uint32_t slice_count = get_arg_val<uint32_t>(rt::slice_count);
    const uint32_t final_skip = get_arg_val<uint32_t>(rt::final_skip);
    const uint32_t final_take = get_arg_val<uint32_t>(rt::final_take);
    const WalkPos final_pos = {
        get_arg_val<uint32_t>(rt::final_pos_base),
        get_arg_val<uint32_t>(rt::final_pos_lane),
        get_arg_val<uint32_t>(rt::final_pos_i)};
    const uint32_t do_local_write = get_arg_val<uint32_t>(rt::do_local_write);
    const address_t barrier_sem = get_arg_val<uint32_t>(rt::barrier_sem);
    const address_t data_valid_sem = get_arg_val<uint32_t>(rt::data_valid_sem);
    const uint8_t barrier_sem_noc_x = get_arg_val<uint32_t>(rt::barrier_noc_x);  // neighbor opposite-dir core
    const uint8_t barrier_sem_noc_y = get_arg_val<uint32_t>(rt::barrier_noc_y);
    const uint8_t data_valid_sem_noc_x = get_arg_val<uint32_t>(rt::data_valid_noc_x);  // mirror core
    const uint8_t data_valid_sem_noc_y = get_arg_val<uint32_t>(rt::data_valid_noc_y);
    const uint32_t num_granular_sends =
        get_arg_val<uint32_t>(rt::num_granular_sends);  // leading sends the downstream relays
    const uint16_t neighbor_chip_id = get_arg_val<uint32_t>(rt::neighbor_chip_id);
    const uint16_t neighbor_mesh_id = get_arg_val<uint32_t>(rt::neighbor_mesh_id);
    size_t arg_idx = rt::fabric_args;  // mux or fabric connection args start here
    [[maybe_unused]] size_t arg_for_fab = arg_idx;

    // A direction with no neighbor (a line endpoint) relays nothing; no fabric/mux connection was appended.
    if (num_iters == 0) {
        return;
    }

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

#ifdef USE_WORKER_MUX
    // Connect to our channel on the Fabric mux.
    //
    // TODO(perf): FabricMuxV2Sender<true> stages payloads into mux slots before the mux is READY. Likely low
    // value here -- a mux is only used when bandwidth-bound, and staging would delay the init barrier inc
    // that unblocks the neighbor's writer -- but cheap to try.
    // TODO(perf): FabricMuxV2Sender<false, N> makes slot wrap-around compile-time; needs num_buffers as a CT arg.
    using SenderT = tt::tt_fabric::FabricMuxV2Sender<>;
    SenderT mux_connection = SenderT::build_from_args(arg_idx);
    mux_connection.open();
    SenderT* sender = &mux_connection;
#else
    // Connect directly to the neighbor's ERISC.
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, 1, arg_for_fab);
    using SenderT = tt::tt_fabric::WorkerToFabricEdmSender;
    SenderT* sender = &fabric_connection.get(0).sender;
#endif

    HopSink<SenderT> hop(noc, sender, neighbor_chip_id, neighbor_mesh_id);
    PacketPacker<packet_size, slot_bytes, HopSink<SenderT>> fabric(hop);

    // Init handshake (send only): tell the neighbor's opposite-direction reader we're alive, so it lets its
    // paired writer start writing into our output. Our own reader does the matching wait.
    hop.atomic_inc(safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem, 0), 1);

    const uint64_t downstream_data_valid_addr =
        safe_get_noc_addr(data_valid_sem_noc_x, data_valid_sem_noc_y, data_valid_sem, 0);
    auto signal = [&](uint32_t chunks) { hop.atomic_inc(downstream_data_valid_addr, chunks); };

    ///////////////////////////////////////////////////
    // RUN SETUP
    ///////////////////////////////////////////////////

    check_walk_stride(output_tensor_accessor, k, out_cpp, walk_stride);
    const RunSource out_src = checked_run_source(out_runs, output_tensor_accessor, k, out_cpp, walk_stride);
    const uint32_t slice_end = slice_first + slice_count;

    TiledWalk<k, walk_stride, walk_xfer> walk;
    StripeMap<cps, num_devices> map;

    auto output_addr = [&](uint32_t page, uint32_t off) {
        return output_tensor_accessor.get_noc_addr(page, off, noc.get_noc_id());
    };

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    uint32_t stripe = initial_stripe;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const bool last = (iter == num_iters - 1);
        // An even ring splits the antipode stripe between the two directions. The split is by seqno, not chunk
        // id, so that the positions data_valid counts still line up downstream.
        const uint32_t skip = last ? final_skip : 0;
        const uint32_t take = last ? final_take : slice_count;
        // The walk does not stop itself: past the slice it would send another worker's chunks.
        ASSERT(skip + take <= slice_count);
        const bool local_copy = (iter == 0) && (do_local_write != 0);
        const bool granular = (iter < num_granular_sends);  // downstream relays this stripe -> signal fine-grained
        map.init(stripe, slice_end);
        // final_pos is the host's walk_seek(final_skip); every other iteration starts at seqno 0.
        walk.init(slice_end, last ? final_pos : walk_start(slice_first));

        uint32_t pending_chunks = 0, pending_entries = 0;
        for (uint32_t chunks_sent = 0; chunks_sent < take;) {
            const uint32_t batch = std::min(chunks_per_entry, take - chunks_sent);
            cb.wait_front(1);
            emit_runs<out_cpp, chunk_size, slot_bytes>(
                walk,
                map,
                output_tensor_accessor,
                out_src,
                cb.get_read_ptr(),
                batch,
                output_addr,
                [&](uint32_t l1, uint32_t page, uint32_t off, uint32_t run) {
                    fabric.queue_segment(
                        l1,
                        tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page, off),
                        run * slot_bytes);
                    if (local_copy) {
                        // Local data -> our output stripe (same address).
                        local_write<slot_bytes>(noc, l1, output_addr(page, off), run);
                    }
                });
            if (local_copy) {
                noc.async_writes_flushed<NocOptions::POSTED>();
            }
            fabric.flush_packet_and_wait();
            cb.pop_front(1);

            pending_chunks += batch;
            if (granular && ++pending_entries == data_valid_granularity) {
                signal(pending_chunks);
                pending_chunks = 0;
                pending_entries = 0;
            }
            chunks_sent += batch;
        }
        // Trailing chunks of a relayed stripe, or the whole of a sink stripe (granular == false).
        if (pending_chunks > 0) {
            signal(pending_chunks);
        }
        stripe = (stripe + stripe_step) % num_devices;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Commit our own NOC writes (the iter-0 local copy, plus the packet writes into the mux buffer) before
    // teardown.
    noc_async_write_barrier();
    noc_async_atomic_barrier();

#ifdef USE_WORKER_MUX
    mux_connection.close();
#else
    close_connections(fabric_connection);
#endif
    noc.async_write_barrier();
}
