// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>

#include "unicast_common.hpp"

using address_t = uint32_t;

// Store-and-forward writer: CB consumer, owns all fabric. It only ever sends (never waits on a semaphore --
// its sole backpressure is wait_front). Each iteration drains the CB and unicasts the stripe one hop to the
// neighbor's output (same address); iteration 0 also writes this device's local data into local output.
// Maintains the downstream reader's data_valid (= chunks delivered; see the note in unicast_common.hpp), and
// sends its one-shot "alive" barrier inc up front.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t packet_size = get_compile_time_arg_val(6);
    constexpr bool do_init_barrier = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t data_valid_granularity = get_compile_time_arg_val(8);
    constexpr uint32_t run_cap_bytes = get_compile_time_arg_val(9);  // longest run the walk may emit; 0 = no cap
    constexpr auto output_tensor_args = TensorAccessorArgs<10>();

    constexpr uint32_t chunks_per_cb_entry = cb_page_size / output_chunk_size;
    constexpr uint32_t xfer_max = chunks_per_transfer(packet_size, output_chunk_size, run_cap_bytes);
    // A chunk bigger than a burst cannot be one NOC command, so it takes the generic path.
    constexpr bool one_command = output_chunk_size <= NOC_MAX_BURST_SIZE;
    // A run is emitted as one scatter chunk starting at its source offset within the packet, so every chunk
    // size has to keep source and destination NoC-write aligned.
    static_assert(output_chunk_size % 16 == 0, "chunk size must be a multiple of the NoC write alignment");

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_first_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_skip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_take = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t do_local_write = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);  // used only if do_init_barrier
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const uint8_t barrier_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // neighbor opposite-dir core
    [[maybe_unused]] const uint8_t barrier_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t data_valid_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // mirror core (data_valid_sem target)
    const uint8_t data_valid_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_granular_sends = get_arg_val<uint32_t>(arg_idx++);  // leading sends the downstream relays
    const uint16_t neighbor_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint16_t neighbor_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] size_t arg_for_fab = arg_idx;  // fabric connection args start here (non-mux path)

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
    // value here -- a mux is only used when bandwidth-bound, and staging would delay the do_init_barrier inc
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

    FabricWriter<packet_size, SenderT> fabric(noc, sender, neighbor_chip_id, neighbor_mesh_id);

    // Init handshake (send only): tell the neighbor's opposite-direction reader we're alive, so it lets its
    // paired writer start writing into our output. Our own reader does the matching wait.
    if constexpr (do_init_barrier) {
        fabric.atomic_inc(safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem, 0), 1);
    }

    const uint64_t downstream_data_valid_addr =
        safe_get_noc_addr(data_valid_sem_noc_x, data_valid_sem_noc_y, data_valid_sem, 0);
    auto signal = [&](uint32_t chunks) { fabric.atomic_inc(downstream_data_valid_addr, chunks); };

    ///////////////////////////////////////////////////
    // RUN SETUP
    ///////////////////////////////////////////////////

    const auto plan = walk_plan<output_chunks_per_page, output_chunk_size, xfer_max>(output_tensor_accessor);

    TiledWalk walk;
    StripeMap<output_chunks_per_stripe, num_devices> map;

    auto run_addr = [&](uint32_t chunk) {
        const uint32_t global = map.at(chunk).global;
        return output_tensor_accessor.get_noc_addr(
            page_of<output_chunks_per_page>(global),
            byte_off_of<output_chunks_per_page, output_chunk_size>(global),
            noc.get_noc_id());
    };

    auto local_write = [&](uint32_t l1_read_addr, uint64_t dst, uint32_t chunks) {
        // Posted write on a separate VC so it doesn't contend with the fabric writes on the same NOC.
        if constexpr (one_command) {
            noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                CoreLocalMem<uint32_t>(l1_read_addr),
                tensor_accessor::Page(dst, 0),
                chunks * output_chunk_size,
                {},
                {},
                {.vc = NOC_UNICAST_WRITE_VC + 1});
        } else {
            noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
                CoreLocalMem<uint32_t>(l1_read_addr),
                tensor_accessor::Page(dst, 0),
                output_chunk_size,
                {},
                {},
                {.vc = NOC_UNICAST_WRITE_VC + 1});
        }
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
        const uint32_t take = last ? final_take : slice_chunks;
        // The walk does not stop itself: past the slice it would send another worker's chunks.
        ASSERT(skip + take <= slice_chunks);
        const bool granular = (iter < num_granular_sends);  // downstream relays this stripe -> signal fine-grained
        const bool local_copy = (iter == 0) && (do_local_write != 0);
        map.init(stripe);
        walk.init(slice_first_chunk, slice_chunks, skip, plan.stride, plan.xfer);

        uint32_t pending_chunks = 0, pending_pages = 0;
        for (uint32_t chunks_sent = 0; chunks_sent < take;) {
            const uint32_t batch = std::min(chunks_per_cb_entry, take - chunks_sent);
            cb.wait_front(1);
            uint32_t l1_read_addr = cb.get_read_ptr();
            for (uint32_t left = batch; left > 0;) {
                const auto pos = map.at(walk.chunk());
                const uint32_t page = page_of<output_chunks_per_page>(pos.global);
                const uint32_t off = byte_off_of<output_chunks_per_page, output_chunk_size>(pos.global);
                const uint32_t run = next_run<output_chunks_per_page>(
                    walk, output_tensor_accessor, plan.out, pos.global, pos.row_end, left);
                ASSERT(run_is_linear(walk, run, output_chunk_size, run_addr(walk.chunk()), run_addr));
                fabric.queue_segment(
                    l1_read_addr,
                    tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor_accessor, page, off),
                    run * output_chunk_size);
                if (local_copy) {
                    // Local data -> our output stripe (same address).
                    local_write(l1_read_addr, output_tensor_accessor.get_noc_addr(page, off, noc.get_noc_id()), run);
                }
                l1_read_addr += run * output_chunk_size;
                left -= run;
                walk.advance(run);
            }
            if (local_copy) {
                noc.async_writes_flushed<NocOptions::POSTED>();
            }
            fabric.flush_packet_and_wait();
            cb.pop_front(1);

            pending_chunks += batch;
            if (granular && ++pending_pages == data_valid_granularity) {
                signal(pending_chunks);
                pending_chunks = 0;
                pending_pages = 0;
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
