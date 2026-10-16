// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

#include <cstdint>

#include "gather_walk.hpp"
#include "gather_noc.hpp"
#include "gather_args.hpp"

using address_t = uint32_t;

// Store-and-forward reader: CB producer, no fabric. Iteration 0 fills the CB from this device's local data; later
// iterations relay the stripe upstream delivered into our output, gated on data_valid.
//
// Both cases walk the same chunks in the same order the writer will drain them; only the source differs.
//
////////////////////////////////////////////////////////////////
// data_valid semaphore protocol
//
// data_valid counts the chunks upstream has relayed into our output -- cumulative over the op. A chunk's
// absolute position is base_seqno + its seqno in the batch, with base_seqno = (iter-1) * slice_count + skip
// (gather_rules.hpp UnicastSchedule). The upstream writer maintains the count (atomic-inc per chunks delivered);
// this reader waits on it with noc_semaphore_wait_min at the last chunk of each batch it reads, then a final wait
// for total_chunks.
//
// This semaphore is reused across cached invocations, so the reader subtracts total_chunks instead of
// clearing, which would drop credits posted by an upstream which is an invocation ahead. total_chunks
// must be exactly what upstream delivers (too many and a later reader passes early); the host gtest
// test_all_gather_schedule.cpp checks it for every device and direction.
//
// Waiting on an absolute position (not a signal count) lets one reader path cover every case with no
// alignment or per-topology special-casing:
//   - full relay, and even-ring split prefix half (skip 0) / suffix half (skip = half): same per-batch
//     wait, differing only in base_seqno/take;
//   - sink stripe (a line endpoint's incoming, or a ring antipode): no relay wait, covered by the final
//     total_chunks wait;
//   - sink direction (num_iters == 0): only the total_chunks wait runs.
// So data_valid_granularity is a pure writer-side perf knob: the reader auto-paces to the writer's cadence.
////////////////////////////////////////////////////////////////
void kernel_main() {
    namespace ct = gather_args::unicast_reader::ct;
    namespace rt = gather_args::unicast_reader::rt;

    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t in_cpp = get_compile_time_arg_val(ct::in_cpp);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(ct::chunk_size);
    constexpr uint32_t out_cpp = get_compile_time_arg_val(ct::out_cpp);
    constexpr uint32_t cps = get_compile_time_arg_val(ct::cps);
    constexpr uint32_t num_devices = get_compile_time_arg_val(ct::num_devices);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(ct::cb_id);
    constexpr uint32_t cb_entry_bytes = get_compile_time_arg_val(ct::cb_entry_bytes);
    constexpr uint32_t k = get_compile_time_arg_val(ct::k);
    constexpr uint32_t walk_stride = get_compile_time_arg_val(ct::walk_stride);
    constexpr uint32_t walk_xfer = get_compile_time_arg_val(ct::walk_xfer);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(ct::slot_bytes);
    constexpr RunSource in_runs = run_source_from_bits(get_compile_time_arg_val(ct::in_runs));
    constexpr RunSource out_runs = run_source_from_bits(get_compile_time_arg_val(ct::out_runs));
    constexpr auto input_tensor_args = TensorAccessorArgs<ct::tensor_args>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    check_walk_ct<k, in_cpp, out_cpp, cps, chunk_size, slot_bytes, cb_entry_bytes, walk_xfer, /*packet_size=*/0>();
    constexpr uint32_t chunks_per_entry = cb_entry_bytes / slot_bytes;

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    const address_t input_tensor_address = get_arg_val<address_t>(rt::input_addr);
    const address_t output_tensor_address = get_arg_val<address_t>(rt::output_addr);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(rt::initial_stripe);
    const uint32_t stripe_step = get_arg_val<uint32_t>(rt::stripe_step);
    const uint32_t num_iters = get_arg_val<uint32_t>(rt::num_iters);
    const uint32_t total_chunks = get_arg_val<uint32_t>(rt::total_chunks);
    const uint32_t slice_first = get_arg_val<uint32_t>(rt::slice_first);
    const uint32_t slice_count = get_arg_val<uint32_t>(rt::slice_count);
    const uint32_t final_skip = get_arg_val<uint32_t>(rt::final_skip);
    const uint32_t final_take = get_arg_val<uint32_t>(rt::final_take);
    const WalkPos final_pos = {
        get_arg_val<uint32_t>(rt::final_pos_base),
        get_arg_val<uint32_t>(rt::final_pos_lane),
        get_arg_val<uint32_t>(rt::final_pos_i)};
    const address_t barrier_sem = get_arg_val<uint32_t>(rt::barrier_sem);
    const address_t data_valid_sem = get_arg_val<uint32_t>(rt::data_valid_sem);

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);
    auto* data_valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_valid_sem);

    ///////////////////////////////////////////////////
    // RUN SETUP
    ///////////////////////////////////////////////////

    check_walk_stride(output_tensor_accessor, k, out_cpp, walk_stride);
    // Iteration 0 reads our input; later ones read back our output (the relay).
    const RunSource in_src = checked_run_source(in_runs, input_tensor_accessor, k, in_cpp, walk_stride);
    const RunSource out_src = checked_run_source(out_runs, output_tensor_accessor, k, out_cpp, walk_stride);
    const uint32_t slice_end = slice_first + slice_count;

    TiledWalk<k, walk_stride, walk_xfer> walk;
    StripeMap<cps, num_devices> map;

    auto input_addr = [&](uint32_t chunk) {
        return input_tensor_accessor.get_noc_addr(
            page_of<in_cpp>(chunk), byte_off_of<in_cpp, chunk_size>(chunk), noc.get_noc_id());
    };
    auto output_addr = [&](uint32_t page, uint32_t off) {
        return output_tensor_accessor.get_noc_addr(page, off, noc.get_noc_id());
    };

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // Startup barrier: wait for downstream remote device to be ready.
    // A sink direction (num_iters == 0) has no upstream here and is never signalled, so it must not wait.
    if (num_iters > 0) {
        auto* barrier_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem);
        noc_semaphore_wait_min(barrier_ptr, 1);
        noc_semaphore_set(barrier_ptr, 0);
    }

    uint32_t stripe = initial_stripe;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const bool last = (iter == num_iters - 1);
        const uint32_t skip = last ? final_skip : 0;
        const uint32_t take = last ? final_take : slice_count;
        // The walk does not stop itself: past the slice it would emit another worker's chunks.
        ASSERT(skip + take <= slice_count);
        const bool from_input = (iter == 0);
        // Where this read starts in the delivered-chunk stream. Iteration 0 reads local data, waits on nothing.
        const uint32_t base_seqno = from_input ? 0 : (iter - 1) * slice_count + skip;
        map.init(stripe, slice_end);
        // final_pos is the host's walk_seek(final_skip); every other iteration starts at seqno 0.
        walk.init(slice_end, last ? final_pos : walk_start(slice_first));

        for (uint32_t chunks_read = 0; chunks_read < take;) {
            const uint32_t batch = std::min(chunks_per_entry, take - chunks_read);
            if (!from_input) {
                noc_semaphore_wait_min(data_valid_ptr, base_seqno + chunks_read + batch);
            }

            cb.reserve_back(1);
            const uint32_t l1_write_addr = cb.get_write_ptr();
            if (from_input) {
                read_input_runs<in_cpp, chunk_size, slot_bytes>(
                    walk,
                    input_tensor_accessor,
                    in_src,
                    slice_end,
                    l1_write_addr,
                    batch,
                    input_addr,
                    [&](uint64_t src, uint32_t l1, uint32_t run) {
                        read_run<slot_bytes, chunk_size>(noc, src, l1, run);
                    });
            } else {
                // What upstream relayed into our output.
                emit_runs<out_cpp, chunk_size, slot_bytes>(
                    walk,
                    map,
                    output_tensor_accessor,
                    out_src,
                    l1_write_addr,
                    batch,
                    output_addr,
                    [&](uint32_t l1, uint32_t page, uint32_t off, uint32_t run) {
                        read_run<slot_bytes>(noc, output_addr(page, off), l1, run);
                    });
            }
            noc.async_read_barrier();
            cb.push_back(1);
            chunks_read += batch;
        }
        stripe = (stripe + stripe_step) % num_devices;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion: wait for every chunk upstream delivers (relayed + sink), then consume them.
    noc_semaphore_wait_min(data_valid_ptr, total_chunks);
    noc_semaphore_inc(get_noc_addr(data_valid_sem), uint32_t{0} - total_chunks);
    noc.async_atomic_barrier();
}
