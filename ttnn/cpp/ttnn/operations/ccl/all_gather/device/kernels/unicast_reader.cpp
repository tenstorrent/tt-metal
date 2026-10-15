// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"
#include "api/core_local_mem.h"

#include <cstdint>

#include "unicast_common.hpp"

using address_t = uint32_t;

// Store-and-forward reader: CB producer, no fabric. It owns every data_valid wait (see the protocol note in
// unicast_common.hpp). Iteration 0 fills the CB from this device's local data; later iterations relay the
// stripe upstream delivered into our output, gated on data_valid.
//
// Both cases walk the same chunks in the same order the writer will drain them; only the source differs.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t split_factor = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t packet_size = get_compile_time_arg_val(7);
    constexpr uint32_t run_cap_bytes = get_compile_time_arg_val(8);  // longest run the walk may emit; 0 = no cap
    constexpr auto input_tensor_args = TensorAccessorArgs<9>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t chunks_per_cb_entry = cb_page_size / output_chunk_size;
    constexpr uint32_t xfer_max = chunks_per_transfer(packet_size, output_chunk_size, run_cap_bytes);

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t total_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_first_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_skip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_take = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);
    auto* data_valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_valid_sem);

    ///////////////////////////////////////////////////
    // RUN SETUP
    ///////////////////////////////////////////////////

    const auto plan = walk_plan<output_chunks_per_page, output_chunk_size, xfer_max>(output_tensor_accessor);
    // Iteration 0 reads our input, which yields runs only when it strides the way the walk does.
    const auto in_src = run_source(
        input_tensor_accessor.get_aligned_page_size() == split_factor * output_chunk_size,
        split_factor,
        input_tensor_accessor.contiguous_page_stride(),
        plan.stride);
    const uint32_t input_end_chunk = slice_first_chunk + slice_chunks;

    TiledWalk walk;
    StripeMap<output_chunks_per_stripe, num_devices> map;

    auto input_addr = [&](uint32_t chunk) {
        return input_tensor_accessor.get_noc_addr(
            page_of<split_factor>(chunk), byte_off_of<split_factor, output_chunk_size>(chunk), noc.get_noc_id());
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
        const uint32_t take = last ? final_take : slice_chunks;
        // The walk does not stop itself: past the slice it would emit another worker's chunks.
        ASSERT(skip + take <= slice_chunks);
        const bool from_input = (iter == 0);
        // Where this read starts in the delivered-chunk stream. Iteration 0 reads local data, waits on nothing.
        const uint32_t base_seqno = from_input ? 0 : (iter - 1) * slice_chunks + skip;
        map.init(stripe);
        walk.init(slice_first_chunk, slice_chunks, skip, plan.stride, plan.xfer);

        for (uint32_t chunks_read = 0; chunks_read < take;) {
            const uint32_t batch = std::min(chunks_per_cb_entry, take - chunks_read);
            if (!from_input) {
                noc_semaphore_wait_min(data_valid_ptr, base_seqno + chunks_read + batch);
            }

            cb.reserve_back(1);
            const uint32_t l1_write_addr = cb.get_write_ptr();
            if (from_input) {
                read_input_runs<split_factor, output_chunk_size>(
                    walk,
                    input_tensor_accessor,
                    in_src,
                    input_end_chunk,
                    l1_write_addr,
                    batch,
                    input_addr,
                    [&](uint64_t src, uint32_t l1, uint32_t run) { read_run<output_chunk_size>(noc, src, l1, run); });
            } else {
                // What upstream relayed into our output.
                emit_runs<output_chunks_per_page, output_chunk_size>(
                    walk,
                    map,
                    output_tensor_accessor,
                    plan.out,
                    l1_write_addr,
                    batch,
                    output_addr,
                    [&](uint32_t l1, uint32_t page, uint32_t off, uint32_t run) {
                        read_run<output_chunk_size>(noc, output_addr(page, off), l1, run);
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
