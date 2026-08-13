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
// Both cases walk the same chunks in the same order the writer will drain them; only where a chunk is read
// from differs. Reads go through the same RunCoalescer, so contiguous ones merge: on a relay that is the
// writer's runs exactly, and on iteration 0 a split input page reassembles into the single page read it
// always was.
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
    constexpr bool do_init_barrier = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t walk_stride = get_compile_time_arg_val(8);
    constexpr bool merge_runs = get_compile_time_arg_val(9) != 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<10>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t outputs_per_cb_page = cb_page_size / output_chunk_size;

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
    const uint32_t slice_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_skip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_take = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);  // used only if do_init_barrier
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);
    auto* data_valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_valid_sem);

    OutputStripeIterator<
        output_chunks_per_stripe,
        output_chunks_per_page,
        output_chunk_size,
        num_devices,
        walk_stride>
        it;

    auto to_cb = [&](uint32_t l1_addr, uint64_t src, uint32_t bytes) {
        // The coalescer caps a run at NOC_MAX_BURST_SIZE, so the one-packet path always applies.
        noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            tensor_accessor::Page(src, 0), CoreLocalMem<uint32_t>(l1_addr), bytes, {}, {}, {});
    };
    RunCoalescer<NOC_MAX_BURST_SIZE, merge_runs, decltype(to_cb)> read_runs{to_cb};

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // Startup barrier: wait for downstream remote device to be ready.
    // A sink direction (num_iters == 0) has no upstream here and is never signalled, so it must not wait.
    if constexpr (do_init_barrier) {
        if (num_iters > 0) {
            auto* barrier_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem);
            noc_semaphore_wait_min(barrier_ptr, 1);
            noc_semaphore_set(barrier_ptr, 0);
        }
    }

    uint32_t stripe = initial_stripe;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const bool last = (iter == num_iters - 1);
        const uint32_t skip = last ? final_skip : 0;
        const uint32_t count = last ? final_take : slice_count;
        const bool from_input = (iter == 0);
        // Where this read begins in the delivered-chunk stream: 0 for a full stripe or an even-ring
        // prefix half, `half` for a suffix half. Iteration 0 reads local data and waits on nothing.
        const uint32_t base_chunk = from_input ? 0 : (iter - 1) * slice_count + skip;
        it.init(stripe, slice_start, slice_count, skip, count);

        for (uint32_t chunks_read = 0; chunks_read < count;) {
            const uint32_t batch = std::min(outputs_per_cb_page, count - chunks_read);
            if (!from_input) {
                noc_semaphore_wait_min(data_valid_ptr, base_chunk + chunks_read + batch);
            }

            cb.reserve_back(1);
            uint32_t l1_write_addr = cb.get_write_ptr();
            // from_input is fixed for the whole iteration, so resolve it outside the inner loop.
            auto fill = [&](auto source_of) {
                for (uint32_t i = 0; i < batch; ++i) {
                    const auto chunk = it.next();
                    read_runs.add(l1_write_addr, source_of(chunk), output_chunk_size);
                    l1_write_addr += output_chunk_size;
                }
            };
            if (from_input) {
                // Our own input, where chunk c is input page c / split_factor.
                fill([&](const auto& chunk) {
                    return input_tensor_accessor.get_noc_addr(
                        chunk.index / split_factor,
                        (chunk.index % split_factor) * output_chunk_size,
                        noc.get_noc_id());
                });
            } else {
                // What upstream relayed into our output.
                fill([&](const auto& chunk) {
                    return output_tensor_accessor.get_noc_addr(chunk.page_id, chunk.byte_off, noc.get_noc_id());
                });
            }
            read_runs.flush();
            noc.async_read_barrier();
            cb.push_back(1);
            chunks_read += batch;
        }
        stripe = (stripe + stripe_step) % num_devices;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion: wait for every chunk upstream delivers (relayed + sink), then reset for reuse.
    noc_semaphore_wait_min(data_valid_ptr, total_chunks);
    noc_semaphore_set(data_valid_ptr, 0);
}
