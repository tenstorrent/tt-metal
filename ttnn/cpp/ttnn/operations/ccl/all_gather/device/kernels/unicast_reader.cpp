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
    constexpr bool do_init_barrier = get_compile_time_arg_val(7) != 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<8>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr bool concat = output_chunks_per_page > 1;
    constexpr uint32_t chunks_per_cb_entry = cb_page_size / output_chunk_size;
    // One NOC command per run needs the run to fit a burst; a chunk bigger than that takes the generic path.
    constexpr bool burst_runs = output_chunk_size <= NOC_MAX_BURST_SIZE;
    constexpr uint32_t max_burst_chunks = burst_runs ? NOC_MAX_BURST_SIZE / output_chunk_size : 1;

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
    [[maybe_unused]] const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);  // used only if do_init_barrier
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);
    auto* data_valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_valid_sem);

    ///////////////////////////////////////////////////
    // RUN SETUP
    ///////////////////////////////////////////////////

    // A run steps by the accessor's own stride; any other step lands in a different bank or shard. A stride
    // longer than a stripe can never take a second step, so that falls back to plain page order.
    // `packed` guards the transfer size: runs step by the aligned page size, but the CB is packed.
    const uint32_t out_page_stride = output_tensor_accessor.contiguous_page_stride();
    const bool out_packed =
        output_tensor_accessor.get_aligned_page_size() == output_chunks_per_page * output_chunk_size;
    const bool out_page_runs =
        out_packed && (concat ? out_page_stride == 1 : out_page_stride <= output_chunks_per_stripe);
    // Concat packs several chunks into an output page, so its walk has to stay in page order.
    const uint32_t stride = (concat || !out_page_runs) ? 1u : out_page_stride;

    const bool in_packed = input_tensor_accessor.get_aligned_page_size() == split_factor * output_chunk_size;
    const bool in_page_runs = in_packed && input_tensor_accessor.contiguous_page_stride() == 1;
    const uint32_t input_end_page = (slice_first_chunk + slice_chunks) / split_factor;

    StripeWalk<output_chunks_per_stripe, output_chunks_per_page, output_chunk_size, num_devices> it;

    auto output_run = [&]() -> uint32_t {
        if constexpr (concat) {
            // Chunks are packed inside an output page, so the intra-page run is always available.
            uint32_t n = output_chunks_per_page - it.byte_off() / output_chunk_size;
            if (out_page_runs) {
                n += (output_tensor_accessor.num_contiguous_pages(it.page_id(), it.end_page_id()) - 1) *
                     output_chunks_per_page;
            }
            return it.seqnos_in_chunk_ids(n);
        } else {
            // num_contiguous_pages already steps by the walk's stride, so this is a seqno count.
            return out_page_runs ? output_tensor_accessor.num_contiguous_pages(it.page_id(), it.end_page_id()) : 1u;
        }
    };

    auto input_run = [&](uint32_t page) -> uint32_t {
        uint32_t n = split_factor - it.chunk_id() % split_factor;  // rest of this input page
        if (in_page_runs) {
            n += (input_tensor_accessor.num_contiguous_pages(page, input_end_page) - 1) * split_factor;
        }
        return it.seqnos_in_chunk_ids(n);
    };

    // Debug-only: a run has to be linear. Checks page/offset truth, which is what a run claims.
    auto run_is_linear = [&](uint32_t chunks, bool input_src) {
        auto probe = it;
        auto addr = [&] {
            return input_src ? input_tensor_accessor.get_noc_addr(
                                   probe.chunk_id() / split_factor,
                                   (probe.chunk_id() % split_factor) * output_chunk_size,
                                   noc.get_noc_id())
                             : output_tensor_accessor.get_noc_addr(probe.page_id(), probe.byte_off(), noc.get_noc_id());
        };
        const uint64_t first = addr();
        for (uint32_t k = 0; k < chunks; ++k) {
            if (addr() != first + k * output_chunk_size) {
                return false;
            }
            probe.advance(1);
        }
        return true;
    };

    auto read_run = [&](uint64_t src, uint32_t l1_write_addr, uint32_t chunks) {
        if constexpr (burst_runs) {
            noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                tensor_accessor::Page(src, 0),
                CoreLocalMem<uint32_t>(l1_write_addr),
                chunks * output_chunk_size,
                {},
                {},
                {});
        } else {
            noc.async_read(
                tensor_accessor::Page(src, 0), CoreLocalMem<uint32_t>(l1_write_addr), output_chunk_size, {}, {}, {});
        }
    };

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
        const uint32_t take = last ? final_take : slice_chunks;
        const bool from_input = (iter == 0);
        // Where this read starts in the delivered-chunk stream. Iteration 0 reads local data, waits on nothing.
        const uint32_t base_seqno = from_input ? 0 : (iter - 1) * slice_chunks + skip;
        it.init(stripe, slice_first_chunk, slice_chunks, skip, take, stride);

        for (uint32_t chunks_read = 0; chunks_read < take;) {
            const uint32_t batch = std::min(chunks_per_cb_entry, take - chunks_read);
            if (!from_input) {
                noc_semaphore_wait_min(data_valid_ptr, base_seqno + chunks_read + batch);
            }

            cb.reserve_back(1);
            uint32_t l1_write_addr = cb.get_write_ptr();
            for (uint32_t left = batch; left > 0;) {
                uint64_t src;
                uint32_t chunks;
                if (from_input) {
                    // Our own input, where chunk c is input page c / split_factor.
                    const uint32_t page = it.chunk_id() / split_factor;
                    src = input_tensor_accessor.get_noc_addr(
                        page, (it.chunk_id() % split_factor) * output_chunk_size, noc.get_noc_id());
                    chunks = input_run(page);
                } else {
                    // What upstream relayed into our output.
                    src = output_tensor_accessor.get_noc_addr(it.page_id(), it.byte_off(), noc.get_noc_id());
                    chunks = output_run();
                }
                chunks = std::min(std::min(chunks, left), max_burst_chunks);
                ASSERT(run_is_linear(chunks, from_input));
                read_run(src, l1_write_addr, chunks);
                l1_write_addr += chunks * output_chunk_size;
                left -= chunks;
                it.advance(chunks);
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

    // Completion: wait for every chunk upstream delivers (relayed + sink), then reset for reuse.
    noc_semaphore_wait_min(data_valid_ptr, total_chunks);
    noc_semaphore_set(data_valid_ptr, 0);
}
