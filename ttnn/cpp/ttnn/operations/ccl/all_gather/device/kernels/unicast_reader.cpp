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
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>

#include "chunk_walk.hpp"
#include "concat.hpp"
#include "unicast_common.hpp"

using address_t = uint32_t;

// Store-and-forward reader: CB producer, touches no fabric. Hop 0 reads our own input; later hops read
// what upstream relayed into our output and hand it to the writer to send on. Owns the init-barrier wait
// and the completion wait (see unicast_common.hpp).
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////

    // --- moving chunks ---
    constexpr uint32_t chunk_size = get_compile_time_arg_val(0);
    constexpr uint32_t in_chunks_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t out_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t payload = get_compile_time_arg_val(3);        // bytes a packet may carry
    constexpr uint32_t asked_run_max = get_compile_time_arg_val(4);  // chunks; 0 = the whole payload
    constexpr uint32_t entry_chunks = get_compile_time_arg_val(5);
    // --- all_gather ---
    constexpr uint32_t stripe = get_compile_time_arg_val(6);
    constexpr uint32_t num_devices = get_compile_time_arg_val(7);
    // --- this kernel ---
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr bool do_init_barrier = get_compile_time_arg_val(9) != 0;
    constexpr auto input_tensor_args = TensorAccessorArgs<10>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t payload_chunks = payload / chunk_size > 0 ? payload / chunk_size : 1;
    constexpr uint32_t run_max_want = run_max_capped(asked_run_max, payload_chunks, chunk_size);

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t total_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_first_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_skip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_take = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb_id);
    auto* data_valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_valid_sem);

    ///////////////////////////////////////////////////
    // SETUP
    ///////////////////////////////////////////////////

    // The walk order follows the output, on every hop: hop 0 reads the input in that order too, and the
    // input yields runs only where it happens to stride the same way.
    const bool out_packed = packed_pages(output_tensor_accessor, out_chunks_per_page, chunk_size);
    const uint32_t bank_step = bank_step_of(output_tensor_accessor, out_chunks_per_page);
    const uint32_t run_max = out_packed ? run_max_want : 1u;
    const bool out_in_page = join_in_page(out_packed, out_chunks_per_page, bank_step);
    const bool out_across_pages =
        join_pages(out_packed, out_chunks_per_page, output_tensor_accessor.contiguous_page_stride(), bank_step);

    const bool in_packed = packed_pages(input_tensor_accessor, in_chunks_per_page, chunk_size);
    const bool in_in_page = join_in_page(in_packed, in_chunks_per_page, bank_step);
    const bool in_across_pages =
        join_pages(in_packed, in_chunks_per_page, input_tensor_accessor.contiguous_page_stride(), bank_step);

    const uint32_t input_end_chunk = slice_first_chunk + slice_chunks;

    Walk walk;

    auto in_addr = [&](uint32_t chunk) {
        return input_tensor_accessor.get_noc_addr(
            page_of<in_chunks_per_page>(chunk), byte_off<in_chunks_per_page, chunk_size>(chunk), noc.get_noc_id());
    };
    auto out_addr = [&](uint32_t out) {
        return output_tensor_accessor.get_noc_addr(
            page_of<out_chunks_per_page>(out), byte_off<out_chunks_per_page, chunk_size>(out), noc.get_noc_id());
    };
    // Address of one of our chunks. Named, not inline: an ASSERT argument is unevaluated, and a
    // lambda cannot appear there.
    uint32_t stripe_base = 0;
    auto run_addr = [&](uint32_t ours) { return out_addr(out_chunk<stripe, num_devices>(ours, stripe_base)); };

    auto read_run = [&](uint64_t src, uint32_t l1_write_addr, uint32_t chunks) {
        if constexpr (chunk_fits_command(chunk_size)) {
            noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                tensor_accessor::Page(src, 0), CoreLocalMem<uint32_t>(l1_write_addr), chunks * chunk_size, {}, {}, {});
        } else {
            noc.async_read(
                tensor_accessor::Page(src, 0), CoreLocalMem<uint32_t>(l1_write_addr), chunk_size, {}, {}, {});
        }
    };

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // Startup barrier: wait for downstream remote device to be ready.
    // A sink direction (num_hops == 0) has no upstream here and is never signalled, so it must not wait.
    if constexpr (do_init_barrier) {
        if (num_hops > 0) {
            auto* barrier_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem);
            noc_semaphore_wait_min(barrier_ptr, 1);
            noc_semaphore_set(barrier_ptr, 0);
        }
    }

    uint32_t stripe_idx = initial_stripe;
    for (uint32_t hop = 0; hop < num_hops; ++hop) {
        const bool last = (hop == num_hops - 1);
        const uint32_t skip = last ? final_skip : 0;
        const uint32_t take = last ? final_take : slice_chunks;
        // The walk does not stop itself: past the slice it would emit another worker's chunks.
        ASSERT(skip + take <= slice_chunks);
        const bool from_input = (hop == 0);
        // Where this read starts in the delivered-chunk stream. Hop 0 reads local data, waits on nothing.
        const uint32_t base = from_input ? 0 : (hop - 1) * slice_chunks + skip;
        stripe_base = stripe_idx * stripe;
        walk.init(slice_first_chunk, slice_chunks, skip, bank_step, run_max);

        for (uint32_t chunks_read = 0; chunks_read < take;) {
            const uint32_t entry = std::min(entry_chunks, take - chunks_read);
            if (!from_input) {
                noc_semaphore_wait_min(data_valid_ptr, base + chunks_read + entry);
            }

            cb.reserve_back(1);
            uint32_t l1_write_addr = cb.get_write_ptr();
            for (uint32_t left = entry; left > 0;) {
                const uint32_t ours = walk.chunk();
                const uint32_t limit = std::min(left, walk.lane_room());
                uint64_t src;
                uint32_t run;
                if (from_input) {
                    src = in_addr(ours);
                    run = run_length<in_chunks_per_page>(
                        input_tensor_accessor, in_in_page, in_across_pages, ours, input_end_chunk, limit);
                    ASSERT(run_is_linear(walk, run, chunk_size, src, in_addr));
                } else {
                    // What upstream relayed into our output.
                    const uint32_t out = out_chunk<stripe, num_devices>(ours, stripe_base);
                    const uint32_t room = row_room<stripe>(ours);
                    src = out_addr(out);
                    run = run_length<out_chunks_per_page>(
                        output_tensor_accessor, out_in_page, out_across_pages, out, out + room, limit);
                    ASSERT(run_is_linear(walk, run, chunk_size, src, run_addr));
                }
                read_run(src, l1_write_addr, run);
                l1_write_addr += run * chunk_size;
                left -= run;
                walk.advance(run);
            }
            noc.async_read_barrier();
            cb.push_back(1);
            chunks_read += entry;
        }
        stripe_idx = (stripe_idx + stripe_step) % num_devices;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion: wait for every chunk upstream delivers (relayed + sink), then consume them.
    noc_semaphore_wait_min(data_valid_ptr, total_chunks);
    noc_semaphore_inc(get_noc_addr(data_valid_sem), uint32_t{0} - total_chunks);
    noc.async_atomic_barrier();
}
