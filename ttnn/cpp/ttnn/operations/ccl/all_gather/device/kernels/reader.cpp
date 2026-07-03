// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

#include <cstdint>
#include <utility>

#include "common.hpp"

using address_t = uint32_t;

// Store-and-forward reader: CB producer, no fabric. It owns every semaphore wait. Each relay iteration fills
// the CB with the stripe the writer will send one hop: iteration 0 reads this device's own input slice; later
// iterations read the stripe the upstream neighbor deposited into our output.
void kernel_main() {
    constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t output_chunks_per_stripe = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(6);
    constexpr bool do_init_barrier = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t data_valid_granularity = 320;  // get_compile_time_arg_val(8);
    constexpr auto input_tensor_args = TensorAccessorArgs<9>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t inputs_per_cb_page = cb_page_size / input_page_size;
    constexpr uint32_t outputs_per_cb_page = cb_page_size / output_chunk_size;

    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t initial_stripe = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_recv = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const address_t ready_sem = get_arg_val<uint32_t>(arg_idx++);  // used only if do_init_barrier

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);
    auto* data_valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_valid_sem);

    OutputStripeIterator<output_chunks_per_stripe, output_chunks_per_page, output_chunk_size, num_devices> it;

    // A relayed stripe is signalled once per data_valid_granularity CB pages; a stripe we only receive (the
    // antipode) arrives as a single inc. incs tracks how many we've consumed so the completion wait is exact.
    const uint32_t packets_per_stripe = (slice_count + outputs_per_cb_page - 1) / outputs_per_cb_page;
    const uint32_t incs_per_stripe = (packets_per_stripe + data_valid_granularity - 1) / data_valid_granularity;
    uint32_t incs = 0;

    uint32_t stripe = initial_stripe;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        if (iter == 0) {
            // Gate the first CB fill (and thus the writer's first remote write) until the downstream is up.
            if constexpr (do_init_barrier) {
                auto* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem);
                noc_semaphore_wait_min(ready_ptr, 1);
                noc_semaphore_set(ready_ptr, 0);
            }
            // Own shard: contiguous input pages. Byte count matches the writer's output-chunk slice.
            uint32_t page = input_page_id_start;
            while (page < input_page_id_end) {
                cb.reserve_back(1);
                uint32_t l1_write_addr = cb.get_write_ptr();
                for (uint32_t i = 0; i < inputs_per_cb_page && page < input_page_id_end; ++i) {
                    noc.async_read(
                        input_tensor_accessor,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        input_page_size,
                        {.page_id = page},
                        {},
                        {});
                    l1_write_addr += input_page_size;
                    ++page;
                }
                noc.async_read_barrier();
                cb.push_back(1);
            }
        } else {
            // Relay: the upstream neighbor must have delivered this stripe into our output first.
            const bool last = (iter == num_iters - 1);
            // An even-ring last relay reads a sub-slice of a stripe the upstream sent as one granular stripe,
            // so its granule boundaries don't line up -- wait for the whole stripe, then read.
            const bool coarse_wait = last && (final_count != slice_count);
            it.init(stripe, last ? final_start : slice_start, last ? final_count : slice_count);
            if (coarse_wait) {
                incs += incs_per_stripe;
                noc_semaphore_wait_min(data_valid_ptr, incs);
            }
            uint32_t packets_since_wait = 0;
            while (it.valid()) {
                if (!coarse_wait && packets_since_wait == 0) {
                    {
                        DeviceZoneScopedN("relay_wait_sem");
                        noc_semaphore_wait_min(data_valid_ptr, ++incs);
                    }
                }
                {
                    DeviceZoneScopedN("relay_wait_cb");
                    cb.reserve_back(1);
                }
                uint32_t l1_write_addr = cb.get_write_ptr();
                for (uint32_t i = 0; i < outputs_per_cb_page && it.valid(); ++i) {
                    auto [page_id, byte_off] = it.next();
                    noc.async_read(
                        output_tensor_accessor,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        output_chunk_size,
                        {.page_id = page_id, .offset_bytes = byte_off},
                        {},
                        {});
                    l1_write_addr += output_chunk_size;
                }
                {
                    DeviceZoneScopedN("relay_wait_read");
                    noc.async_read_barrier();
                }
                cb.push_back(1);
                packets_since_wait = (packets_since_wait + 1 == data_valid_granularity) ? 0 : packets_since_wait + 1;
            }
        }
        stripe = (stripe + stripe_step) % num_devices;
    }

    // Completion: wait for every inc we're owed, then reset for cached reuse. We forward (num_iters - 1)
    // stripes -- each incs_per_stripe increments -- and receive but never forward the rest (num_recv beyond
    // that), each arriving as a single inc.
    const uint32_t relayed = (num_iters > 0) ? (num_iters - 1) : 0;
    const uint32_t total_incs = relayed * incs_per_stripe + (num_recv - relayed);
    noc_semaphore_wait_min(data_valid_ptr, total_incs);
    noc_semaphore_set(data_valid_ptr, 0);
}
