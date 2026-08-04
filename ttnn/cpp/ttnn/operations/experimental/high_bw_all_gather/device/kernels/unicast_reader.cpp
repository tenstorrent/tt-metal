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
#include <utility>

#include "unicast_common.hpp"

using address_t = uint32_t;

// Device 2.0 Semaphore only constructs from a fixed semaphore ID. Global
// semaphores are allocated dynamically and arrive as an L1 address, for which
// there is no Device 2.0 address-based constructor. Keep this narrow wrapper
// operation-local until that API exists.
class DynamicL1Semaphore {
public:
    explicit DynamicL1Semaphore(address_t l1_addr) : l1_addr_(l1_addr) {}

    void wait_min(uint32_t value) const {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr_), value);
    }

    void consume(uint32_t value) const { noc_semaphore_inc(get_noc_addr(l1_addr_), uint32_t{0} - value); }

private:
    address_t l1_addr_;
};

// Store-and-forward reader: CB producer, no fabric. It owns every data_valid wait (see the protocol note in
// unicast_common.hpp). Iteration 0 fills the CB from this device's local data; later iterations relay the
// stripe upstream delivered into our output, gated on data_valid.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t output_chunk_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_chunks_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t slice_step = get_compile_time_arg_val(6);
    // The maximum per-rank output slot is structural even for selected-prefix gathers, so its
    // stripe width stays baked and keeps the iterator arithmetic constexpr.
    constexpr uint32_t static_output_chunks_per_stripe = get_compile_time_arg_val(7);
    constexpr bool linearized_mesh_ring = get_compile_time_arg_val(8) != 0;
    constexpr auto snake_orientation =
        static_cast<ttnn::operations::experimental::high_bw_all_gather::snake_ring::Orientation>(
            get_compile_time_arg_val(9));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(10);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(11);
    constexpr auto input_tensor_args = TensorAccessorArgs<12>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t inputs_per_cb_page = cb_page_size / input_page_size;
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
    const uint32_t final_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const address_t ready_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const address_t data_valid_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_chunks_per_stripe = get_arg_val<uint32_t>(arg_idx++);

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);
    const DynamicL1Semaphore ready_sem(ready_sem_addr);
    const DynamicL1Semaphore data_valid_sem(data_valid_sem_addr);

    OutputStripeIterator<
        output_chunks_per_page,
        output_chunk_size,
        num_devices,
        slice_step,
        static_output_chunks_per_stripe,
        linearized_mesh_ring,
        snake_orientation,
        mesh_rows,
        mesh_cols>
        it;

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // A remote writer can reach this device before this device has completed
    // earlier input/output transfers on its own command queue. Gate the local
    // writer's CB producer until the downstream device announces that its
    // corresponding collective program has started. Each invocation owns and
    // consumes one readiness credit.
    if (num_iters > 0) {
        ready_sem.wait_min(1);
        ready_sem.consume(1);
    }

    uint32_t stripe = initial_stripe;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        if (iter == 0) {
            // Local data (our own input tensor)
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
                    page += slice_step;
                }
                noc.async_read_barrier();
                cb.push_back(1);
            }
        } else {
            // Relay: read the stripe upstream delivered into our output, waiting per CB-page batch for its
            // chunks to arrive. base_chunk is where this read begins in the delivered-chunk stream: 0 for a full
            // stripe or an even-ring prefix half, `half` for a suffix half.
            const bool last = (iter == num_iters - 1);
            const uint32_t start = last ? final_start : slice_start;
            const uint32_t count = last ? final_count : slice_count;
            const uint32_t base_chunk = (iter - 1) * slice_count + (start - slice_start) / slice_step;
            it.init(stripe, start, count, output_chunks_per_stripe);
            for (uint32_t chunks_read = 0; chunks_read < count;) {
                const uint32_t batch = std::min(outputs_per_cb_page, count - chunks_read);
                data_valid_sem.wait_min(base_chunk + chunks_read + batch);

                cb.reserve_back(1);
                uint32_t l1_write_addr = cb.get_write_ptr();
                if constexpr (slice_step > 1 && outputs_per_cb_page > 1) {
                    // In a bank-owned schedule, stepping by the DRAM-bank count advances to the next physically
                    // contiguous chunk in the same bank. Read the complete CB batch with one NOC transaction.
                    auto [first_page_id, first_byte_off] = it.next();
                    const uint64_t first_noc_addr =
                        output_tensor_accessor.get_noc_addr(first_page_id, first_byte_off, noc.get_noc_id());
                    for (uint32_t i = 1; i < batch; ++i) {
                        (void)it.next();
                    }
                    noc.async_read(
                        tensor_accessor::Page(first_noc_addr, 0),
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        batch * output_chunk_size,
                        {},
                        {});
                } else {
                    for (uint32_t i = 0; i < batch; ++i) {
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
                }
                noc.async_read_barrier();
                cb.push_back(1);
                chunks_read += batch;
            }
        }
        stripe = (stripe + stripe_step) % num_devices;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion: wait for every chunk upstream delivers (relayed + sink),
    // then atomically consume this invocation's credits.
    data_valid_sem.wait_min(total_chunks);
    data_valid_sem.consume(total_chunks);
    noc.async_atomic_barrier();
}
