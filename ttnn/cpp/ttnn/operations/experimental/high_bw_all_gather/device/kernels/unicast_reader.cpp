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
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/high_bw_all_gather_partition.hpp"

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
    constexpr auto input_tensor_args = TensorAccessorArgs<8>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    // Trace-safe slot select. The factory appends this block LAST, after both accessor blocks, so their
    // offsets stay stable. The flag is always pushed (0 on the scalar path); the two values and the
    // metadata accessor follow only when it is set.
    constexpr uint32_t meta_ct_base = output_tensor_args.next_compile_time_args_offset();
    constexpr bool batch_index_from_metadata = get_compile_time_arg_val(meta_ct_base) != 0;
    // Guard the INDEX, not the value: `cond ? get_compile_time_arg_val(N) : 0` still hard-fails with
    // "Index out of range" when arg N is absent, because the call is instantiated regardless of the
    // branch. Collapsing the base to 0 re-reads a known-present arg instead, and the values are unused
    // on the scalar path.
    constexpr uint32_t meta_scalar_base = batch_index_from_metadata ? meta_ct_base + 1 : 0;
    constexpr uint32_t pages_per_batch_slot = get_compile_time_arg_val(meta_scalar_base + 0);
    constexpr uint32_t cb_meta_id = get_compile_time_arg_val(meta_scalar_base + 1);
    constexpr auto batch_index_meta_args = TensorAccessorArgs<batch_index_from_metadata ? meta_ct_base + 3 : 0>();
    // Active-extent block. Same guarded-index convention as above.
    constexpr uint32_t ext_ct_base =
        batch_index_from_metadata ? batch_index_meta_args.next_compile_time_args_offset() : meta_ct_base + 1;
    constexpr bool extent_from_metadata = get_compile_time_arg_val(ext_ct_base) != 0;
    // The 14 scalars below are always pushed (zeros when unused), so no index guarding is needed here --
    // unlike the slot block, this one is too wide to collapse its base to 0 safely.
    constexpr uint32_t ext_base = ext_ct_base + 1;
    constexpr uint32_t ext_pages_per_slab = get_compile_time_arg_val(ext_base + 0);
    constexpr uint32_t ext_full_gathered_dim = get_compile_time_arg_val(ext_base + 1);
    constexpr uint32_t ext_slab_global = get_compile_time_arg_val(ext_base + 2);
    constexpr uint32_t ext_split_factor = get_compile_time_arg_val(ext_base + 3);
    constexpr uint32_t ext_total_slices = get_compile_time_arg_val(ext_base + 4);
    constexpr uint32_t ext_num_links = get_compile_time_arg_val(ext_base + 5);
    constexpr uint32_t ext_workers_per_dir = get_compile_time_arg_val(ext_base + 6);
    constexpr uint32_t ext_num_dram_banks = get_compile_time_arg_val(ext_base + 7);
    constexpr bool ext_bank_owned = get_compile_time_arg_val(ext_base + 8) != 0;
    constexpr uint32_t ext_slice_step = get_compile_time_arg_val(ext_base + 9);
    constexpr bool ext_ring_even_split = get_compile_time_arg_val(ext_base + 10) != 0;
    constexpr uint32_t ext_input_page_size = get_compile_time_arg_val(ext_base + 11);
    constexpr uint32_t ext_output_chunk_size = get_compile_time_arg_val(ext_base + 12);
    constexpr uint32_t ext_packet_size = get_compile_time_arg_val(ext_base + 13);
    constexpr uint32_t cb_meta_writer_id = get_compile_time_arg_val(ext_base + 14);
    constexpr auto gathered_prefix_meta_args = TensorAccessorArgs<extent_from_metadata ? ext_ct_base + 16 : 0>();

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
    const uint32_t batch_index_meta_addr = get_arg_val<uint32_t>(arg_idx++);
    // Layer-constant recomposition terms. Runtime rather than compile-time so every layer shares one
    // cached program (per-layer programs would allocate per-layer global semaphores and exhaust
    // L1_SMALL); a capture freezing them is fine because they do not vary per chunk or per request.
    const uint32_t batch_slot_num_layers = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t batch_slot_layer_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t gathered_prefix_meta_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ext_slice_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ext_link = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ext_worker = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ext_is_forward = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ext_num_recv = get_arg_val<uint32_t>(arg_idx++);

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    // Slot select. On the scalar path the host already folded the page base into input_page_id_start/end,
    // so this contributes 0. On the metadata path the host leaves the base at 0 and the base is derived
    // HERE from the slot id the tensor holds, because a trace replay never re-runs the host patch and
    // would otherwise re-read whichever slot was live at capture time -- silently, since the KV write is
    // metadata-driven and lands in the correct slot.
    //
    // The cache batch dim is user-major (ttMLA._cache_batch_idx), so the flat slot is
    //   user_id * batch_slot_num_layers + batch_slot_layer_idx
    // where the layer terms are RUNTIME args (see above): they are layer-constant, and making them
    // compile-time would fork the program per layer and exhaust L1_SMALL on the semaphores.
    uint32_t input_page_base = 0;
    if constexpr (batch_index_from_metadata) {
        CircularBuffer cb_meta(cb_meta_id);
        cb_meta.reserve_back(1);
        const uint32_t meta_l1 = cb_meta.get_write_ptr();
        const uint32_t user_id =
            trace_metadata::read_metadata_scalar_u32(noc, batch_index_meta_args, batch_index_meta_addr, meta_l1);
        input_page_base = (user_id * batch_slot_num_layers + batch_slot_layer_idx) * pages_per_batch_slot;
    }
    // Active-extent derivation. The host built this program for the WORST-CASE extent (so worker
    // selection, CB depth, schedule choice and semaphores are all fixed and capturable) and left every
    // N-dependent runtime value at that worst case. Narrow them here from the chunk start the metadata
    // tensor holds, using the same closed form the host uses, then publish the writer's share: reader and
    // writer share this core, and one derivation cannot drift from itself.
    uint32_t eff_total_chunks = total_chunks;
    uint32_t eff_slice_start = slice_start;
    uint32_t eff_slice_count = slice_count;
    uint32_t eff_final_start = final_start;
    uint32_t eff_final_count = final_count;
    uint32_t eff_page_start = input_page_id_start;
    uint32_t eff_page_end = input_page_id_end;
    if constexpr (extent_from_metadata) {
        namespace part = ttnn::operations::experimental::high_bw_all_gather::partition;
        CircularBuffer cb_writer_meta(cb_meta_writer_id);
        cb_writer_meta.reserve_back(1);
        const uint32_t writer_meta_l1 = cb_writer_meta.get_write_ptr();
        // Lands at the page base, is consumed immediately, then the five published words overwrite it --
        // so one CB serves as both the NoC landing slot and the reader->writer mailbox.
        const uint32_t prefix_start = trace_metadata::read_metadata_scalar_u32(
            noc, gathered_prefix_meta_args, gathered_prefix_meta_addr, writer_meta_l1);
        const uint32_t gathered =
            part::gathered_dim_size_for_prefix(prefix_start + ext_slab_global, ext_slab_global, ext_full_gathered_dim);
        const uint32_t active_pages = part::active_num_input_pages(gathered, ext_slab_global, ext_pages_per_slab);
        const auto sched = part::worker_schedule(
            active_pages,
            ext_split_factor,
            ext_total_slices,
            ext_slice_idx,
            ext_bank_owned,
            ext_num_links,
            ext_workers_per_dir,
            ext_num_dram_banks,
            ext_link,
            ext_worker,
            ext_slice_step,
            ext_ring_even_split,
            ext_is_forward != 0,
            ext_num_recv,
            ext_input_page_size,
            ext_output_chunk_size,
            ext_packet_size);
        eff_total_chunks = sched.total_chunks;
        eff_slice_start = sched.local_output_start;
        eff_slice_count = sched.slice_count;
        eff_final_start = sched.final_start;
        eff_final_count = sched.final_count;
        eff_page_start = sched.input_page_start;
        eff_page_end = sched.input_page_end;

        volatile tt_l1_ptr uint32_t* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_meta_l1);
        mailbox[0] = sched.local_output_start;
        mailbox[1] = sched.slice_count;
        mailbox[2] = sched.final_start;
        mailbox[3] = sched.final_count;
        mailbox[4] = sched.data_valid_granularity;
        cb_writer_meta.push_back(1);
    }

    const DynamicL1Semaphore ready_sem(ready_sem_addr);
    const DynamicL1Semaphore data_valid_sem(data_valid_sem_addr);

    OutputStripeIterator<
        output_chunks_per_page,
        output_chunk_size,
        num_devices,
        slice_step,
        static_output_chunks_per_stripe>
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
            uint32_t page = input_page_base + eff_page_start;
            const uint32_t page_end = input_page_base + eff_page_end;
            while (page < page_end) {
                cb.reserve_back(1);
                uint32_t l1_write_addr = cb.get_write_ptr();
                for (uint32_t i = 0; i < inputs_per_cb_page && page < page_end; ++i) {
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
            const uint32_t start = last ? eff_final_start : eff_slice_start;
            const uint32_t count = last ? eff_final_count : eff_slice_count;
            const uint32_t base_chunk = (iter - 1) * eff_slice_count + (start - eff_slice_start) / slice_step;
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
    data_valid_sem.wait_min(eff_total_chunks);
    data_valid_sem.consume(eff_total_chunks);
    noc.async_atomic_barrier();
}
