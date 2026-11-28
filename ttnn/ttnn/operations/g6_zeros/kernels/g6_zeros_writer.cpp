// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_zeros writer: the only kernel of the program (BRISC, NoC1).
//
// Each block is a run of `core_slots` same-bank slots starting at `first_page_id`. Slots of one
// DRAM bank are aligned_page_size apart, so a run is one contiguous byte range, written as
// <= zero_source_bytes NoC writes that all source the same core-local zero region.
//
// store_bank_run is a custom block operation over raw noc_async_write: the DRAM overload of
// Noc::async_write_zeros (noc_zero_dram.inl) zeroes at most one page per call, which would force
// one packet per page. The run loop below is that helper's burst loop applied to a whole bank run.
//
// cb_zero_source is a raw scratch region, not a FIFO: no reserve/push/wait/pop. A fresh CB has
// read_ptr == write_ptr, and the fill and every write use that one address.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_zero_source = get_compile_time_arg_val(0);
constexpr uint32_t aligned_page_size = get_compile_time_arg_val(1);
constexpr uint32_t zero_source_bytes = get_compile_time_arg_val(2);
constexpr uint32_t num_banks = get_compile_time_arg_val(3);
constexpr auto output_args = TensorAccessorArgs<4>();

// Runtime args: [num_blocks, (first_page_id, core_slots) x num_blocks].
constexpr uint32_t kBlockArgsBase = 1;
constexpr uint32_t kArgsPerBlock = 2;

FORCE_INLINE uint32_t block_first_page_id(uint32_t block_idx) {
    return get_arg_val<uint32_t>(kBlockArgsBase + block_idx * kArgsPerBlock);
}

FORCE_INLINE uint32_t block_core_slots(uint32_t block_idx) {
    return get_arg_val<uint32_t>(kBlockArgsBase + block_idx * kArgsPerBlock + 1);
}

template <typename Accessor>
FORCE_INLINE void store_bank_run(const Accessor& output, uint32_t zero_src_addr, uint32_t block_idx) {
    const uint32_t first_page_id = block_first_page_id(block_idx);
    const uint32_t core_slots = block_core_slots(block_idx);
    if (core_slots == 0) {
        return;
    }
    ASSERT(output.num_contiguous_pages(first_page_id, first_page_id + (core_slots - 1) * num_banks + 1) == core_slots);

    uint64_t dst = output.get_noc_addr(first_page_id);
    uint32_t remaining = core_slots * aligned_page_size;
    while (remaining > 0) {
        // A chunk never exceeds the zero region: with a larger size the NoC would advance the
        // source past it and stream stale L1 into DRAM.
        const uint32_t chunk = remaining < zero_source_bytes ? remaining : zero_source_bytes;
        noc_async_write<zero_source_bytes>(zero_src_addr, dst, chunk);
        dst += chunk;
        remaining -= chunk;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    const auto output = TensorAccessor(output_args, output_addr, aligned_page_size);
    ASSERT(output.contiguous_page_stride() == num_banks);

    Noc noc;
    CircularBuffer zero_source(cb_zero_source);
    const uint32_t zero_src_addr = zero_source.get_write_ptr();

    uint32_t max_run_bytes = 0;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t run_bytes = block_core_slots(block_idx) * aligned_page_size;
        max_run_bytes = run_bytes > max_run_bytes ? run_bytes : max_run_bytes;
    }
    const uint32_t fill_bytes = max_run_bytes < zero_source_bytes ? max_run_bytes : zero_source_bytes;

    {
        DeviceZoneScopedN("G6Z-FILL");
        noc.async_write_zeros(zero_source, fill_bytes);
    }
    {
        DeviceZoneScopedN("G6Z-FILL-WAIT");
        noc.write_zeros_l1_barrier();
    }
    {
        DeviceZoneScopedN("G6Z-STORE");
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            store_bank_run(output, zero_src_addr, block_idx);
        }
    }
    {
        DeviceZoneScopedN("G6Z-DRAIN");
        noc_async_write_barrier();
    }
}
