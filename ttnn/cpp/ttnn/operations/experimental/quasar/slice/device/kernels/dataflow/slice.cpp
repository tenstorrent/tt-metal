// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar slice: one data-movement kernel of up to 6 threads per cluster, no DataflowBuffer.
//
// Work model (host: slice_program_factory.cpp). The output is a sequence of pages (ROW_MAJOR: one row,
// TILE: one tile) and every page is cut into chunks_per_page chunks of chunk_bytes (the last one may be
// shorter). item = page * chunks_per_page + chunk. A cluster owns the contiguous item range
// [item_start, item_start + item_count) and thread t of T moves items item_start + t, + T, ... of it.
//
// Page mapping. The output page id is a mixed-radix number over the sliced dims that have more than one
// output index (extent[d], innermost last); the source page is
//     base_page + sum_d digit[d] * coef[d],   coef[d] = step[d] * (input pages per unit index of dim d),
// with the begins folded into base_page by the host.
//
// Staging. A thread owns `slots` slots of slot_bytes in the per-cluster scratchpad. Per batch it issues one
// NoC read per item into its slots, waits once, then issues one NoC write per item and waits for the writes
// to depart before the slots are reused. A page chunk is read at its byte offset inside the source page (the
// Quasar NoC takes unaligned addresses), so a last-dim begin needs no fix-up copy.
//
// SLICE_STRIDED_LAST_DIM: the last dim of a ROW_MAJOR slice has step > 1. A chunk is then elems output
// elements, read as the source span that covers them and compacted in place in L1 before the write.
// The compaction runs under a Scratchpad lock: the DM harts reach L1 through their cache, so the lock
// drops the stale lines before the loads and flushes the stores before the NoC write reads L1.
//
// SLICE_TENSOR_ARGS: the begins arrive in a device tensor (uint32, one page); each thread reads it once
// and adds the begin pages to base_page.
//
// Compile-time args: [elem_bytes]
// Common runtime args: base_page, chunks_per_page, chunk_bytes, last_chunk_bytes, src_offset_bytes, slots,
//                      slot_bytes, last_dim_step
// Common runtime array args: extent[], coef[] (one entry per page dim) [, start_div[], start_coef[] (one
//                            entry per input dim)]
// Runtime args (per cluster): item_start, item_count

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Row chunks are read and written at arbitrary byte offsets and sizes.
static_assert(
    NOC_L1_READ_ALIGNMENT_BYTES == 1 && NOC_L1_WRITE_ALIGNMENT_BYTES == 1 && NOC_DRAM_READ_ALIGNMENT_BYTES == 1 &&
        NOC_DRAM_WRITE_ALIGNMENT_BYTES == 1,
    "quasar slice needs a NoC without address alignment rules");

#ifdef SLICE_STRIDED_LAST_DIM
namespace {

// dst[e] = src[e * step] for e in [1, count); element 0 is already in place.
template <typename Elem>
FORCE_INLINE void compact(uint8_t* slot, uint32_t count, uint32_t step) {
    Elem* elems = reinterpret_cast<Elem*>(slot);
    for (uint32_t e = 1; e < count; ++e) {
        elems[e] = elems[e * step];
    }
}

}  // namespace
#endif

void kernel_main() {
    const uint32_t item_start = get_arg(args::item_start);
    const uint32_t item_count = get_arg(args::item_count);

    uint32_t base_page = get_arg(args::base_page);
    const uint32_t chunks_per_page = get_arg(args::chunks_per_page);
    const uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    const uint32_t last_chunk_bytes = get_arg(args::last_chunk_bytes);
    const uint32_t src_offset_bytes = get_arg(args::src_offset_bytes);
    const uint32_t slots = get_arg(args::slots);
    const uint32_t slot_bytes = get_arg(args::slot_bytes);

    const auto extent = get_arg(args::extent);
    const auto coef = get_arg(args::coef);

    const uint32_t thread = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    Noc noc;
    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);
    Scratchpad<uint8_t> stage(scratch::stage);
    const uint32_t region = thread * slots * slot_bytes;  // this thread's slots in the scratchpad

#ifdef SLICE_TENSOR_ARGS
    {
        const auto start_div = get_arg(args::start_div);
        const auto start_coef = get_arg(args::start_coef);
        constexpr uint32_t start_rank = start_div.size();
        const auto start = TensorAccessor(tensor::start);
        noc.async_read(start, stage, start_rank * sizeof(uint32_t), {.page_id = 0}, {.offset_bytes = region});
        noc.async_read_barrier();
        const auto lock = stage.scoped_lock(region, start_rank * sizeof(uint32_t));
        const uint32_t* begins = reinterpret_cast<const uint32_t*>(&stage[region]);
        for (uint32_t d = 0; d < start_rank; ++d) {
            base_page += (begins[d] / start_div[d]) * start_coef[d];
        }
    }
#endif

#ifdef SLICE_STRIDED_LAST_DIM
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    const uint32_t last_dim_step = get_arg(args::last_dim_step);
#endif

    struct Item {
        uint32_t page;        // output page
        uint32_t offset;      // byte offset of the chunk in the output page
        uint32_t bytes;       // bytes written
        uint32_t src_offset;  // byte offset of the chunk's source span in the source page
        uint32_t src_bytes;   // bytes read
    };
    auto item_of = [&](uint32_t i) __attribute__((always_inline)) {
        Item it;
        it.page = i / chunks_per_page;
        const uint32_t chunk = i - it.page * chunks_per_page;
        it.offset = chunk * chunk_bytes;
        it.bytes = chunk + 1 == chunks_per_page ? last_chunk_bytes : chunk_bytes;
#ifdef SLICE_STRIDED_LAST_DIM
        it.src_offset = src_offset_bytes + it.offset * last_dim_step;
        it.src_bytes = (it.bytes / elem_bytes - 1) * last_dim_step * elem_bytes + elem_bytes;
#else
        it.src_offset = src_offset_bytes + it.offset;
        it.src_bytes = it.bytes;
#endif
        return it;
    };
    auto src_page_of = [&](uint32_t page) __attribute__((always_inline)) {
        uint32_t src = base_page;
        for (uint32_t d = extent.size(); d-- > 0;) {
            const uint32_t q = page / extent[d];
            src += (page - q * extent[d]) * coef[d];
            page = q;
        }
        return src;
    };

    const uint32_t item_end = item_start + item_count;
    for (uint32_t batch = item_start + thread; batch < item_end; batch += slots * num_threads) {
        uint32_t n = 0;
        for (uint32_t i = batch; n < slots && i < item_end; ++n, i += num_threads) {
            const Item it = item_of(i);
            noc.async_read(
                input,
                stage,
                it.src_bytes,
                {.page_id = src_page_of(it.page), .offset_bytes = it.src_offset},
                {.offset_bytes = region + n * slot_bytes});
        }
        noc.async_read_barrier();

        uint32_t i = batch;
#ifdef SLICE_STRIDED_LAST_DIM
        for (uint32_t s = 0; s < n; ++s, i += num_threads) {
            const Item it = item_of(i);
            // Lock exactly the bytes read: the cache lines they touch hold only this slot and guard bytes.
            const auto lock = stage.scoped_lock(region + s * slot_bytes, it.src_bytes);
            uint8_t* slot = &stage[region + s * slot_bytes];
            const uint32_t count = it.bytes / elem_bytes;
            if constexpr (elem_bytes == 4) {
                compact<uint32_t>(slot, count, last_dim_step);
            } else if constexpr (elem_bytes == 2) {
                compact<uint16_t>(slot, count, last_dim_step);
            } else {
                compact<uint8_t>(slot, count, last_dim_step);
            }
        }
#endif

        i = batch;
        for (uint32_t s = 0; s < n; ++s, i += num_threads) {
            const Item it = item_of(i);
            noc.async_write(
                stage,
                output,
                it.bytes,
                {.offset_bytes = region + s * slot_bytes},
                {.page_id = it.page, .offset_bytes = it.offset});
        }
        noc.async_writes_flushed();
    }
    noc.async_write_barrier();
}
