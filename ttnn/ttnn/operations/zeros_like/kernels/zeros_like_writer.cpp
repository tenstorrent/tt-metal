// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// zeros_like writer (op_design.md, regime `fill_from_local_zero_page`). The only kernel of the program: there
// is no reader (the input is never read, spec R1) and no compute kernel (zero bits need no FPU/SFPU).
//
// Block schedule, per writer thread t of W:
//   zero_source()   iDMA-zero this thread's `zero_src_bytes` slice of the zero_page scratchpad, once per kernel
//   store_block()   for each block of `pages_per_barrier` pages of this core's range: write pages t, t+W, ...
//                   of the block from the slice (chunks of zero_src_bytes), all tagged with user trid 0, then
//                   ONE trid write barrier per block
//   release_tag()   hand the hart's tag register back with user id 0
//
// Raw NoC API, no kernel_lib helper (design, "Helpers considered and rejected"):
//  - dataflow_kernel_lib::zero_tile / prepare_zero_tile size the zero from a DataflowBuffer's tile size, which a
//    Scratchpad does not have, and a ring would need a second kernel as its other endpoint. This kernel calls
//    the same two primitives those helpers wrap (noc.async_write_zeros + noc.write_zeros_l1_barrier).
//  - Noc::async_write_zeros(TensorAccessor, ..., scratch) issues UNTAGGED writes; it also
//    static_asserts a DRAM accessor, so the L1 TARGET value would need a second path. Every write here is
//    tagged with trid 0 instead.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t num_pages_this_core = get_arg(args::num_pages_this_core);

    constexpr uint32_t writer_threads = get_arg(args::writer_threads);
    constexpr uint32_t pages_per_barrier = get_arg(args::pages_per_barrier);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t zero_src_bytes = get_arg(args::zero_src_bytes);
    constexpr uint32_t chunks_per_page = (page_bytes + zero_src_bytes - 1) / zero_src_bytes;
    static_assert(writer_threads >= 1 && pages_per_barrier >= 1 && zero_src_bytes >= 1);

    constexpr NocOptVals user_txn = {.trid = 0};

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t slice_offset = thread_id * zero_src_bytes;  // this thread's slice of the scratchpad

    Noc noc;
    Scratchpad<uint32_t> zero_page(scratch::zero_page);
    const auto out = TensorAccessor(tensor::out);

    // ---- zero_source: once per kernel. No NoC write may be issued until the barrier returns (cmd buffer 0 is
    // borrowed by the iDMA zero mode until then).
    noc.async_write_zeros(zero_page, zero_src_bytes, {.offset_bytes = slice_offset});
    noc.write_zeros_l1_barrier();

    // ---- store_block over this core's range [start_page, start_page + num_pages_this_core).
    const uint32_t num_blocks = (num_pages_this_core + pages_per_barrier - 1) / pages_per_barrier;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t block_first = block * pages_per_barrier;
        const uint32_t remaining = num_pages_this_core - block_first;
        const uint32_t extent = remaining < pages_per_barrier ? remaining : pages_per_barrier;  // ragged last block

        for (uint32_t i = thread_id; i < extent; i += writer_threads) {
            const uint32_t page_id = start_page + block_first + i;
            for (uint32_t k = 0; k < chunks_per_page; ++k) {
                const uint32_t chunk_offset = k * zero_src_bytes;
                const uint32_t left = page_bytes - chunk_offset;
                const uint32_t chunk_bytes = left < zero_src_bytes ? left : zero_src_bytes;
                noc.async_write<NocOptions::TXN_ID>(
                    zero_page,
                    out,
                    chunk_bytes,
                    {.offset_bytes = slice_offset},
                    {.page_id = page_id, .offset_bytes = chunk_offset},
                    user_txn);
            }
        }
        noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);  // one barrier per block
    }

    // ---- release_tag: hand the hart back with the user id in its tag register (target page, ring protocols).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
