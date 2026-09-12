// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// N-way ROW_MAJOR width-concat reader.
//
// Each output stick is assembled directly from all original input sticks.  No
// pairwise intermediate is materialized, which avoids both extra dispatches and
// the shape-dependent corruption seen when an unaligned intermediate width is
// fed into another two-input concat on Blackhole.
//
// All inputs share one TensorAccessor ABI (same interleaved memory placement),
// but carry independent logical stick sizes and aligned page sizes.  Per input,
// a stick whose physical page pitch equals its logical size has no trailing pad
// bytes to avoid, and a destination offset that is a multiple of the shared
// transport alignment is a legal NOC endpoint -- the same in0_aligned /
// in1_direct predicate the two-input width reader uses, generalized to N
// inputs and evaluated at runtime (stick/page sizes are runtime, not
// compile-time, for this reader).  When every input is direct, every read is
// issued straight into the reserved output page and barriered once; otherwise
// each non-direct input falls back to a scratch-staged, per-byte copy.
//
// CT args:
//   cb_out, cb_scratch, N_INPUTS, OUT_PAGE_SIZE, BATCH, NOC_ALIGNMENT,
//   TensorAccessorArgs(shared input placement)
// RT args:
//   num_pages, start_page,
//   bases[N_INPUTS], stick_sizes[N_INPUTS], page_sizes[N_INPUTS]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#if !defined(ARCH_QUASAR)
// ckernel::load_blocking, for the staged copy's store-visibility drain. WH/BH only: on Quasar this
// header is unusable from a data-movement build and has no load_blocking, and the host gate keeps
// Quasar off the staged path entirely.
#include "ckernel.h"
#endif

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);
    uint32_t src_page = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t N_INPUTS = get_compile_time_arg_val(2);
    constexpr uint32_t OUT_PAGE_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t BATCH = get_compile_time_arg_val(4);
    constexpr uint32_t NOC_ALIGNMENT = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

    constexpr uint32_t base_rt = 2;
    constexpr uint32_t stick_rt = base_rt + N_INPUTS;
    constexpr uint32_t page_rt = stick_rt + N_INPUTS;

    uint32_t bases[N_INPUTS];
    uint32_t stick_sizes[N_INPUTS];
    uint32_t page_sizes[N_INPUTS];
    uint32_t dst_offsets[N_INPUTS];
    bool direct[N_INPUTS];
    bool all_direct = true;
    {
        uint32_t offset = 0;
        for (uint32_t input = 0; input < N_INPUTS; ++input) {
            bases[input] = get_arg_val<uint32_t>(base_rt + input);
            stick_sizes[input] = get_arg_val<uint32_t>(stick_rt + input);
            page_sizes[input] = get_arg_val<uint32_t>(page_rt + input);
            dst_offsets[input] = offset;
            direct[input] = (stick_sizes[input] == page_sizes[input]) && (offset % NOC_ALIGNMENT == 0);
            all_direct = all_direct && direct[input];
            offset += stick_sizes[input];
        }
    }

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer scratch_cb(cb_scratch);

    scratch_cb.reserve_back(1);
    const uint32_t scratch_addr = scratch_cb.get_write_ptr();

    while (num_pages > 0) {
        const uint32_t batch = num_pages < BATCH ? num_pages : BATCH;
        out_cb.reserve_back(batch);
        const uint32_t out_base = out_cb.get_write_ptr();

        if (all_direct) {
            // Every input of every page in this batch lands directly in a
            // disjoint range of the reserved output pages: issue every read
            // for the whole batch first, then barrier once -- matching the
            // two-input reader's fast path, generalized across BATCH pages
            // instead of re-barriering after each individual page.
            for (uint32_t page = 0; page < batch; ++page) {
                for (uint32_t input = 0; input < N_INPUTS; ++input) {
                    const auto accessor = TensorAccessor(src_args, bases[input], page_sizes[input]);
                    noc.async_read(
                        accessor,
                        out_cb,
                        stick_sizes[input],
                        {.page_id = src_page + page},
                        {.offset_bytes = page * OUT_PAGE_SIZE + dst_offsets[input]});
                }
            }
            noc.async_read_barrier();
        } else {
            uint32_t staged_end = 0;
            for (uint32_t page = 0; page < batch; ++page) {
                uint32_t dst = out_base + page * OUT_PAGE_SIZE;
                for (uint32_t input = 0; input < N_INPUTS; ++input) {
                    const auto accessor = TensorAccessor(src_args, bases[input], page_sizes[input]);
                    if (direct[input]) {
                        noc.async_read(
                            accessor,
                            out_cb,
                            stick_sizes[input],
                            {.page_id = src_page + page},
                            {.offset_bytes = page * OUT_PAGE_SIZE + dst_offsets[input]});
                        noc.async_read_barrier();
                        continue;
                    }

                    // Physical pages are always NOC-aligned.  Read one complete
                    // source page, then copy only its logical bytes into the packed
                    // output stick.  Halfwords: the host gate admits 2- and 4-byte
                    // dtypes only, so every stick size -- and so every cumulative
                    // segment offset -- is even, and the drain below needs the final
                    // store to begin at an even address.
                    noc.async_read(
                        accessor, scratch_cb, page_sizes[input], {.page_id = src_page + page}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    CoreLocalMem<volatile uint16_t> src(scratch_addr);
                    CoreLocalMem<volatile uint16_t> target(dst + dst_offsets[input]);
                    for (uint32_t half = 0; half < stick_sizes[input] / 2; ++half) {
                        target[half] = src[half];
                    }
                    staged_end = dst + dst_offsets[input] + stick_sizes[input];
                }
            }

            if (staged_end != 0) {
                // A RISC store can retire before its write-request reaches L1, and the NoC and the
                // unpacker are separate L1 clients with no program order against this core
                // (TensixTile/BabyRISCV/MemoryOrdering.md). push_back only bumps a stream register -- a
                // different memory region -- so neither it nor a fence orders the copy against it. Read
                // the last stored halfword back instead: obtaining the read-response proves L1 processed
                // that write, and the store queue is FIFO into a single region, so every earlier store
                // landed with it. The load has to start at exactly the last store's address -- Wormhole
                // detects store/load conflicts on the starting byte address alone, so a load of the
                // containing word is not ordered against a halfword stored partway into it.
                volatile tt_l1_ptr uint16_t* drain_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staged_end - 2);
#if defined(ARCH_QUASAR)
                (void)*drain_ptr;
#else
                (void)ckernel::load_blocking(drain_ptr);
#endif
            }
        }

        out_cb.push_back(batch);
        src_page += batch;
        num_pages -= batch;
    }
}
