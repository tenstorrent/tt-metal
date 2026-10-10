// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_tile_interleaved_unified.cpp (same directory). Identical transport loop and
// sequencer arithmetic; only the resource plumbing moves to the Metal 2.0 named bindings: the staging
// CB becomes dfb::in, the source tensor becomes tensor::src (so the src_addr runtime arg, the positional
// TensorAccessorArgs compile-time args and the accessor's explicit page-size argument all disappear),
// and the struct overlay on the runtime-arg block becomes per-field named runtime args. Forked rather
// than converted in place because the legacy file is still bound by repeat_interleave/codegen on the
// positional-arg API; delete this fork once every binder has adopted the named-binding form.
//
// Only the sequencers with a Metal 2.0 binder are carried: SEQ_REPEAT and SEQ_REPEAT_INTERLEAVE, which
// share one named runtime-arg set (num_pages, start_id, num_repeats, lower_pages, rep_dim_pages). The
// args:: / dfb:: / tensor:: names are generated from the binding factory's schema, and `if constexpr`
// does not suppress name lookup on a discarded branch, so a sequencer that reads other arguments (the
// SLICE / PERMUTE variable-length tails, PAD's cb_pad buffer index, CONCAT's second source address)
// cannot sit beside these unguarded. A future binder adds its branch behind a #ifdef selected through
// KernelSpec::compiler_options.defines, keeping the shared branches compiling for everyone else.
//
// Unified batched tile reader for interleaved tensors.
//
// One transport loop, pluggable address sequencer selected by "seq_id".
// The sequencer maps output_page -> source_page. All sequencer logic
// lives in sequencers.h as FORCE_INLINE functions.
//
// Named CT args: seq_id, batch, src_page_pitch (0 => absent, see below)
// Bindings:      tensor::src (source tensor), dfb::in (staging buffer this reader fills)
// Named RT args:
//   num_pages   — pages this core reads
//   start_id    — starting output page ID (meaning varies by sequencer)
//   num_repeats, lower_pages, rep_dim_pages — sequencer params (see the Seq*State
//                 structs in sequencers.h)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "sequencers.h"

// A caller may supply an authoritative source-pitch override via the named CT
// arg "src_page_pitch" (0 ⇒ absent). Otherwise the accessor itself remains the
// authority; deriving the pitch from a separate dtype table is wrong whenever
// that table and the allocated buffer ABI diverge.
//
// With the tensor binding the accessor's own page pitch is fixed by the binding
// (there is no explicit page-size constructor argument any more), so the override
// only bounds the per-page transfer size below. Every current binder passes 0.

// ── Transport loop (written ONCE) ───────────────────────────────────
// For simple sequencers (repeat, repeat_interleave):
//   each page = bounded noc.async_read(accessor, dfb, source_read_size,
//                                      {.page_id = next_page}, {.offset_bytes})

template <typename Accessor, typename State, typename NextFn>
FORCE_INLINE void read_pages(
    DataflowBuffer& dfb,
    uint32_t BATCH,
    uint32_t dfb_page_size,
    uint32_t source_read_size,
    const Accessor& accessor,
    uint32_t num_pages,
    State& state,
    NextFn next_fn) {
    Noc noc;

    uint32_t pages_left = num_pages;
    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        dfb.reserve_back(batch);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            const uint32_t source_page = next_fn(state);
            noc.async_read(
                accessor,
                dfb,
                source_read_size,
                {.page_id = source_page, .offset_bytes = 0},
                {.offset_bytes = l1_offset});
            l1_offset += dfb_page_size;
        }
        noc.async_read_barrier();
        dfb.push_back(batch);
        pages_left -= batch;
    }
}

// ── Main ────────────────────────────────────────────────────────────

void kernel_main() {
    // Named CT args
    constexpr uint32_t SEQ_ID = get_arg(args::seq_id);
    constexpr uint32_t BATCH = get_arg(args::batch);
    static_assert(
        SEQ_ID == SEQ_REPEAT || SEQ_ID == SEQ_REPEAT_INTERLEAVE,
        "reader_tile_interleaved_unified_metal2.cpp carries only the SEQ_REPEAT and SEQ_REPEAT_INTERLEAVE "
        "sequencers; see the header comment for how to add another");

    // Named RT args: the common header every sequencer shares, then the
    // repeat-family sequencer params.
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_repeats = get_arg(args::num_repeats);
    const uint32_t lower_pages = get_arg(args::lower_pages);
    const uint32_t rep_dim_pages = get_arg(args::rep_dim_pages);

    // dfb::in — the staging buffer this reader fills, one slot per source page,
    // for the writer downstream.
    DataflowBuffer dfb(dfb::in);
    const auto s = TensorAccessor(tensor::src);

    constexpr uint32_t source_page_size_override = get_arg(args::src_page_pitch);
    const uint32_t source_page_size =
        source_page_size_override != 0 ? source_page_size_override : s.get_aligned_page_size();
    const uint32_t dfb_page_size = dfb.get_entry_size();
    // The binding's pitch controls source page addressing. The transfer itself
    // must fit the destination DFB slot; keep the independent authorities even
    // when their current standard tile sizes happen to match.
    const uint32_t source_read_size = source_page_size < dfb_page_size ? source_page_size : dfb_page_size;

    // ── REPEAT ──────────────────────────────────────────────────────
    if constexpr (SEQ_ID == SEQ_REPEAT) {
        auto st = seq_repeat_init(start_id, num_repeats, lower_pages, rep_dim_pages);
        read_pages(dfb, BATCH, dfb_page_size, source_read_size, s, num_pages, st, seq_repeat_next);
    }

    // ── REPEAT_INTERLEAVE (per-element AABB replication) ─────────────
    // Reuses the repeat named args — only the addressing function differs.
    else if constexpr (SEQ_ID == SEQ_REPEAT_INTERLEAVE) {
        auto st = seq_repeat_interleave_init(start_id, num_repeats, lower_pages, rep_dim_pages);
        read_pages(dfb, BATCH, dfb_page_size, source_read_size, s, num_pages, st, seq_repeat_interleave_next);
    }
}
