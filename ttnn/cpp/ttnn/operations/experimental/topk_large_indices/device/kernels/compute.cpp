// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Each program builds ONE split body -- long_rows picks reduce_split_segmented_row or reduce_split_row at
// build time from the physical width. Compiling both behind a runtime branch does not fit: under watcher and
// assert builds this kernel comes close to the kernel config buffer it shares with the reader and the writer.
//
// Outlining shared code enlarges this kernel rather than shrinking it: the topk_xl MOP and replay-buffer
// programming folds constants hard at each call site, so one generic copy is bigger than several specialised
// ones. Delete a path with a compile-time argument instead. The exception is arm_untilize_pack, out of line in
// two-level programs: every call site passes the same constants, so one copy folds them as well.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#ifdef TOPK_TWO_LEVEL
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/compute/reduce.h"
#include "topk_large_indices_two_level.hpp"
#endif
#include "api/dataflow/circular_buffer.h"
#include "topk_large_indices_compute_body_mode.hpp"
#include "topk_large_indices_metadata.hpp"
#include "topk_large_indices_runtime_args.hpp"

namespace {

// The input's rows span two or more chunks; a program of one-chunk rows compiles neither the split nor the short init.
constexpr bool multi_chunk_rows = get_compile_time_arg_val(8) != 0;

using ComputeBodyMode = ttnn::operations::experimental::topk_large_indices::program::ComputeBodyMode;
// Shared host/device constant (topk_large_indices_compute_body_mode.hpp), re-declared locally so the kernel
// reads as kernel code rather than reaching into a ttnn:: namespace.
constexpr uint32_t max_fused_chunks = ttnn::operations::experimental::topk_large_indices::program::max_fused_chunks;

constexpr uint32_t fused_chunk_id_mask = max_fused_chunks - 1;

template <uint32_t K>
constexpr uint32_t tiles_per_sequence = (K + TILE_R_DIM * TILE_C_DIM - 1) / (TILE_R_DIM * TILE_C_DIM);

// UNPACK's share of a row: feed every chunk once, plus the next row's first chunk when MATH will prefetch it.
template <uint32_t K>
FORCE_INLINE void feed_row(
    CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements, bool first_prefetched, bool prefetch_next) {
    const uint32_t input_cb = input.get_cb_id();
    for (uint32_t chunk = first_prefetched ? 1 : 0; chunk < num_chunks; ++chunk) {
        input.wait_front(tiles_per_sequence<K>);
        if (chunk == (first_prefetched ? 1 : 0)) {
            topk_xl_copy_tile_init_unpack(input_cb);
        }
        topk_xl_copy_tile_unpack<K>(input_cb, 0, chunk + 1 == num_chunks ? tail_elements : K);
        input.pop_front(tiles_per_sequence<K>);
        topk_xl_unpack_set_srcb_dummy_valid();
        if (chunk > 0) {
            topk_xl_unpack_set_srcb_dummy_valid();
        }
    }
    if (prefetch_next) {
        // MATH copies the next row's first chunk before this row's index transposes, so UNPACK feeds it first.
        // K, not tail_elements: every row in a program has the same width, and can_prefetch() demands an even
        // chunk count, so the next row's FIRST chunk is always full -- only its last can be short.
        ASSERT(num_chunks >= 2);
        input.wait_front(tiles_per_sequence<K>);
        topk_xl_copy_tile_unpack<K>(input_cb, 0, K);
        input.pop_front(tiles_per_sequence<K>);
        topk_xl_unpack_set_srcb_dummy_valid();
    }
}

template <uint32_t K>
FORCE_INLINE void math_row(
    CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements, bool first_prefetched, bool prefetch_next) {
    const uint32_t input_cb = input.get_cb_id();
    topk_xl_split_sfpu_init_once();
    if (first_prefetched) {
        // PACK may take the SFPU over only after this init.
        topk_xl_split_post_token();
    }
    // The previous row's index transposes reprogram MATH after a prefetched first chunk, so the first copy made here
    // inits it.
    bool init_copy = true;
    topk_xl_split_math_row<K>(
        num_chunks,
        [&](uint32_t chunk, uint32_t tile) {
            if (init_copy) {
                topk_xl_copy_tile_init_math(input_cb);
                init_copy = false;
            }
            topk_xl_copy_tile_math<K>(input_cb, tile, chunk + 1 == num_chunks ? tail_elements : K);
        },
        first_prefetched,
        prefetch_next);
}

// PACK's share of a whole row: every SFPU stage, then the index split and the -inf marking.
template <uint32_t K>
FORCE_INLINE void sfpu_row(uint32_t num_chunks, bool first_prefetched) {
    if (first_prefetched) {
        topk_xl_split_take_math_token();
    }
    topk_xl_split_pack_row<K>(num_chunks, false, [] {
        topk_xl_split_indices<K>(0);
        topk_large_indices_mark_neginf_indices_init_at_split();
        topk_large_indices_mark_neginf_indices_at_split<K>();
    });
}

// UNPACK's share of one 32-chunk segment of a long K' 2048 row.
template <uint32_t K>
FORCE_INLINE void feed_segment(CircularBuffer& input, uint32_t seg_chunks, uint32_t seg_tail) {
    // A segment is fed exactly like a row that neither inherits a prefetched first chunk nor leaves one.
    feed_row<K>(input, seg_chunks, seg_tail, false, false);
}

template <uint32_t K>
FORCE_INLINE void math_segment(CircularBuffer& input, uint32_t seg_chunks, uint32_t seg_tail) {
    const uint32_t input_cb = input.get_cb_id();
    topk_xl_split_sfpu_init_once();
    topk_xl_split_math_row<K>(seg_chunks, [&](uint32_t chunk, uint32_t tile) {
        if (chunk == 0) {
            topk_xl_copy_tile_init_math(input_cb);
        }
        topk_xl_copy_tile_math<K>(input_cb, tile, chunk + 1 == seg_chunks ? seg_tail : K);
    });
}

// PACK's share of one segment. `block` is the segment's 32-chunk block of the row, which decodes the
// segment-local stamps; `mirror` ends the segment ascending, which a later segment needs for its unfused merge
// and the last segment needs when the row's survivor leaves this core. `keep_fused` leaves the survivor fused for a
// fused cross-core join; only a single-segment row may ask for it.
template <uint32_t K>
FORCE_INLINE void sfpu_segment(uint32_t seg_chunks, bool later, uint32_t block, bool mirror, bool keep_fused) {
    constexpr uint32_t accumulator_slot = 0;
    auto epilogue = [block, later, keep_fused] {
        if (keep_fused) {
            return;
        }
        topk_xl_split_indices<K>(block * (max_fused_chunks * K));
        if (later) {
            topk_large_indices_move_segment_result_at_split<K>();
        }
    };
    topk_xl_split_pack_row<K>(seg_chunks, mirror, epilogue);
}

// MATH hands Dst over and PACK spills the unfused accumulator to L1, so the next segment can use all eight tiles.
template <uint32_t K>
FORCE_INLINE void spill_accumulator(CircularBuffer& acc, uint32_t segment) {
    topk_xl_split_release_accumulator();
    constexpr uint32_t accumulator_tiles = 2 * tiles_per_sequence<K>;
    constexpr uint32_t accumulator_slot = 0;
    const uint32_t acc_cb = acc.get_cb_id();
    topk_xl_split_hold_accumulator([&] {
        // Segment 1 finds the packer in its untilize mode, a later segment in pack_tile mode with the MOP the pack
        // thread's SFPU work reprogrammed.
        if (segment == 1) {
            pack_untilize_uninit(acc_cb);
        } else {
            pack_init(acc_cb);
        }
        acc.reserve_back(accumulator_tiles);
        for (uint32_t tile = 0; tile < accumulator_tiles; ++tile) {
            pack_tile<true>(accumulator_slot + tile, acc_cb, tile);
        }
        acc.push_back(accumulator_tiles);
    });
}

// Compositions that pair two compute-API calls, and the mailbox accessor.

template <uint32_t K>
FORCE_INLINE void mark_neginf_indices(uint32_t dst) {
    topk_large_indices_mark_neginf_indices_init();
    topk_large_indices_mark_neginf_indices<K>(dst);
}

// One word of the reader-to-compute mailbox.
FORCE_INLINE uint32_t mailbox_word(uint32_t meta_cb, uint32_t word) { return read_tile_value(meta_cb, 0, word); }

// MATH raises the two Tensix semaphores it shares with PACK to the token depth the schedule needs (their
// default max of 1 would saturate), then posts MATH_PACK to hand Dst over. The separate "MATH's SFPU init is
// done" token is the first post inside the row schedule, not this one.
FORCE_INLINE void begin_thread_split() {
    topk_xl_split_begin();
    topk_xl_split_wait_start();
}

// FullCopyInit: the first chunk after a configuration of the unpacker and math unit runs the full copy init; later
// chunks find that intact and set up only the copy.
template <uint32_t K, bool FullCopyInit = true>
FORCE_INLINE void copy_chunk(CircularBuffer& input, uint32_t dst, uint32_t active_elements) {
    const uint32_t input_cb = input.get_cb_id();

    input.wait_front(tiles_per_sequence<K>);
    if constexpr (FullCopyInit || !multi_chunk_rows) {
        topk_xl_copy_tile_init(input_cb);
    } else {
        topk_xl_copy_tile_init_short(input_cb);
    }
    topk_xl_copy_tile<K>(input_cb, dst, 0, active_elements);
    input.pop_front(tiles_per_sequence<K>);
}

template <uint32_t K>
FORCE_INLINE void sort_classic_chunk(CircularBuffer& input, uint32_t dst, uint32_t active_elements, bool ascending) {
    copy_chunk<K>(input, dst, active_elements);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, 0>(dst);
    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst, ascending);

    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(dst);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

// FullInit is compile-time and this helper is force-inlined so the hot loop has
// neither a mode branch nor a repeated full TopK configuration sequence. It also selects the full copy init
// (copy_chunk) over the short one.
template <uint32_t K, bool FullInit>
FORCE_INLINE void sort_fused_chunk(
    CircularBuffer& input, uint32_t dst, uint32_t active_elements, bool ascending, uint32_t local_chunk_id) {
    copy_chunk<K, FullInit>(input, dst, active_elements);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices_rt<K>(dst, local_chunk_id);
    if constexpr (FullInit) {
        topk_xl_init<K, true>();
    } else {
        topk_large_indices_reinit_fused_after_stamp();
    }
    topk_xl_local_sort<K>(dst, ascending);
}

template <uint32_t K>
FORCE_INLINE void reduce_classic_row(CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements) {
    constexpr uint32_t survivor_slot = 0;
    constexpr uint32_t incoming_slot = 2 * tiles_per_sequence<K>;

    topk_xl_separate_indices_row_major_init_static<0, 0>();
    sort_classic_chunk<K>(input, survivor_slot, num_chunks == 1 ? tail_elements : K, false);

    if (num_chunks == 1) {
        topk_xl_init<K, false>();
        topk_xl_rebuild<K, false>(survivor_slot, false);
        return;
    }

    for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
        const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
        sort_classic_chunk<K>(input, incoming_slot, active_elements, true);

        topk_xl_init<K, false>();
        topk_xl_merge<K, false>(survivor_slot);
        topk_xl_rebuild<K, false>(survivor_slot, false);
    }
}

// Each body reduces its chunks into an unfused survivor at DST slot 0 with row-global indices. first_chunk is the
// row-global id of the first chunk handed over, nonzero only for the second half of a split row. final_ascending
// is a parameter, not a template argument: topk_xl_rebuild takes it as a value, and making it a template
// argument would build the whole body twice for nothing.
template <uint32_t K>
FORCE_INLINE void reduce_fused_row(
    CircularBuffer& input,
    uint32_t first_chunk,
    uint32_t num_chunks,
    uint32_t tail_elements,
    bool final_ascending,
    bool keep_fused) {
    constexpr uint32_t survivor_slot = 0;
    // A fused survivor has no separate index tiles. Its merge operand is
    // therefore adjacent at one sequence, rather than two sequences, away.
    constexpr uint32_t incoming_slot = tiles_per_sequence<K>;

    // The row has at most 32 chunks, so the row-global chunk id fits the stamp directly.
    sort_fused_chunk<K, true>(input, survivor_slot, num_chunks == 1 ? tail_elements : K, false, first_chunk);

    if (num_chunks == 1) {
        topk_xl_rebuild<K, true>(survivor_slot, final_ascending);
    } else {
        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
            sort_fused_chunk<K, false>(input, incoming_slot, active_elements, true, first_chunk + chunk);
            topk_xl_merge<K, true>(survivor_slot);
            topk_xl_rebuild<K, true>(survivor_slot, final_ascending && chunk + 1 == num_chunks);
        }
    }

    // Decode the five-bit chunk stamp once, after the final fused survivor is
    // known, and leave the standard [values, indices] epilogue layout. A half
    // joined fused across cores stays fused; the receiver decodes after the join.
    if (keep_fused) {
        return;
    }
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global<K>(survivor_slot);
}

template <uint32_t K>
FORCE_INLINE void reduce_segmented_row(
    CircularBuffer& input,
    uint32_t first_chunk,
    uint32_t num_chunks,
    uint32_t tail_elements,
    bool final_ascending,
    bool keep_fused) {
    constexpr uint32_t segment_capacity = max_fused_chunks;
    constexpr uint32_t accumulator_slot = 0;
    constexpr uint32_t segment_slot = 2 * tiles_per_sequence<K>;

    const uint32_t end_chunk_total = first_chunk + num_chunks;
    // Segments follow the row's 32-chunk blocks, so the stamp is the chunk id within its block. A split row's
    // second half can start mid-block, which is why the first segment here may be shorter than the capacity.
    uint32_t segment_first = first_chunk;
    for (uint32_t segment = 0; segment_first < end_chunk_total; ++segment) {
        const uint32_t block_end = (segment_first / segment_capacity + 1) * segment_capacity;
        const uint32_t end_chunk = block_end < end_chunk_total ? block_end : end_chunk_total;
        const uint32_t last_chunk = end_chunk - 1;
        const bool last_segment = end_chunk == end_chunk_total;

        // Segment zero is already in the final accumulator slot. Later
        // segments use the second unfused sequence. Their fused input chunk
        // is adjacent to the fused survivor at base + tiles_per_sequence<K>.
        const uint32_t base = segment == 0 ? accumulator_slot : segment_slot;
        const uint32_t chunk_slot = base + tiles_per_sequence<K>;
        const bool mirror_final_survivor = segment != 0 || (last_segment && final_ascending);

        // The segment's first chunk sorts descending into the base, every later one ascending into the chunk slot and
        // merges in. One loop, so each sort_fused_chunk form has a single call site.
        for (uint32_t chunk = segment_first; chunk <= last_chunk; ++chunk) {
            const bool head = chunk == segment_first;
            const uint32_t active_elements = chunk + 1 == end_chunk_total ? tail_elements : K;
            if (head && segment == 0) {
                sort_fused_chunk<K, true>(input, base, active_elements, false, chunk & fused_chunk_id_mask);
            } else {
                sort_fused_chunk<K, false>(
                    input, head ? base : chunk_slot, active_elements, !head, chunk & fused_chunk_id_mask);
            }
            if (!head) {
                topk_xl_merge<K, true>(base);
            }
            if (!head || segment_first == last_chunk) {
                topk_xl_rebuild<K, true>(base, mirror_final_survivor && chunk == last_chunk);
            }
        }

        // A fused cross-core join takes this single segment's survivor fused.
        if (keep_fused) {
            ASSERT(last_segment);
            break;
        }
        // Split once per segment. The base is a multiple of 32*K, so it does
        // not overlap the decoded segment-local index bits.
        topk_xl_separate_indices_row_major_global_init();
        topk_xl_separate_indices_row_major_global_base<K>(
            base, (segment_first / segment_capacity) * (segment_capacity * K));

        if (segment != 0) {
            // Operand zero is descending and each later segment was rebuilt
            // ascending, satisfying the unfused bitonic merge precondition.
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(accumulator_slot);
            topk_xl_rebuild<K, false>(accumulator_slot, last_segment && final_ascending);
        }
        segment_first = end_chunk;
    }
}

// The partner's unfused survivor lands as raw 32-bit words (values and indices) and merges into ours.
template <uint32_t K>
FORCE_INLINE void merge_landed_survivor(CircularBuffer& landing, uint32_t input_cb) {
    constexpr uint32_t survivor_tiles = 2 * tiles_per_sequence<K>;
    // The receiver always merges into slot zero and always ends descending: it writes the row itself.
    constexpr uint32_t survivor_slot = 0;
    const uint32_t landing_cb = landing.get_cb_id();
    const uint32_t incoming_slot = survivor_slot + survivor_tiles;

    landing.wait_front(survivor_tiles);
    reconfig_data_format_srca(landing_cb);
    copy_init(landing_cb);
    for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
        copy_tile(landing_cb, tile, incoming_slot + tile);
    }
    landing.pop_front(survivor_tiles);
    // The integer landing format leaves the FPU in int math; the rebuild transposes need the float path.
    reconfig_data_format_srca(input_cb);

    topk_xl_init<K, false>();
    topk_xl_merge<K, false>(survivor_slot);
    topk_xl_rebuild<K, false>(survivor_slot, false);
}

// The fused join: both halves arrive fused ([bf16 value | u16 stamp], row-global stamps), so the fused compare
// orders equal values by stamp exactly as one core reducing the whole row does. The unfused merge above compares
// values only, which returned tied keys of a split row in a different order (#63408). Only a row of at most 32
// chunks has unique stamps; the receiver then decodes the joined survivor once.
template <uint32_t K>
FORCE_INLINE void merge_landed_fused_survivor(CircularBuffer& landing, uint32_t input_cb) {
    constexpr uint32_t survivor_tiles = 2 * tiles_per_sequence<K>;
    constexpr uint32_t survivor_slot = 0;
    // A fused survivor is one sequence, so its merge operand sits one sequence away.
    constexpr uint32_t incoming_slot = survivor_slot + tiles_per_sequence<K>;
    const uint32_t landing_cb = landing.get_cb_id();

    // The hand-off is sized for an unfused survivor; a fused one fills only its first sequence.
    landing.wait_front(survivor_tiles);
    reconfig_data_format_srca(landing_cb);
    copy_init(landing_cb);
    for (uint32_t tile = 0; tile < tiles_per_sequence<K>; ++tile) {
        copy_tile(landing_cb, tile, incoming_slot + tile);
    }
    landing.pop_front(survivor_tiles);
    reconfig_data_format_srca(input_cb);

    topk_xl_init<K, true>();
    topk_xl_merge<K, true>(survivor_slot);
    topk_xl_rebuild<K, true>(survivor_slot, false);
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global<K>(survivor_slot);
}

template <uint32_t K>
FORCE_INLINE void materialize_index_rank_order(uint32_t dst, uint32_t indices_cb) {
    transpose_dest_init<true, false>(indices_cb);
    for (uint32_t tile = 0; tile < tiles_per_sequence<K>; ++tile) {
        transpose_dest<true, false>(dst + tiles_per_sequence<K> + tile);
    }
}

// A row split across MATH and PACK (the tt-llk split scheduler; kernel_main says from how many chunks); it ends with
// the -inf indices marked and the index tiles transposed, as mark_neginf_indices and materialize_index_rank_order
// leave a row.
// The fused stamp carries the chunk id in five bits, so this body is only valid within one segment.
template <uint32_t K>
FORCE_INLINE void reduce_split_row(
    CircularBuffer& input,
    uint32_t indices_cb,
    uint32_t num_chunks,
    uint32_t tail_elements,
    bool first_prefetched,
    bool prefetch_next) {
    // Each of these is one RISC's whole share and compiles to nothing on the other two.
    feed_row<K>(input, num_chunks, tail_elements, first_prefetched, prefetch_next);
    math_row<K>(input, num_chunks, tail_elements, first_prefetched, prefetch_next);
    sfpu_row<K>(num_chunks, first_prefetched);
    transpose_dest_init<true, false>(indices_cb);
    for (uint32_t tile = 0; tile < tiles_per_sequence<K>; ++tile) {
        transpose_dest<true, false>(tiles_per_sequence<K> + tile);
    }
}

// A K = 2048 row of more than 32 chunks, its segments split as above. A later segment's chunks cycle through all eight
// Dst tiles and its result moves to tiles 4 to 7, so the unfused accumulator (tiles 0 to 3) waits in acc_cb's L1 until
// its merge with the segment.
template <uint32_t K>
FORCE_INLINE void reduce_split_segmented_row(
    CircularBuffer& input,
    uint32_t acc_cb,
    uint32_t indices_cb,
    uint32_t row_first_chunk,
    uint32_t num_chunks,
    uint32_t tail_elements,
    bool final_ascending,
    bool emit_indices,
    bool keep_fused) {
    static_assert(K == 2048, "the segmented body is K = 2048's");
    constexpr uint32_t accumulator_tiles = 2 * tiles_per_sequence<K>;
    constexpr uint32_t accumulator_slot = 0;
    constexpr uint32_t segment_slot = accumulator_tiles;
    // A one-chunk later segment is the only one that cannot run across the threads -- 33 and 65 chunk rows
    // return wrong results if it does.
    constexpr uint32_t min_split_later_chunks = 2;
    const uint32_t input_cb = input.get_cb_id();
    const uint32_t end_chunk_total = row_first_chunk + num_chunks;
    CircularBuffer acc(acc_cb);
    // Segments follow the row's 32-chunk blocks. The split body stamps chunks segment-locally and decodes with
    // the block base, so a segment must start on a block boundary -- kernel_main only sends a half here when its
    // first chunk is one.
    uint32_t segment_first = row_first_chunk;
    for (uint32_t segment = 0; segment_first < end_chunk_total; ++segment) {
        const uint32_t block_end = (segment_first / max_fused_chunks + 1) * max_fused_chunks;
        const uint32_t end_chunk = block_end < end_chunk_total ? block_end : end_chunk_total;
        const uint32_t seg_chunks = end_chunk - segment_first;
        const bool last_segment = end_chunk == end_chunk_total;
        const uint32_t seg_tail = last_segment ? tail_elements : K;
        const uint32_t block = segment_first / max_fused_chunks;
        const bool later = segment != 0;
        const bool split = !later || seg_chunks >= min_split_later_chunks;
        // A later segment merges into the descending accumulator, so it ends ascending; the last one ends
        // ascending when this core's survivor leaves for its partner.
        const bool mirror = later || (last_segment && final_ascending);

        if (!split) {
            // Reachable only at one chunk, so there is nothing to merge into it.
            sort_fused_chunk<K, true>(input, segment_slot, seg_tail, false, 0);
            topk_xl_rebuild<K, true>(segment_slot, true);
        } else {
            if (later) {
                spill_accumulator<K>(acc, segment);
            }
            feed_segment<K>(input, seg_chunks, seg_tail);
            math_segment<K>(input, seg_chunks, seg_tail);
            sfpu_segment<K>(seg_chunks, later, block, mirror, keep_fused);
        }

        if (later) {
            if (split) {
                // copy_init sets no formats: the unpack to Dst needs acc_cb's, the merge and transposes the input's.
                reconfig_data_format_srca(input_cb, acc_cb);
                copy_init(acc_cb);
                acc.wait_front(accumulator_tiles);
                for (uint32_t tile = 0; tile < accumulator_tiles; ++tile) {
                    copy_tile(acc_cb, tile, accumulator_slot + tile);
                }
                acc.pop_front(accumulator_tiles);
                reconfig_data_format_srca(acc_cb, input_cb);
                // No explicit sync between these copies and the merge below: acc_cb is UInt32, so copy_tile
                // takes the LLK's 32-bit unpack-to-dest path, whose math_unpack_to_dest_math_ready() holds MATH
                // on the MATH_DONE semaphore until the unpack has landed. A tensix_sync here would add nothing
                // -- it blocks only the issuing RISC, never MATH -- so it cannot order unpack against the merge.
            }
            if (!split) {
                topk_xl_separate_indices_row_major_global_init();
                topk_xl_separate_indices_row_major_global_base<K>(segment_slot, block * (max_fused_chunks * K));
            }
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(accumulator_slot);
            topk_xl_rebuild<K, false>(accumulator_slot, last_segment && final_ascending);
        }
        segment_first = end_chunk;
    }
    // A survivor that leaves this core is packed raw by the caller, so it skips the index epilogue.
    if (emit_indices) {
        mark_neginf_indices<K>(accumulator_slot);
        materialize_index_rank_order<K>(accumulator_slot, indices_cb);
    }
}

// Arms the untilize pack of a row's indices. Out of line in two-level programs: every call site passes the same
// constant CB, so the one copy loses no folding. Each passes its own line on to the compute-kernel sentinel's
// diagnostics.
#ifdef TOPK_TWO_LEVEL
template <uint32_t K>
__attribute__((noinline)) void arm_untilize_pack(uint32_t indices_cb, uint32_t call_line = __builtin_LINE()) {
    pack_untilize_dest_init<tiles_per_sequence<K>, tiles_per_sequence<K>>(indices_cb, call_line);
}
#else
template <uint32_t K>
FORCE_INLINE void arm_untilize_pack(uint32_t indices_cb, uint32_t call_line = __builtin_LINE()) {
    pack_untilize_dest_init<tiles_per_sequence<K>, tiles_per_sequence<K>>(indices_cb, call_line);
}
#endif

// A raw pack and unpack round trip can leave unpacker state a later program's init does not reset; one discarded
// copy resets it. The copy deliberately reads whatever the input CB's read pointer happens to address -- the value
// is thrown away, and the address is always inside the CB's own L1 allocation.
FORCE_INLINE void settle_next_program(uint32_t input_cb) {
    reconfig_data_format_srca(input_cb);
    copy_init(input_cb);
    tile_regs_acquire();
    copy_tile(input_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
}

#ifdef TOPK_TWO_LEVEL
#ifdef TRISC_MATH
// The block maxima of one tile unpacked for a max row reduce (data transposed within each face, one face per
// handoff), pooled into DEST rows `row` (rows 0-15 of the tile, faces 0 and 1) and `row + 4` (rows 16-31, faces 2 and
// 3) of the current tile. GMPOOL max-accumulates into the top row of an aligned 4-row block and zeroes the other three;
// DEST rows start undefined, which it reads as -inf. Unlike reduce_tile, the result row stays a row: no transpose
// back into a column.
inline void two_level_pool_tile(uint32_t row) {
    TT_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, row);
    TT_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, row);
    TT_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, row + 4);
    TT_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, row + 4);
}
#endif

// Phase 1 of a two-level program (see topk_large_indices_two_level.hpp): one unit's 32 tiles, tilized one tile wide so
// tile t holds the unit's keys from 1024 t and its row r is block 32 t + r. Every row is reduced to its maximum, eight
// tiles per output tile: tile i of a group leaves its 32 maxima in DEST rows 8 i and 8 i + 4, 16 each, so the output
// tile's rows 4 j hold blocks 16 j .. 16 j + 15 of the group, in block order. The writer sends them to the maxima
// tensor.
FORCE_INLINE void two_level_unit() {
    constexpr uint32_t cb_stage1 = get_named_compile_time_arg_val("tl_cb_stage1");
    constexpr uint32_t cb_tiled = get_named_compile_time_arg_val("tl_cb_tiled");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("tl_cb_scaler");
    constexpr uint32_t cb_red = get_named_compile_time_arg_val("tl_cb_red");
    constexpr uint32_t stage1_batch = get_named_compile_time_arg_val("tl_stage1_batch");
    constexpr uint32_t group_tiles = 8;
    static_assert(stage1_batch % group_tiles == 0);
    CircularBuffer tiled(cb_tiled);
    CircularBuffer reduced(cb_red);
    CircularBuffer(cb_scaler).wait_front(1);
    compute_kernel_lib::tilize<1, cb_stage1, cb_tiled>(stage1_batch);
    reconfig_data_format(cb_tiled, cb_scaler);
    pack_reconfig_data_format(cb_red);
    // reduce_init without its pack edge mask, which would keep only column 0.
    state_configure(cb_tiled, cb_scaler, cb_red, __builtin_LINE());
    UNPACK((llk_unpack_AB_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tiled, cb_scaler)));
    MATH((llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE, MATH_FIDELITY>(
        cb_tiled, cb_scaler)));
    tiled.wait_front(stage1_batch);
    for (uint32_t group = 0; group < stage1_batch; group += group_tiles) {
        reduced.reserve_back(1);
        tile_regs_acquire();
        MATH((math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0)));
        for (uint32_t i = 0; i < group_tiles; ++i) {
            UNPACK((llk_unpack_AB_reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tiled, cb_scaler, group + i, 0)));
            MATH((two_level_pool_tile(8 * i)));
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_red);
        tile_regs_release();
        reduced.push_back(1);
    }
    tiled.pop_front(stage1_batch);
    reduce_uninit();
}

// Runs the phase-1 units the reader sends until it announces anything else, which it returns: one of this core's rows
// (header_row, or header_split for a split row's stage 2), its half of a split row (header_half), or the end
// (header_end). Phase 1 runs on the ordinary schedule, so a split in progress is ended first.
FORCE_INLINE uint32_t two_level_next_row(uint32_t total_units, bool& split_on) {
    constexpr uint32_t cb_stage1 = get_named_compile_time_arg_val("tl_cb_stage1");
    constexpr uint32_t stage1_batch = get_named_compile_time_arg_val("tl_stage1_batch");
    CircularBuffer stage1(cb_stage1);
    while (true) {
        stage1.wait_front(1);
        // The two header slots are reused: drop any cached copy of this one before reading it.
        UNPACK(invalidate_l1_cache());
        const uint32_t code = read_tile_value(cb_stage1, 0, 0);
        stage1.pop_front(1);
        if (code >= total_units) {
            stage1.wait_front(stage1_batch);
            stage1.pop_front(stage1_batch);
            return code;
        }
        // No settle after the split: two-level items never round-trip an accumulator through L1 (stage 2 and stage 3
        // are at most 32 chunks, one segment), and the unit's tilize and reduce inits configure what they use.
        if (split_on) {
            topk_xl_split_end();
            split_on = false;
        }
        two_level_unit();
    }
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(topk_core_args::compute_num_rows);
    constexpr bool row_split = get_named_compile_time_arg_val("row_split") != 0;
    constexpr bool split_whole_rows = get_named_compile_time_arg_val("split_whole_rows") != 0;
    constexpr bool long_rows = get_named_compile_time_arg_val("long_rows") != 0;
    const bool has_split = row_split && get_arg_val<uint32_t>(topk_core_args::compute_has_split) != 0;
    const bool second_half = row_split && get_arg_val<uint32_t>(topk_core_args::compute_second_half) != 0;
    const bool sends_survivor = has_split && second_half;
    const bool receives_survivor = has_split && !second_half;
    const uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::compute_search_length);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr auto body_mode = static_cast<ComputeBodyMode>(get_compile_time_arg_val(3));
    constexpr bool valid_length_from_metadata = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(5);
    constexpr uint32_t landing_cb = get_compile_time_arg_val(6);
    constexpr uint32_t send_cb = get_compile_time_arg_val(7);
    constexpr uint32_t acc_cb = get_compile_time_arg_val(9);
    constexpr bool has_acc_cb = get_compile_time_arg_val(10) != 0;

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    static_assert(
        body_mode == ComputeBodyMode::Classic || body_mode == ComputeBodyMode::FusedEndToEnd ||
            body_mode == ComputeBodyMode::FusedSegmented,
        "invalid TopK compute body mode");

    constexpr uint32_t survivor_tiles = 2 * tiles_per_sequence<K>;
    constexpr uint32_t final_survivor = 0;

    compute_kernel_hw_startup(input_cb, indices_cb);
    arm_untilize_pack<K>(indices_cb);

    uint32_t num_chunks;
    uint32_t tail_elements;
    uint32_t split_num_chunks;
    uint32_t split_tail_elements;
    // Clipped like the reader's: an empty half would otherwise stamp chunk ids past the end of the row.
    uint32_t split_first_chunk;
#ifdef TOPK_TWO_LEVEL
    // A two-level program (see topk_large_indices_two_level.hpp) learns its mode and search length from the mailbox.
    bool two_level_mode = false;
    uint32_t two_level_search_length = 0;
    static_assert(valid_length_from_metadata, "a two-level program reads its bounds from the mailbox");
    static_assert(split_whole_rows, "a two-level program runs its stages on the whole-row split body");
#endif
    if constexpr (valid_length_from_metadata) {
        // The reader already clipped the rows and the split half against the metadata length.
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.wait_front(1);
        num_chunks = mailbox_word(meta_cb, topk_metadata_num_chunks_word);
        tail_elements = mailbox_word(meta_cb, topk_metadata_tail_elements_word);
        if constexpr (row_split) {
            split_num_chunks = mailbox_word(meta_cb, topk_metadata_split_word_offset + topk_metadata_num_chunks_word);
            split_tail_elements =
                mailbox_word(meta_cb, topk_metadata_split_word_offset + topk_metadata_tail_elements_word);
        } else {
            split_num_chunks = num_chunks;
            split_tail_elements = tail_elements;
        }
        // A split core's mailbox whole-row count is the row's own, so half_bounds places the cut exactly as
        // the reader did. Deriving it with the same helper is what keeps the two from drifting apart.
        split_first_chunk = half_bounds({num_chunks, tail_elements}, K, second_half).first_chunk;
#ifdef TOPK_TWO_LEVEL
        two_level_mode = mailbox_word(meta_cb, topk_two_level::mailbox_mode_word) != 0;
        two_level_search_length = mailbox_word(meta_cb, topk_two_level::mailbox_search_length_word);
#endif
        meta_cb_obj.pop_front(1);
    } else if constexpr (!row_split) {
        const auto bounds = calculate_topk_bounds(search_len, K);
        num_chunks = bounds.num_chunks;
        tail_elements = bounds.tail_elements;
        split_num_chunks = num_chunks;
        split_tail_elements = tail_elements;
        split_first_chunk = 0;
    } else {
        // A core's whole rows are always the whole row: only its one half is ever a chunk range.
        const auto row = calculate_topk_bounds(search_len, K);
        const auto split = half_bounds(row, K, second_half);
        num_chunks = row.num_chunks;
        tail_elements = row.tail_elements;
        split_num_chunks = split.num_chunks;
        split_tail_elements = split.tail_elements;
        split_first_chunk = split.first_chunk;
    }

    CircularBuffer input(input_cb);
    CircularBuffer indices(indices_cb);
    CircularBuffer landing(landing_cb);
    CircularBuffer send(send_cb);

    // The physical width decides which split body a program needs, so only that one is built. A shorter
    // valid_length can still drop the row below a segment at runtime; the segmented body handles that as a
    // single segment, it just cannot prefetch, which is why prefetch is off for these programs.
    constexpr bool split_segments = K == 2048 && body_mode == ComputeBodyMode::FusedSegmented && has_acc_cb;
    constexpr bool segmented_split = split_segments && long_rows;
    // Shorter rows run on one thread: the set-up costs more than the overlap saves.
    constexpr uint32_t min_split_chunks = K == 2048 ? 2 : 3;
#ifdef TOPK_TWO_LEVEL
    // A split row's halves of stage 3 run as soon as they can, not after the end (see the reader).
    constexpr bool early_halves = get_named_compile_time_arg_val("tl_early_halves") != 0;
    static_assert(!early_halves || segmented_split, "early halves run on the segmented split body");
#endif

    const uint32_t num_whole_rows = has_split ? num_rows - 1 : num_rows;
    uint32_t first_row = 0;
    // A split-row half runs on the split body like a whole row, when it starts on a 32-chunk block or (two-level) its
    // stamps are offset to the row's; the body keeps one call site. As the loop's last item its survivor stays in DEST
    // for the split-row path; a two-level early half (tl_early_halves) runs mid-loop and packs its survivor out.
    bool half_done = false;
    // A two-level split row's halves run on the split body as 16-chunk rows from chunk 0, stamped with the row's chunk
    // ids (the split body decodes stamps per 32-chunk block and needs a row to start on one).
    bool shifted_half = false;
    if constexpr (split_whole_rows) {
        const bool half_in_split = segmented_split && has_split && split_num_chunks >= min_split_chunks &&
                                   (split_first_chunk & fused_chunk_id_mask) == 0;
#ifdef TOPK_TWO_LEVEL
        // A two-level row is two items on this loop's split body, both read from input_cb: its block maxima (stage
        // 2), then its gathered candidates (stage 3). One call site keeps a single copy of the body in the binary.
        const uint32_t whole_row_chunks = num_chunks;
        const uint32_t whole_row_tail = tail_elements;
        // A two-level core runs its items until the reader's end header: two per row, between phase-1 units.
        const uint32_t items = two_level_mode ? 0xFFFFFFFFu : num_whole_rows + (half_in_split ? 1 : 0);
        const uint32_t maxima_chunks = topk_two_level::stage2_chunks(two_level_search_length);
        const uint32_t total_units =
            topk_two_level::Units::of(
                two_level_search_length,
                get_named_compile_time_arg_val("tl_stage1_batch"),
                get_common_arg_val<uint32_t>(topk_common_args::compute_two_level_num_input_rows))
                .total;
        bool split_on = false;
        if (two_level_mode || half_in_split ||
            (num_whole_rows > 0 && num_chunks >= min_split_chunks &&
             (num_chunks <= max_fused_chunks || segmented_split))) {
            if (!two_level_mode) {
                begin_thread_split();
                split_on = true;
            }
            const bool prefetch = !two_level_mode && !segmented_split && num_chunks <= max_fused_chunks &&
                                  topk_xl_split_can_prefetch<K>(num_chunks);
            // A split row's stage 2 runs here. Its stage 3 runs in halves here too with the segmented split body;
            // only non-segmented programs take the split-row path below for it.
            bool stage2_only = false;
            for (uint32_t row = 0; row < items; ++row) {
                const bool maxima_item = two_level_mode && (row & 1) == 0;
                bool half_item = !two_level_mode && row == num_whole_rows;
                if (maxima_item) {
                    const uint32_t next = two_level_next_row(total_units, split_on);
                    if (early_halves && next == topk_two_level::header_end) {
                        break;
                    }
                    if (next == (early_halves ? topk_two_level::header_half : topk_two_level::header_end)) {
                        if constexpr (!segmented_split) {
                            break;
                        }
                        // Each core of the pair sorts its half of the split row's 32 candidate chunks here, as a
                        // 16-chunk row whose chunk stamps start at the half's first chunk: the stamps are the
                        // row's, so the halves join fused (ties ordered as on one core) and decode to the row's
                        // positions.
                        static_assert(topk_two_level::candidate_keys / K / 2 >= min_split_chunks);
                        static_assert(topk_two_level::candidate_keys / K <= max_fused_chunks);
                        if (!has_split) {
                            break;
                        }
                        shifted_half = true;
                        num_chunks = topk_two_level::candidate_keys / K;
                        tail_elements = K;
                        const auto half = half_bounds({num_chunks, tail_elements}, K, second_half);
                        split_num_chunks = half.num_chunks;
                        split_tail_elements = half.tail_elements;
                        split_first_chunk = half.first_chunk;
                        half_item = true;
                    }
                    stage2_only = next == topk_two_level::header_split;
                    if (!split_on) {
                        begin_thread_split();
                        split_on = true;
                    }
                } else if (two_level_mode && stage2_only) {
                    continue;
                }
                // Shadow the row bounds with this item's: the call below is the single-level loop's.
                const bool fused_join = num_chunks <= max_fused_chunks;
                const uint32_t num_chunks =
                    half_item ? split_num_chunks
                              : (!two_level_mode ? whole_row_chunks
                                                 : (maxima_item ? maxima_chunks : topk_two_level::candidate_keys / K));
                const uint32_t tail_elements = half_item ? split_tail_elements : (two_level_mode ? K : whole_row_tail);
                const bool prefetch_next = prefetch && row + 1 < num_whole_rows;
#else
        if (half_in_split || (num_whole_rows > 0 && num_chunks >= min_split_chunks &&
                              (num_chunks <= max_fused_chunks || segmented_split))) {
            begin_thread_split();
            const bool prefetch =
                !segmented_split && num_chunks <= max_fused_chunks && topk_xl_split_can_prefetch<K>(num_chunks);
            const bool fused_join = num_chunks <= max_fused_chunks;
            const uint32_t whole_row_chunks = num_chunks;
            const uint32_t whole_row_tail = tail_elements;
            for (uint32_t row = 0; row < num_whole_rows + (half_in_split ? 1 : 0); ++row) {
                const bool half_item = row == num_whole_rows;
                const uint32_t num_chunks = half_item ? split_num_chunks : whole_row_chunks;
                const uint32_t tail_elements = half_item ? split_tail_elements : whole_row_tail;
                const bool prefetch_next = prefetch && row + 1 < num_whole_rows;
#endif
                tile_regs_acquire();
                if constexpr (segmented_split) {
#ifdef TOPK_TWO_LEVEL
                    if (shifted_half) {
                        topk_xl_split_set_stamp_offset(split_first_chunk);
                    }
#endif
                    // A half ends ascending when its survivor leaves this core, and stays fused for a fused join.
                    reduce_split_segmented_row<K>(
                        input,
                        acc_cb,
                        indices_cb,
                        half_item && !shifted_half ? split_first_chunk : 0,
                        num_chunks,
                        tail_elements,
                        half_item && sends_survivor,
                        !half_item,
                        half_item && fused_join);
#ifdef TOPK_TWO_LEVEL
                    if (shifted_half) {
                        topk_xl_split_set_stamp_offset(0);
                    }
                    if (half_item && early_halves && two_level_mode) {
                        // The half's survivor leaves DEST raw: to the partner from the second core, into this core's
                        // own send CB until the end's join on the first.
                        topk_xl_split_end();
                        split_on = false;
                        arm_untilize_pack<K>(indices_cb);
                        tile_regs_commit();
                        tile_regs_wait();
                        send.reserve_back(survivor_tiles);
                        pack_untilize_uninit(send_cb);
                        pack_reconfig_data_format(send_cb);
                        for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
                            pack_tile(final_survivor + tile, send_cb);
                        }
                        send.push_back(survivor_tiles);
                        tile_regs_release();
                        arm_untilize_pack<K>(indices_cb);
                        shifted_half = false;
                        // The half took a maxima item's slot: the next item is a header again.
                        ++row;
                        continue;
                    }
#endif
                    if (half_item) {
                        half_done = true;
                        break;
                    }
                } else {
                    reduce_split_row<K>(
                        input, indices_cb, num_chunks, tail_elements, prefetch && row > 0, prefetch_next);
                }
                tile_regs_commit();
                tile_regs_wait();

                // The pack thread's SFPU work has reprogrammed its ADDR_MODs and MOP, so the untilize pack
                // below must re-init rather than inherit them.
                arm_untilize_pack<K>(indices_cb);
                indices.reserve_back(1);
                pack_untilize_dest<tiles_per_sequence<K>, tiles_per_sequence<K>>(
                    indices_cb, 1, 0, final_survivor + tiles_per_sequence<K>);
                indices.push_back(1);

                if (prefetch_next) {
                    topk_xl_split_release_dest_bank0<DST_ACCUM_MODE>();
                } else {
                    tile_regs_release();
                }
            }
            // Pairs with the begin_thread_split() above, on both ways out of this block. Leaving the two
            // semaphores at max 15 is invisible to this op and breaks the next one.
#ifdef TOPK_TWO_LEVEL
            // A two-level core may end on a phase-1 unit, with no split open.
            if (split_on) {
                topk_xl_split_end();
            }
#else
            topk_xl_split_end();
#endif
            if (half_done) {
                // The split body leaves the packer in its last segment's mode; both output forms of the split-row
                // path start from the untilize one. No settle: DEST holds the half's survivor.
                arm_untilize_pack<K>(indices_cb);
#ifdef TOPK_TWO_LEVEL
            } else if (segmented_split || two_level_mode) {
#else
            } else if constexpr (segmented_split) {
#endif
                settle_next_program(input_cb);
            }
#ifdef TOPK_TWO_LEVEL
            if (two_level_mode && early_halves) {
                if (has_split && !second_half) {
                    // The split row's join: this core's parked half, then the partner's landed one merged into it.
                    arm_untilize_pack<K>(indices_cb);
                    tile_regs_acquire();
                    send.wait_front(survivor_tiles);
                    reconfig_data_format_srca(input_cb, send_cb);
                    copy_init(send_cb);
                    for (uint32_t tile = 0; tile < tiles_per_sequence<K>; ++tile) {
                        copy_tile(send_cb, tile, final_survivor + tile);
                    }
                    send.pop_front(survivor_tiles);
                    merge_landed_fused_survivor<K>(landing, input_cb);
                    mark_neginf_indices<K>(final_survivor);
                    materialize_index_rank_order<K>(final_survivor, indices_cb);
                    tile_regs_commit();
                    tile_regs_wait();
                    indices.reserve_back(1);
                    pack_untilize_dest<tiles_per_sequence<K>, tiles_per_sequence<K>>(
                        indices_cb, 1, 0, final_survivor + tiles_per_sequence<K>);
                    indices.push_back(1);
                    tile_regs_release();
                    settle_next_program(input_cb);
                }
                return;
            }
            if (two_level_mode) {
                if (!has_split) {
                    return;
                }
                if (!half_done) {
                    // This core's half of the split row's 32 candidate chunks, on the split-row path: only programs
                    // without the segmented split body come here (with it, both cores sort their half above).
                    num_chunks = topk_two_level::candidate_keys / K;
                    tail_elements = K;
                    const auto half = half_bounds({num_chunks, tail_elements}, K, second_half);
                    split_num_chunks = half.num_chunks;
                    split_tail_elements = half.tail_elements;
                    split_first_chunk = half.first_chunk;
                }
            }
#endif
            if (!has_split) {
                return;
            }
            first_row = num_whole_rows;
        }
    }

    for (uint32_t row = first_row; row < num_rows; ++row) {
        const bool split_row = has_split && row == num_whole_rows;
        const uint32_t first_chunk = split_row ? split_first_chunk : 0;
        const uint32_t row_chunks = split_row ? split_num_chunks : num_chunks;
        const uint32_t row_tail = split_row ? split_tail_elements : tail_elements;
        const bool row_receives = split_row && receives_survivor;
        // A survivor that leaves this core is merge operand one of its partner and must end ascending.
        const bool row_sends = split_row && sends_survivor;
        // A split row's halves are joined fused whenever its stamps are unique, i.e. the whole row is at most 32
        // chunks; both partners derive this from the same whole-row chunk count, so they agree on the format.
        const bool fused_join = split_row && num_chunks <= max_fused_chunks;

        // A half that ran as the split-body loop's last item (see half_done) left its survivor in tile 0 with DEST
        // held, across MATH and PACK.
        const bool half_on_both_threads = split_row && half_done;
        if (!half_on_both_threads) {
            tile_regs_acquire();
        }
        if (half_on_both_threads) {
            // Nothing more to reduce.
        } else if constexpr (body_mode == ComputeBodyMode::FusedSegmented) {
            reduce_segmented_row<K>(input, first_chunk, row_chunks, row_tail, row_sends, fused_join);
        } else if constexpr (body_mode == ComputeBodyMode::FusedEndToEnd) {
            // The rebuild assembles its instructions from the direction, so each direction gets its own body.
            reduce_fused_row<K>(input, first_chunk, row_chunks, row_tail, row_sends, fused_join);
        } else {
            // row_split_enabled() keeps has_split false for this body, so these are the whole-row bounds.
            reduce_classic_row<K>(input, row_chunks, row_tail);
        }

        if (row_receives) {
            if (fused_join) {
                merge_landed_fused_survivor<K>(landing, input_cb);
            } else {
                merge_landed_survivor<K>(landing, input_cb);
            }
        }

        if (row_sends) {
            tile_regs_commit();
            tile_regs_wait();

            // Raw tile pack so the partner can unpack the survivor straight back into DST.
            send.reserve_back(survivor_tiles);
            pack_untilize_uninit(send_cb);
            pack_reconfig_data_format(send_cb);
            for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
                pack_tile(final_survivor + tile, send_cb);
            }
            send.push_back(survivor_tiles);
        } else {
            mark_neginf_indices<K>(final_survivor);
            materialize_index_rank_order<K>(final_survivor, indices_cb);

            tile_regs_commit();
            tile_regs_wait();

            indices.reserve_back(1);
            pack_untilize_dest<tiles_per_sequence<K>, tiles_per_sequence<K>>(
                indices_cb, 1, 0, final_survivor + tiles_per_sequence<K>);
            indices.push_back(1);
        }

        tile_regs_release();

        if (row_sends) {
            // Re-arm the untilize pack the row output path expects.
            arm_untilize_pack<K>(indices_cb);
        }
    }
    if (receives_survivor || sends_survivor) {
        settle_next_program(input_cb);
    }
}
