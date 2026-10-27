// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Each program builds ONE split body -- long_rows picks reduce_split_segmented_row or reduce_split_row at
// build time from the physical width. Compiling both behind a runtime branch does not fit: under watcher and
// assert builds this kernel comes close to the kernel config buffer it shares with the reader and the writer.
//
// Outlining shared code enlarges this kernel rather than shrinking it: the topk_xl MOP and replay-buffer
// programming folds constants hard at each call site, so one generic copy is bigger than several specialised
// ones. Delete a path with a compile-time argument instead.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
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
// and the last segment needs when the row's survivor leaves this core.
template <uint32_t K>
FORCE_INLINE void sfpu_segment(uint32_t seg_chunks, bool later, uint32_t block, bool mirror) {
    constexpr uint32_t accumulator_slot = 0;
    auto epilogue = [block, later] {
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
    CircularBuffer& input, uint32_t first_chunk, uint32_t num_chunks, uint32_t tail_elements, bool final_ascending) {
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
    // known, and leave the standard [values, indices] epilogue layout.
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global<K>(survivor_slot);
}

template <uint32_t K>
FORCE_INLINE void reduce_segmented_row(
    CircularBuffer& input, uint32_t first_chunk, uint32_t num_chunks, uint32_t tail_elements, bool final_ascending) {
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

        const uint32_t first_elements = segment_first + 1 == end_chunk_total ? tail_elements : K;
        if (segment == 0) {
            sort_fused_chunk<K, true>(input, base, first_elements, false, segment_first & fused_chunk_id_mask);
        } else {
            sort_fused_chunk<K, false>(input, base, first_elements, false, segment_first & fused_chunk_id_mask);
        }

        if (segment_first == last_chunk) {
            topk_xl_rebuild<K, true>(base, mirror_final_survivor);
        } else {
            for (uint32_t chunk = segment_first + 1; chunk <= last_chunk; ++chunk) {
                const uint32_t active_elements = chunk + 1 == end_chunk_total ? tail_elements : K;
                sort_fused_chunk<K, false>(input, chunk_slot, active_elements, true, chunk & fused_chunk_id_mask);
                topk_xl_merge<K, true>(base);
                topk_xl_rebuild<K, true>(base, mirror_final_survivor && chunk == last_chunk);
            }
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
    bool emit_indices) {
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
            sfpu_segment<K>(seg_chunks, later, block, mirror);
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
    pack_untilize_dest_init<tiles_per_sequence<K>, tiles_per_sequence<K>>(indices_cb);

    uint32_t num_chunks;
    uint32_t tail_elements;
    uint32_t split_num_chunks;
    uint32_t split_tail_elements;
    // Clipped like the reader's: an empty half would otherwise stamp chunk ids past the end of the row.
    uint32_t split_first_chunk;
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

    const uint32_t num_whole_rows = has_split ? num_rows - 1 : num_rows;
    uint32_t first_row = 0;
    if constexpr (split_whole_rows) {
        if (num_whole_rows > 0 && num_chunks >= min_split_chunks &&
            (num_chunks <= max_fused_chunks || segmented_split)) {
            begin_thread_split();
            const bool prefetch =
                !segmented_split && num_chunks <= max_fused_chunks && topk_xl_split_can_prefetch<K>(num_chunks);
            for (uint32_t row = 0; row < num_whole_rows; ++row) {
                const bool prefetch_next = prefetch && row + 1 < num_whole_rows;
                tile_regs_acquire();
                if constexpr (segmented_split) {
                    reduce_split_segmented_row<K>(
                        input,
                        acc_cb,
                        indices_cb,
                        0 /*row_first_chunk*/,
                        num_chunks,
                        tail_elements,
                        false /*final_ascending*/,
                        true /*emit_indices*/);
                } else {
                    reduce_split_row<K>(
                        input, indices_cb, num_chunks, tail_elements, prefetch && row > 0, prefetch_next);
                }
                tile_regs_commit();
                tile_regs_wait();

                // The pack thread's SFPU work has reprogrammed its ADDR_MODs and MOP, so the untilize pack
                // below must re-init rather than inherit them.
                pack_untilize_dest_init<tiles_per_sequence<K>, tiles_per_sequence<K>>(indices_cb);
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
            topk_xl_split_end();
            if constexpr (segmented_split) {
                settle_next_program(input_cb);
            }
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

        tile_regs_acquire();

        // A half long enough, and starting on a 32-chunk block so the split body's segment-local stamps
        // decode, runs across MATH and PACK like the whole rows do. The output below is shared with every
        // other row.
        bool half_on_both_threads = false;
        if constexpr (segmented_split) {
            if (split_row && row_chunks >= min_split_chunks && (first_chunk & fused_chunk_id_mask) == 0) {
                begin_thread_split();
                reduce_split_segmented_row<K>(
                    input, acc_cb, indices_cb, first_chunk, row_chunks, row_tail, row_sends, false);
                topk_xl_split_end();
                // The split body leaves the packer in its last segment's mode; both output forms below start
                // from the untilize one.
                pack_untilize_dest_init<tiles_per_sequence<K>, tiles_per_sequence<K>>(indices_cb);
                half_on_both_threads = true;
            }
        }
        if (half_on_both_threads) {
            // The split body already left the survivor in tile 0; nothing more to reduce.
        } else if constexpr (body_mode == ComputeBodyMode::FusedSegmented) {
            reduce_segmented_row<K>(input, first_chunk, row_chunks, row_tail, row_sends);
        } else if constexpr (body_mode == ComputeBodyMode::FusedEndToEnd) {
            // The rebuild assembles its instructions from the direction, so each direction gets its own body.
            reduce_fused_row<K>(input, first_chunk, row_chunks, row_tail, row_sends);
        } else {
            // row_split_enabled() keeps has_split false for this body, so these are the whole-row bounds.
            reduce_classic_row<K>(input, row_chunks, row_tail);
        }

        if (row_receives) {
            merge_landed_survivor<K>(landing, input_cb);
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
            pack_untilize_dest_init<tiles_per_sequence<K>, tiles_per_sequence<K>>(indices_cb);
        }
    }
    if (receives_survivor || sends_survivor) {
        settle_next_program(input_cb);
    }
}
