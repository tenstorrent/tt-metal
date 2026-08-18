// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Row-parallel compute, return_values variant: identical to compute.cpp up to
// the final materialization, then ALSO materializes and packs the FP32 value
// region as BFLOAT16 (values sit in DST beside the indices already — the
// unfused merge keeps [values, indices] regions; emitting values costs one
// extra in-DST transpose pass and one extra pack_untilize per row).
//
// Kept as a separate source (not an #ifdef in compute.cpp) so the default
// indices-only program's kernel binary stays byte-identical.
//
// Pack ordering per row: the values pack and the indices pack target CBs with
// different formats (Float16_b vs Float32) and face geometry, so
// pack_untilize_dest_init is re-run per pack (init derives format/geometry
// from the output CB; back-to-back inits are the supported reconfig path).
// The fp32->bf16 conversion in the packer is exact for the bf16-origin value
// words, and maps the 0xFF800000 (-inf) sentinel-lane values to bf16 -inf.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"

#include "topk_large_indices_compute_common.hpp"
#include "topk_large_indices_chunk_skip.hpp"

// Data-dependent chunk-skip early-out (row-parallel path only; see
// topk_large_indices_chunk_skip.hpp for design + soundness proof). One-line
// A/B toggle; keep in lockstep with compute.cpp's kChunkSkipEnable.
constexpr bool kChunkSkipEnable = true;

namespace {

// Copy-only half of topk_large_indices::process_chunk (kept local: the shared
// common header also feeds the column-parallel tree kernels, which must stay
// untouched by chunk-skip work).
template <uint32_t K>
FORCE_INLINE void copy_chunk_only(CircularBuffer& input_cb, uint32_t dst_base, uint32_t active_elements) {
    constexpr uint32_t tiles = (K + topk_large_indices::elements_per_tile - 1) / topk_large_indices::elements_per_tile;
    const uint32_t input_cb_id = input_cb.get_cb_id();
    input_cb.wait_front(tiles);
    topk_xl_copy_tile_init(input_cb_id);
    topk_xl_copy_tile<K>(input_cb_id, dst_base, 0, active_elements);
    input_cb.pop_front(tiles);
}

// Sort/split half of topk_large_indices::process_chunk.
template <uint32_t K>
FORCE_INLINE void finish_chunk_only(uint32_t dst_base, bool ascending) {
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, 0>(dst_base);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst_base, ascending);

    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(dst_base);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

}  // namespace

void kernel_main() {
    using namespace topk_large_indices;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t values_cb = get_compile_time_arg_val(3);
    constexpr uint32_t USER_K = get_compile_time_arg_val(4);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    namespace skip = topk_large_indices_chunk_skip;

    compute_kernel_hw_startup(input_cb, indices_cb);
    if constexpr (kChunkSkipEnable) {
        skip::chunk_skip_configure();
    }

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer indices_cb_obj(indices_cb);
    CircularBuffer values_cb_obj(values_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();
#ifdef CHUNK_SKIP_TELEMETRY
        skip::telemetry_row_begin(num_chunks);
#endif

        topk_xl_separate_indices_row_major_init_static<0, 0>();

        const uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
        process_chunk<K>(input_cb_obj, slot0, first_chunk_elements, false);

        if (num_chunks == 1) {
            topk_xl_init<K, false>();
            topk_xl_rebuild<K, false>(slot0, false);
        }

        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
            copy_chunk_only<K>(input_cb_obj, slot1, active_elements);

            if constexpr (kChunkSkipEnable) {
#ifdef CHUNK_SKIP_TELEMETRY
                // Telemetry variant: identical gate predicate and identical
                // per-chunk mailbox traffic on every TRISC (see compute.cpp).
                if (chunk >= skip::first_tested_chunk<USER_K>()) {
                    const bool skipped = skip::chunk_skip_decide<K, USER_K>(slot1);
                    skip::telemetry_record(chunk, skipped);
                    if (skipped) {
                        topk_xl_separate_indices_row_major_advance_chunk_base<K>();
                        continue;
                    }
                }
#else
                if (chunk >= skip::first_tested_chunk<USER_K>() && skip::chunk_skip_decide<K, USER_K>(slot1)) {
                    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
                    continue;
                }
#endif
            }

            finish_chunk_only<K>(slot1, true);

            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            topk_xl_rebuild<K, false>(slot0, false);
        }
#ifdef CHUNK_SKIP_TELEMETRY
        skip::telemetry_row_end<USER_K>(row, num_chunks);
#endif

        mark_neginf_indices<K>(slot0);
        materialize_index_rank_order<K>(slot0, indices_cb);
        materialize_values_rank_order<K>(slot0);

        tile_regs_commit();
        tile_regs_wait();

        values_cb_obj.reserve_back(1);
        pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(values_cb);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(values_cb, 1, 0, slot0);
        values_cb_obj.push_back(1);

        indices_cb_obj.reserve_back(1);
        pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_cb_obj.push_back(1);

        tile_regs_release();
    }
}
