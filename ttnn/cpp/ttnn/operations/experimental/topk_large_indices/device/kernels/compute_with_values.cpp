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

void kernel_main() {
    using namespace topk_large_indices;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t values_cb = get_compile_time_arg_val(3);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    compute_kernel_hw_startup(input_cb, indices_cb);

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer indices_cb_obj(indices_cb);
    CircularBuffer values_cb_obj(values_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        topk_xl_separate_indices_row_major_init_static<0, 0>();

        const uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
        process_chunk<K>(input_cb_obj, slot0, first_chunk_elements, false);

        if (num_chunks == 1) {
            topk_xl_init<K, false>();
            topk_xl_rebuild<K, false>(slot0, false);
        }

        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
            process_chunk<K>(input_cb_obj, slot1, active_elements, true);

            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            topk_xl_rebuild<K, false>(slot0, false);
        }

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
