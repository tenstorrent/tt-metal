// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Column-parallel local compute: reduces this core's contiguous slice of each
// row's chunks to a single K-element unfused sequence (FP32 values + row-major
// UINT32 global indices, TopK XL engine layout), then packs both regions raw
// (as opaque 32-bit words) for the local writer to ship to the final core.
//
// The final core merges the gathered sequences pairwise. The bitonic halver in
// `topk_xl_merge` needs its two operands sorted in OPPOSITE directions
// (slot0 descending, slot1 ascending), so slice 0 emits its sequence sorted
// descending and every other slice emits ascending. Only the LAST rebuild's
// direction is flipped; intermediate rebuilds must stay descending because
// they feed this core's own next merge as the descending operand.
// (`test_topk_xl_rebuild_ascending` in tt-llk validates that an ascending
// rebuild is the exact lane-rank mirror of the descending one — which is the
// anti-alignment the merge halver requires.)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/dataflow/circular_buffer.h"

#include "topk_large_indices_compute_common.hpp"

void kernel_main() {
    using namespace topk_large_indices;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);
    // Index of this slice's first chunk within the row (slice base = start_chunk * K).
    const uint32_t start_chunk = get_arg_val<uint32_t>(3);
    const bool output_ascending = get_arg_val<uint32_t>(4) != 0;

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t values_cb = get_compile_time_arg_val(1);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    // An empty slice (valid_length cut this core's chunk range entirely) does
    // no compute; its writer sends a prefilled all--inf sequence instead.
    if (num_chunks == 0) {
        return;
    }

    compute_kernel_hw_startup(input_cb, values_cb);

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer values_cb_obj(values_cb);
    CircularBuffer indices_cb_obj(indices_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        // Per-core global index base: this slice starts at start_chunk * K
        // within the row. The chunk-base latch is loaded through
        // TTI_SFPLOADI-immediate config writes, so a runtime value can NOT be
        // passed to topk_xl_separate_indices_row_major_init (impossible asm
        // constraint at JIT time). Instead latch the compile-time base 0 and
        // step it forward with the all-constant advance primitive (+K per
        // call) under a runtime loop bound. Re-latched per row because
        // advance_chunk_base also mutates the latch across the chunk loop.
        topk_xl_separate_indices_row_major_init_static<0, 0>();
        for (uint32_t c = 0; c < start_chunk; ++c) {
            topk_xl_separate_indices_row_major_advance_chunk_base<K>();
        }

        const uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
        process_chunk<K>(input_cb_obj, slot0, first_chunk_elements, false);

        if (num_chunks == 1) {
            topk_xl_init<K, false>();
            topk_xl_rebuild<K, false>(slot0, output_ascending);
        }

        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
            process_chunk<K>(input_cb_obj, slot1, active_elements, true);

            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            const bool last_chunk = (chunk + 1 == num_chunks);
            topk_xl_rebuild<K, false>(slot0, last_chunk ? output_ascending : false);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Raw pack of the survivor: FP32 value tiles then UINT32 index tiles.
        // Both CBs are 32-bit formats, so pack is a bit-exact move; the final
        // core's unpack-to-dest restores the exact DST image for its merges.
        values_cb_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0, values_cb, tiles_per_sequence);
        values_cb_obj.push_back(tiles_per_sequence);

        indices_cb_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0 + tiles_per_sequence, indices_cb, tiles_per_sequence);
        indices_cb_obj.push_back(tiles_per_sequence);

        tile_regs_release();
    }
}
