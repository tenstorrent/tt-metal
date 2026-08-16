// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Column-parallel final compute: receives num_slices gathered K-element
// unfused sequences (FP32 value tiles + row-major UINT32 index tiles, packed
// raw by the local cores) and reduces them with the unfused TopK XL merge
// tree, then materializes the indices exactly like the single-core kernel.
//
// The gathered tiles are bit-exact copies of the local cores' DST regions:
// pack (32-bit raw) -> NoC -> unpack-to-dest (32-bit raw) round-trips the
// TopK XL engine layout unchanged, so the merge sees the same DST image the
// local core produced. Sequence 0 arrives sorted descending (merge operand 0),
// sequences 1..P-1 arrive sorted ascending (merge operand 1) — the direction
// anti-alignment the bitonic halver requires.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"

#include "topk_large_indices_compute_common.hpp"

void kernel_main() {
    using namespace topk_large_indices;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    // Runtime loop bound (not a compile-time arg) so the merge tree stays a
    // single code instance — the inlined merge/rebuild bodies are large and
    // full unrolling would overflow the TRISC code region.
    const uint32_t num_slices = get_arg_val<uint32_t>(1);

    constexpr uint32_t gathered_values_cb = get_compile_time_arg_val(0);
    constexpr uint32_t gathered_indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t indices_out_cb = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    compute_kernel_hw_startup(gathered_values_cb, indices_out_cb);
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_out_cb);

    CircularBuffer gathered_values_obj(gathered_values_cb);
    CircularBuffer gathered_indices_obj(gathered_indices_cb);
    CircularBuffer indices_out_obj(indices_out_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        const uint32_t gathered_tiles = num_slices * tiles_per_sequence;
        gathered_values_obj.wait_front(gathered_tiles);
        gathered_indices_obj.wait_front(gathered_tiles);

        tile_regs_acquire();

        // Sequence 0 (descending) -> slot0. Unpack-to-dest raw 32-bit copies.
        copy_tile_to_dst_init_short(gathered_values_cb);
        for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
            copy_tile(gathered_values_cb, t, slot0 + t);
        }
        copy_tile_to_dst_init_short(gathered_indices_cb);
        for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
            copy_tile(gathered_indices_cb, t, slot0 + tiles_per_sequence + t);
        }

        for (uint32_t p = 1; p < num_slices; ++p) {
            // Sequence p (ascending) -> slot1. Re-init the datacopy each
            // iteration: the preceding topk_xl_init/merge/rebuild reprogram
            // math-thread state (MOP expander, ADDR_MODs).
            copy_tile_to_dst_init_short(gathered_values_cb);
            for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
                copy_tile(gathered_values_cb, p * tiles_per_sequence + t, slot1 + t);
            }
            copy_tile_to_dst_init_short(gathered_indices_cb);
            for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
                copy_tile(gathered_indices_cb, p * tiles_per_sequence + t, slot1 + tiles_per_sequence + t);
            }

            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            // Keep the survivor descending: it is the next merge's operand 0,
            // and the final materialization emits rank order (largest first).
            topk_xl_rebuild<K, false>(slot0, false);
        }

        mark_neginf_indices<K>(slot0);
        materialize_index_rank_order<K>(slot0, indices_out_cb);

        tile_regs_commit();
        tile_regs_wait();

        indices_out_obj.reserve_back(1);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_out_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_out_obj.push_back(1);

        tile_regs_release();

        gathered_values_obj.pop_front(gathered_tiles);
        gathered_indices_obj.pop_front(gathered_tiles);
    }
}
