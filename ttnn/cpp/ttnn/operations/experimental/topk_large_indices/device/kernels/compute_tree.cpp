// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Column-parallel TREE node compute (every rectangle core except the root).
//
// Phase 1 (leaf work): reduce this core's chunk slice to one K survivor in
// DST, exactly like the old local kernel.
// Phase 2 (tree work): for each of this core's num_merges winning levels,
// consume one partner sequence from the recv CB (delivered by this core's
// writer via the pairwise handshake), copy it to DST operand-1, and
// merge+rebuild IN PLACE — the survivor never leaves DST between levels.
// Phase 3 (ship): pack the survivor raw (FP32 values + UINT32 indices) for
// the writer to send to this core's winner.
//
// Directions: the bitonic halver needs operand0 descending and operand1
// ascending, so every DST-mutating action rebuilds DESCENDING except the
// LAST one before shipping, which rebuilds ASCENDING (validated upstream by
// test_topk_xl_rebuild_ascending: rebuild(asc) after a merge is the exact
// rank-mirror of rebuild(desc)).
//
// Empty slice (valid_length cut):
//   * num_merges == 0 -> nothing to compute; the writer ships the prefilled
//     all--inf scratch sequence instead.
//   * num_merges > 0 -> the FIRST partner is ADOPTED (copied to operand-0 and
//     rebuilt) instead of merged; an ascending monotonic run is bitonic, so
//     the rebuild is valid in either direction.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/dataflow/circular_buffer.h"

#include "topk_large_indices_compute_common.hpp"

void kernel_main() {
    using namespace topk_large_indices;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);
    const uint32_t start_chunk = get_arg_val<uint32_t>(3);
    const uint32_t num_merges = get_arg_val<uint32_t>(4);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t ship_values_cb = get_compile_time_arg_val(1);
    constexpr uint32_t ship_indices_cb = get_compile_time_arg_val(2);
    constexpr uint32_t recv_cb = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    const bool own_empty = (num_chunks == 0);
    // An empty slice with no partners contributes nothing; the writer ships
    // the -inf scratch sequence.
    if (own_empty && num_merges == 0) {
        return;
    }

    compute_kernel_hw_startup(input_cb, ship_values_cb);

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer ship_values_obj(ship_values_cb);
    CircularBuffer ship_indices_obj(ship_indices_cb);
    CircularBuffer recv_obj(recv_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        if (!own_empty) {
            // Leaf reduction of this core's chunk slice. The chunk base is
            // latched at 0 and stepped with the all-constant advance
            // primitive: a runtime chunk_base cannot reach the
            // TTI-immediate config write inside the init.
            topk_xl_separate_indices_row_major_init_static<0, 0>();
            for (uint32_t c = 0; c < start_chunk; ++c) {
                topk_xl_separate_indices_row_major_advance_chunk_base<K>();
            }

            const uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
            process_chunk<K>(input_cb_obj, slot0, first_chunk_elements, false);

            if (num_chunks == 1) {
                topk_xl_init<K, false>();
                topk_xl_rebuild<K, false>(slot0, num_merges == 0 /* ship next -> ascending */);
            }

            for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
                const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
                process_chunk<K>(input_cb_obj, slot1, active_elements, true);

                topk_xl_init<K, false>();
                topk_xl_merge<K, false>(slot0);
                const bool last_chunk = (chunk + 1 == num_chunks);
                topk_xl_rebuild<K, false>(slot0, last_chunk && num_merges == 0);
            }
        }

        // Tree merges: one partner sequence per winning level, in level order.
        for (uint32_t m = 0; m < num_merges; ++m) {
            const bool ship_next = (m + 1 == num_merges);
            const bool adopt = own_empty && (m == 0);
            const uint32_t dst_base = adopt ? slot0 : slot1;

            recv_obj.wait_front(sequence_tiles);
            // EXPLICIT format reconfig (bf16 -> UInt32): the preceding chunk phase
            // (topk_xl_copy_tile_init) hw-configures the unpacker for the BF16
            // input CB, and the plain _short init's state_configure is a no-op
            // in production JIT builds (the compute-kernel sentinel is compiled
            // out), so the _with_dt variant's llk reconfig is required here.
            // The reverse transition is explicit inside topk_xl_copy_tile_init.
            copy_tile_to_dst_init_short_with_dt(input_cb, recv_cb);
            for (uint32_t t = 0; t < sequence_tiles; ++t) {
                copy_tile(recv_cb, t, dst_base + t);
            }
            recv_obj.pop_front(sequence_tiles);

            topk_xl_init<K, false>();
            if (!adopt) {
                topk_xl_merge<K, false>(slot0);
            }
            topk_xl_rebuild<K, false>(slot0, ship_next /* ascending for the winner's halver */);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Ship the survivor raw: bit-exact FP32 value tiles then UINT32 index
        // tiles; the winner's copy_tile round-trips the DST image unchanged.
        ship_values_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0, ship_values_cb, tiles_per_sequence);
        ship_values_obj.push_back(tiles_per_sequence);

        ship_indices_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0 + tiles_per_sequence, ship_indices_cb, tiles_per_sequence);
        ship_indices_obj.push_back(tiles_per_sequence);

        tile_regs_release();
    }
}
