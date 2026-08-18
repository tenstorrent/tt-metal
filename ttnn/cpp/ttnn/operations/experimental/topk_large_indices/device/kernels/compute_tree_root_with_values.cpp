// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Column-parallel TREE root compute, return_values variant: identical to
// compute_tree_root.cpp plus the value materialization/pack (see
// compute_with_values.cpp for the pack-ordering and fp32->bf16 notes).
// Separate source so the indices-only tree root binary stays independent.

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
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);
    const uint32_t start_chunk = get_arg_val<uint32_t>(3);  // always 0 for slice 0; kept for arg-layout parity
    const uint32_t num_merges = get_arg_val<uint32_t>(4);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t recv_cb = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t values_out_cb = get_compile_time_arg_val(4);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    compute_kernel_hw_startup(input_cb, indices_out_cb);

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer indices_out_obj(indices_out_cb);
    CircularBuffer values_out_obj(values_out_cb);
    CircularBuffer recv_obj(recv_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        topk_xl_separate_indices_row_major_init_static<0, 0>();
        for (uint32_t c = 0; c < start_chunk; ++c) {
            topk_xl_separate_indices_row_major_advance_chunk_base<K>();
        }

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

        for (uint32_t m = 0; m < num_merges; ++m) {
            recv_obj.wait_front(sequence_tiles);
            // EXPLICIT format reconfig (bf16 -> UInt32): the preceding chunk phase
            // (topk_xl_copy_tile_init) hw-configures the unpacker for the BF16
            // input CB, and the plain _short init's state_configure is a no-op
            // in production JIT builds (the compute-kernel sentinel is compiled
            // out), so the _with_dt variant's llk reconfig is required here.
            // The reverse transition is explicit inside topk_xl_copy_tile_init.
            copy_tile_to_dst_init_short_with_dt(input_cb, recv_cb);
            for (uint32_t t = 0; t < sequence_tiles; ++t) {
                copy_tile(recv_cb, t, slot1 + t);
            }
            recv_obj.pop_front(sequence_tiles);

            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            topk_xl_rebuild<K, false>(slot0, false);
        }

#ifndef TOPK_SKIP_NEGINF_SENTINEL
        mark_neginf_indices<K>(slot0);
#endif
        materialize_index_rank_order<K>(slot0, indices_out_cb);
        materialize_values_rank_order<K>(slot0);

        tile_regs_commit();
        tile_regs_wait();

        values_out_obj.reserve_back(1);
        pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(values_out_cb);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(values_out_cb, 1, 0, slot0);
        values_out_obj.push_back(1);

        indices_out_obj.reserve_back(1);
        pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_out_cb);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_out_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_out_obj.push_back(1);

        tile_regs_release();
    }
}
