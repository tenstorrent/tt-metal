// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/topk_xl.h"
#include "api/compute/transpose_wh_dest.h"
#include "api/dataflow/circular_buffer.h"

namespace {

template <uint32_t K>
FORCE_INLINE void materialize_index_rank_order(uint32_t idst) {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + 1023u) / 1024u;

    transpose_wh_dest_init_short<true, false>();
    for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
        transpose_wh_dest<true, false>(idst + tiles_per_sequence + t);
    }
}

template <uint32_t K, uint32_t upper16, uint32_t max_lower_mod, uint32_t lower_mod_case = 0>
FORCE_INLINE void init_chunk_base_lower(uint32_t lower_mod) {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    static_assert(max_lower_mod == 32 || max_lower_mod == 64 || max_lower_mod == 128);
    static_assert(lower_mod_case < max_lower_mod);

    if constexpr (lower_mod_case + 1 == max_lower_mod) {
        topk_xl_separate_indices_row_major_init_static<upper16, (lower_mod_case * K) & 0xFFFFu>();
    } else {
        if (lower_mod == lower_mod_case) {
            topk_xl_separate_indices_row_major_init_static<upper16, (lower_mod_case * K) & 0xFFFFu>();
        } else {
            init_chunk_base_lower<K, upper16, max_lower_mod, lower_mod_case + 1>(lower_mod);
        }
    }
}

template <uint32_t PublicK>
FORCE_INLINE void init_chunk_base(uint32_t chunk) {
    if constexpr (PublicK == 512) {
        const uint32_t lower_mod = chunk & 0x7Fu;
        if (chunk < 128) {
            init_chunk_base_lower<PublicK, 0, 128>(lower_mod);
        } else {
            init_chunk_base_lower<PublicK, 1, 128>(lower_mod);
        }
    } else if constexpr (PublicK == 1024) {
        const uint32_t lower_mod = chunk & 0x3Fu;
        if (chunk < 64) {
            init_chunk_base_lower<PublicK, 0, 64>(lower_mod);
        } else {
            init_chunk_base_lower<PublicK, 1, 64>(lower_mod);
        }
    } else {
        const uint32_t lower_mod = chunk & 0x1Fu;
        if (chunk < 32) {
            init_chunk_base_lower<PublicK, 0, 32>(lower_mod);
        } else if (chunk < 64) {
            init_chunk_base_lower<PublicK, 1, 32>(lower_mod);
        } else if (chunk < 96) {
            init_chunk_base_lower<PublicK, 2, 32>(lower_mod);
        } else {
            init_chunk_base_lower<PublicK, 3, 32>(lower_mod);
        }
    }
}

template <uint32_t K>
FORCE_INLINE void process_chunk(
    CircularBuffer& input_cb, uint32_t dst_base, uint32_t chunk, uint32_t active_elements, bool ascending) {
    constexpr uint32_t tiles_per_sequence = (K + 1023u) / 1024u;
    const uint32_t input_cb_id = input_cb.get_cb_id();
    input_cb.wait_front(tiles_per_sequence);
    topk_xl_copy_tile_init(input_cb_id);
    topk_xl_copy_tile<K>(input_cb_id, dst_base, 0, active_elements);
    input_cb.pop_front(tiles_per_sequence);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, 0>(dst_base);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst_base, ascending);

    init_chunk_base<K>(chunk);
    topk_xl_separate_indices_row_major<K>(dst_base);
}

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t tail_elements = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    static_assert(tail_elements >= 1 && tail_elements <= K, "tail_elements must be in [1, K]");
    constexpr uint32_t tiles_per_sequence = (K + 1023u) / 1024u;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    compute_kernel_hw_startup(input_cb, indices_cb);
    MATH((llk_math_reconfig_remap(true)));
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence, false, TILE_C_DIM, false, false>(indices_cb);

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer indices_cb_obj(indices_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        constexpr uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
        process_chunk<K>(input_cb_obj, slot0, 0, first_chunk_elements, false);

        if (num_chunks == 1) {
            topk_xl_init<K, false>();
            topk_xl_rebuild<K, false>(slot0, false);
        }

        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
            process_chunk<K>(input_cb_obj, slot1, chunk, active_elements, true);

            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            topk_xl_rebuild<K, false>(slot0, false);
        }

        materialize_index_rank_order<K>(slot0);
        topk_xl_copy_tile_reinit_mop(input_cb);

        tile_regs_commit();
        tile_regs_wait();

        indices_cb_obj.reserve_back(1);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_cb_obj.push_back(1);

        tile_regs_release();
    }
}
