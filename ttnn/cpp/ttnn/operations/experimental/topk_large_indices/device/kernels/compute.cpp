// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"

#include "topk_large_indices_chunk_skip.hpp"

// Data-dependent chunk-skip early-out (row-parallel path only; see
// topk_large_indices_chunk_skip.hpp for design + soundness proof).
// One-line A/B toggle: flip to false to reproduce the pre-skip kernel
// (kernels JIT-compile, no host rebuild needed for the toggle).
constexpr bool kChunkSkipEnable = true;

// Bring-up diagnostic: dump the post-rebuild DST values region over DPRINT to
// calibrate rank_to_values_word against torch rank order. Never enable
// together with performance runs.
// #define CHUNK_SKIP_DIAG 1
#ifdef CHUNK_SKIP_DIAG
#include "api/debug/dprint.h"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// topk_large_indices keeps final values and indices in the TopK XL LLK DST
// layout until the last rank-order materialization step. The generic compute
// APIs (`isneginf_tile` + `where_tile`) operate on normal tile layouts and do
// not line up with this intermediate value/index pairing, so they only replaced
// a subset of the final -inf lanes during validation.
//
// Keep this as op-local SFPU functionality for now instead of exporting a
// public LLK API: it is tied to the final TopK XL LLK DST contract below,
// where the value words start at the normal `idst` base and the UINT32 index
// words start at `indices_offset`. The helper walks that layout directly,
// compares final values against exact BF16 -inf stored in the FP32 DST
// container (`0xFF800000`), and conditionally writes the sentinel index
// `0xFFFFFFFF`.
inline void _topk_large_indices_mark_neginf_indices_init_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

template <uint32_t K>
inline void _topk_large_indices_mark_neginf_indices_() {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = K == 2048 ? 2 : 1;
    constexpr uint32_t indices_offset = tiles_per_sequence * 64;
    constexpr uint32_t iterations = (K == 512 ? 1 : K == 1024 ? 2 : 4) * 16;

    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFFFF);

    for (uint32_t i = 0; i < iterations; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace {

constexpr uint32_t elements_per_tile = TILE_R_DIM * TILE_C_DIM;

template <uint32_t K>
FORCE_INLINE void materialize_index_rank_order(uint32_t idst, uint32_t indices_cb) {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;

    transpose_dest_init<true, false>(indices_cb);
    for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
        transpose_dest<true, false>(idst + tiles_per_sequence + t);
    }
}

template <uint32_t K>
FORCE_INLINE void mark_neginf_indices(uint32_t idst) {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    MATH((ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>, idst, VectorMode::RC_custom)));
}

// First half of chunk processing: pull the chunk from the input CB into DST.
// Always runs -- the chunk must be resident in DST to be inspected at all.
template <uint32_t K>
FORCE_INLINE void copy_chunk(CircularBuffer& input_cb, uint32_t dst_base, uint32_t active_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    const uint32_t input_cb_id = input_cb.get_cb_id();
    input_cb.wait_front(tiles_per_sequence);
    topk_xl_copy_tile_init(input_cb_id);
    topk_xl_copy_tile<K>(input_cb_id, dst_base, 0, active_elements);
    input_cb.pop_front(tiles_per_sequence);
}

// Second half: stamp fused LSB indices, locally sort, split into unfused
// values + row-major indices, and advance the chunk base. Skipped (except for
// the chunk-base advance) when the chunk-skip test proves the chunk cannot
// contribute to the final top-k.
template <uint32_t K>
FORCE_INLINE void finish_chunk(uint32_t dst_base, bool ascending) {
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, 0>(dst_base);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst_base, ascending);

    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(dst_base);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

template <uint32_t K>
FORCE_INLINE void process_chunk(CircularBuffer& input_cb, uint32_t dst_base, uint32_t active_elements, bool ascending) {
    copy_chunk<K>(input_cb, dst_base, active_elements);
    finish_chunk<K>(dst_base, ascending);
}

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t USER_K = get_compile_time_arg_val(3);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;
    constexpr uint32_t slot1 = sequence_tiles;

    namespace skip = topk_large_indices_chunk_skip;

    compute_kernel_hw_startup(input_cb, indices_cb);
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);
    if constexpr (kChunkSkipEnable) {
        skip::chunk_skip_configure();
    }

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer indices_cb_obj(indices_cb);

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
            copy_chunk<K>(input_cb_obj, slot1, active_elements);

            if constexpr (kChunkSkipEnable) {
#ifdef CHUNK_SKIP_TELEMETRY
                // Telemetry variant: identical gate predicate and identical
                // per-chunk mailbox traffic on every TRISC; the decision is
                // recorded (MATH only) before the skip acts.
                if (chunk >= skip::first_tested_chunk<USER_K>()) {
                    const bool skipped = skip::chunk_skip_decide<K, USER_K>(slot1);
                    skip::telemetry_record(chunk, skipped);
                    if (skipped) {
                        topk_xl_separate_indices_row_major_advance_chunk_base<K>();
                        continue;
                    }
                }
#else
                // Gated start: chunk >= max(2, USER_K/4). The floor of 2 is a
                // layout requirement (threshold address valid only after the
                // first merge+rebuild); USER_K/4 amortizes the test to where
                // skips are actually probable (see the header).
                if (chunk >= skip::first_tested_chunk<USER_K>() && skip::chunk_skip_decide<K, USER_K>(slot1)) {
                    // Chunk proven irrelevant (all elements strictly below
                    // the running USER_K-th survivor). The chunk was popped;
                    // only the MATH-side chunk-base bookkeeping must advance.
                    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
                    continue;
                }
#endif
            }

            finish_chunk<K>(slot1, true);

            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            topk_xl_rebuild<K, false>(slot0, false);
        }
#ifdef CHUNK_SKIP_TELEMETRY
        skip::telemetry_row_end<USER_K>(row, num_chunks);
#endif

#if defined(CHUNK_SKIP_DIAG) && defined(TRISC_MATH)
        // Calibration dump: raw post-rebuild values-region words of slot0.
        if (row == 0) {
            skip::chunk_skip_configure();
            ckernel::tensix_sync();
            volatile uint32_t* mm = reinterpret_cast<volatile uint32_t*>(RISCV_DEST_START_ADDR);
            for (uint32_t w = 0; w < tiles_per_sequence * 1024; ++w) {
                DPRINT("DW {} {}\n", w, mm[w]);
            }
        }
#endif

        mark_neginf_indices<K>(slot0);
        materialize_index_rank_order<K>(slot0, indices_cb);

        tile_regs_commit();
        tile_regs_wait();

        indices_cb_obj.reserve_back(1);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_cb_obj.push_back(1);

        tile_regs_release();
    }
}
