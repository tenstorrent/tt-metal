// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_to_fp8, step 5: out = cast(input / scale) to e4m3, scale = clamp(amax,1e-4)/448
// per 128-element block.
//
// Per block = tile_h rows x 128 cols = 4 tiles for default 32-wide tiles:
//   Phase 1: cb_in -> cb_tile: tilize (ROW_MAJOR input) or copy (INPUT_TILE_LAYOUT: input already tiled).
//   Phase 2: compute per-row amax over the 128-element block, clamp(>=1e-4), multiply by 1/448
//            -> scale (col 0) -> cb_scale_tiles. recip(scale) -> 1/scale -> cb_inv_scale_tiles.
//   Phase 3: divide (cb_tile * bcast_col(cb_inv_scale_tiles)) into DST, then pack_untilize_dest straight
//            to cb_output_e4m3 (scaled, cast to e4m3) -- fused divide+untilize, no cb_out_tile round-trip.
// The writer extracts column 0 of cb_scale_tiles into the scale output [..., M, H/128].
//
// fp32_dest_acc_en=True (required for e4m3 on Blackhole; also gives fp32 reduce/divide precision).

#include <cstdint>

// Chunks (128 columns each) per work unit on the count-aware TILE FAST_1X32 path; 1 everywhere else.
#ifndef Q_UNIT
#define Q_UNIT 1
#endif

#include "api/compute/common.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"  // abs_tile / abs_tile_init
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/eltwise_unary/identity.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute/pow2_inv_scale_sfpu.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// The input is positive and normal because the amax is clamped before this
// helper runs. Preserve an exact power of two; otherwise increment its biased
// exponent and clear the mantissa.
template <int ITERATIONS = 8>
inline void calculate_ceil_power_of_two() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vInt exponent = sfpi::exexp(value, sfpi::ExponentMode::Biased);
        sfpi::vFloat mantissa = sfpi::setexp(value, 127);
        sfpi::vFloat result = sfpi::setexp(sfpi::vFloat(1.0f), exponent);
        v_if(mantissa != 1.0f) { result = sfpi::setexp(sfpi::vFloat(1.0f), exponent + 1); }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

inline void ceil_power_of_two_tile(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_ceil_power_of_two, (8 /* ITERATIONS */), idst, VectorMode::RC);
}

#endif

void kernel_main() {
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(0);
    CircularBuffer cb_in(cb_in_id);
    constexpr uint32_t cb_tile_id = get_compile_time_arg_val(1);
    CircularBuffer cb_tile(cb_tile_id);
    constexpr uint32_t cb_scaler_id = get_compile_time_arg_val(2);
    CircularBuffer cb_scaler(cb_scaler_id);
    constexpr uint32_t cb_abs_id = get_compile_time_arg_val(3);
    CircularBuffer cb_abs(cb_abs_id);
    constexpr uint32_t cb_scale_tiles_id = get_compile_time_arg_val(4);
    CircularBuffer cb_scale_tiles(cb_scale_tiles_id);
    constexpr uint32_t cb_inv_scale_tiles_id = get_compile_time_arg_val(5);
    CircularBuffer cb_inv_scale_tiles(cb_inv_scale_tiles_id);
    constexpr uint32_t cb_output_e4m3_id = get_compile_time_arg_val(6);
    CircularBuffer cb_output_e4m3(cb_output_e4m3_id);
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(7);
    constexpr uint32_t clamp_max_bits = get_compile_time_arg_val(8);
    constexpr uint32_t inv_448_bits = get_compile_time_arg_val(9);

    // Tile width from the tensor's tile spec.
    constexpr uint32_t block_w = 128;  // BlockW: the I/O chunk
    constexpr uint32_t tile_w = get_compile_time_arg_val(10);
    constexpr bool round_scale_to_power_of_two = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t scale_block_wt = get_compile_time_arg_val(12);  // tiles per scale block (1, 2 or 4)
    constexpr bool round_to_nearest_even = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t block_wt = block_w / tile_w;                     // BlockWt
    constexpr uint32_t chunk_scale_blocks = block_wt / scale_block_wt;  // scales per chunk row
    constexpr uint32_t block_ht = 1;                                    // BlockHt
    constexpr uint32_t tiles_per_block = block_ht * block_wt;

    constexpr uint32_t IDST0 = 0;
    constexpr uint32_t IDST1 = 1;

#ifdef TOKEN_COUNT_AWARE
    // num_blocks is computed on device by the reader (expert-buffer valid prefix) and mailed here.
    constexpr uint32_t cb_loop_count_id = get_compile_time_arg_val(14);
    CircularBuffer cb_loop_count(cb_loop_count_id);
    cb_loop_count.wait_front(1);
    uint32_t num_blocks = cb_loop_count.read_tile_value(0, 0);
    const uint32_t first_unit = cb_loop_count.read_tile_value(0, 1);  // global index of this core's first unit
    cb_loop_count.pop_front(1);
#else
    uint32_t num_blocks = get_arg_val<uint32_t>(0);  // tile_h x 128 blocks for this core
#endif

    // Configure the unpacker hw on the fp32 reduce/abs operand so num_faces / tile dims are full
    // (4 faces). Configuring on a bf16 cb_in instead leaves the fp32 reduce reading only 2 faces
    // (within-tile column reduce sees cols 0-15) for bf16 input; tilize_init re-inits for cb_in.
    compute_kernel_hw_startup(cb_abs_id, cb_output_e4m3_id);
    cb_scaler.wait_front(1);  // reader-filled 1.0 scaler, reused for every reduce

#ifdef FAST_1X32
    // 1x32 pow2 scales (one scale per tile row): three passes per 128-wide chunk, each one DEST acquire.
    //   A: |x| of the 4 tiles -> cb_abs.
    //   B: row max per tile (FPU reduce) -> one SFPU pass on column 0: scale and its exact 1/scale.
    //   C: x * bcast_col(1/scale) (FPU) -> pack_untilize to e4m3.
    // A bf16 / bf8_b TILE input is read straight from cb_in (exact in SrcA); ROW_MAJOR or fp32 input goes
    // through the fp32 cb_tile first.
#ifdef FAST_BLOCK4
    // 1x128: the 4 tiles of a chunk share one scale per row (one 1/scale tile per chunk).
    static_assert(scale_block_wt == 4 && tiles_per_block == 4, "FAST_BLOCK4: one scale per 4-tile chunk row");
    constexpr uint32_t inv_per_chunk = 1;
#else
    static_assert(scale_block_wt == 1 && tiles_per_block == 4, "FAST_1X32: one scale per tile row, 4-tile chunks");
    constexpr uint32_t inv_per_chunk = tiles_per_block;
#endif
#ifdef DIRECT_SRC
    constexpr uint32_t cb_src_id = cb_in_id;
#else
    constexpr uint32_t cb_src_id = cb_tile_id;
#endif
    CircularBuffer cb_src(cb_src_id);
    // num_blocks counts units of Q_UNIT chunks; each chunk's e4m3 rows land at its column of a Q_UNIT * 4 tile wide
    // output row, so the writer moves one Q_UNIT * 128 B run per row.
    constexpr uint32_t unit_tiles = Q_UNIT * tiles_per_block;
#if defined(DIRECT_SRC) && defined(INPUT_TILE_LAYOUT)
    // Tiled bf16 / bf8_b input: run each phase across the whole unit (Q_UNIT chunks) so every LLK init runs once
    // per unit instead of once per 4-tile chunk.
#ifndef Q_WGROUP
#define Q_WGROUP 1
#endif
#ifndef Q_UPR
#define Q_UPR 1
#endif
    // Write groups: Q_WGROUP consecutive units of one tile-row share one output block, each unit's chunks at their
    // column of a Q_WGROUP * unit wide row, so the writer moves one run of up to Q_WGROUP units per row.
    constexpr uint32_t group_tiles = Q_WGROUP * unit_tiles;
#ifndef TOKEN_COUNT_AWARE
    constexpr uint32_t first_unit = 0;
#endif
    uint32_t group_pos = 0;
    for (uint32_t unit = 0; unit < num_blocks; ++unit) {
        cb_src.wait_front(unit_tiles);

#ifdef Q_SFPU_AMAX
        // A+B in DEST: copy, then one fused SFPU pass per tile (|x| -> row max -> pow2 1/scale, kept in registers);
        // 1x128 (FAST_BLOCK4): row max per tile, then one fold of the 4 tiles' maxima -> pow2 into tile 0.
        constexpr uint32_t unit_inv_tiles = Q_UNIT * inv_per_chunk;
        cb_scale_tiles.reserve_back(unit_inv_tiles);
        cb_inv_scale_tiles.reserve_back(unit_inv_tiles);
        reconfig_full_operand_srca(cb_src_id);
        pack_reconfig_data_format(cb_inv_scale_tiles_id);
        for (uint32_t c = 0; c < Q_UNIT; ++c) {
            copy_init(cb_src_id);
            tile_regs_acquire();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                copy_tile(cb_src_id, c * tiles_per_block + k, k);
            }
            MATH((SFPU_UNARY_INIT(unused)));
#ifdef FAST_BLOCK4
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                MATH((amax_rows_tile_call(k)));
            }
            MATH((fold4_pow2_inv_scale_tile(0, clamp_min_bits, inv_448_bits)));
#else
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                MATH((amax_pow2_inv_scale_tile(k, clamp_min_bits, inv_448_bits)));
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < inv_per_chunk; ++k) {
                pack_tile(k, cb_inv_scale_tiles_id);
            }
            for (uint32_t k = 0; k < inv_per_chunk; ++k) {
                pack_tile(k, cb_scale_tiles_id);
            }
            tile_regs_release();
        }
        cb_scale_tiles.push_back(unit_inv_tiles);
        cb_inv_scale_tiles.push_back(unit_inv_tiles);
#else
        // A: |x|
        reconfig_full_operand_srca(cb_src_id);
        pack_reconfig_data_format(cb_abs_id);
        copy_init(cb_src_id);
        abs_tile_init();
        cb_abs.reserve_back(unit_tiles);
        for (uint32_t c = 0; c < Q_UNIT; ++c) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                copy_tile(cb_src_id, c * tiles_per_block + k, k);
                abs_tile(k);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                pack_tile(k, cb_abs_id);
            }
            tile_regs_release();
        }
        cb_abs.push_back(unit_tiles);

        // B: row max -> 1/scale, packed for the multiply and for the writer
        cb_abs.wait_front(unit_tiles);
        cb_scale_tiles.reserve_back(unit_tiles);
        cb_inv_scale_tiles.reserve_back(unit_tiles);
        reconfig_data_format(cb_abs_id, cb_scaler_id);
        pack_reconfig_data_format(cb_inv_scale_tiles_id);
        for (uint32_t c = 0; c < Q_UNIT; ++c) {
            tile_regs_acquire();
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_inv_scale_tiles_id);
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                    cb_abs_id, cb_scaler_id, c * tiles_per_block + k, 0, k);
            }
            reduce_uninit();
            MATH((SFPU_UNARY_INIT(unused)));
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                MATH((pow2_inv_scale_tile(k, clamp_min_bits, inv_448_bits)));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                pack_tile(k, cb_inv_scale_tiles_id);
            }
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                pack_tile(k, cb_scale_tiles_id);
            }
            tile_regs_release();
        }
        cb_scale_tiles.push_back(unit_tiles);
        cb_inv_scale_tiles.push_back(unit_tiles);
        cb_abs.pop_front(unit_tiles);
#endif

        // C: x * (1/scale) per row -> e4m3 rows, each chunk at its column of the unit-wide output rows
        cb_inv_scale_tiles.wait_front(Q_UNIT * inv_per_chunk);
        reconfig_data_format(cb_src_id, cb_inv_scale_tiles_id);
        mul_bcast_cols_init(cb_src_id, cb_inv_scale_tiles_id);
        pack_untilize_dest_init<tiles_per_block, group_tiles>(cb_output_e4m3_id);
        if (group_pos == 0) {
            cb_output_e4m3.reserve_back(group_tiles);
        }
        for (uint32_t c = 0; c < Q_UNIT; ++c) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                const uint32_t t = c * tiles_per_block + k;
                mul_tiles_bcast_cols(cb_src_id, cb_inv_scale_tiles_id, t, c * inv_per_chunk + k % inv_per_chunk, k);
            }
            if constexpr (round_to_nearest_even) {
                MATH((SFPU_UNARY_INIT(unused)));
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    MATH((round_to_e4m3_rne_tile(k)));
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<tiles_per_block, group_tiles>(cb_output_e4m3_id, 1, group_pos * Q_UNIT + c);
            tile_regs_release();
        }
        pack_untilize_uninit(cb_output_e4m3_id);
        const uint32_t g = first_unit + unit;
        if (group_pos + 1 == Q_WGROUP || (g + 1) % Q_UPR == 0 || unit + 1 == num_blocks) {
            cb_output_e4m3.push_back(group_tiles);
            group_pos = 0;
        } else {
            ++group_pos;
        }
        cb_src.pop_front(unit_tiles);
        cb_inv_scale_tiles.pop_front(Q_UNIT * inv_per_chunk);
    }
#else
    for (uint32_t unit = 0; unit < num_blocks; ++unit) {
        cb_output_e4m3.reserve_back(unit_tiles);
        for (uint32_t chunk = 0; chunk < Q_UNIT; ++chunk) {
#if defined(INPUT_TILE_LAYOUT) && !defined(DIRECT_SRC)
            reconfig_data_format_srca(cb_in_id);
            pack_reconfig_data_format(cb_tile_id);
            copy_init(cb_in_id);
            cb_in.wait_front(tiles_per_block);
            cb_tile.reserve_back(tiles_per_block);
            tile_regs_acquire();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                copy_tile(cb_in_id, k, k);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                pack_tile(k, cb_tile_id);
            }
            tile_regs_release();
            cb_tile.push_back(tiles_per_block);
            cb_in.pop_front(tiles_per_block);
#elif !defined(INPUT_TILE_LAYOUT)
            compute_kernel_lib::tilize<tiles_per_block, cb_in_id, cb_tile_id>(block_ht);
#endif
            cb_src.wait_front(tiles_per_block);

            // A: |x|
            {
                reconfig_full_operand_srca(cb_src_id);
                pack_reconfig_data_format(cb_abs_id);
                copy_init(cb_src_id);
                abs_tile_init();
                cb_abs.reserve_back(tiles_per_block);
                tile_regs_acquire();
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    copy_tile(cb_src_id, k, k);
                    abs_tile(k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    pack_tile(k, cb_abs_id);
                }
                tile_regs_release();
                cb_abs.push_back(tiles_per_block);
            }

            // B: row max -> 1/scale (one acquire for the 4 tiles), packed for the multiply and for the writer
            {
                cb_abs.wait_front(tiles_per_block);
                cb_scale_tiles.reserve_back(tiles_per_block);
                cb_inv_scale_tiles.reserve_back(tiles_per_block);
                reconfig_data_format(cb_abs_id, cb_scaler_id);
                pack_reconfig_data_format(cb_inv_scale_tiles_id);
                tile_regs_acquire();
                reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_inv_scale_tiles_id);
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, k, 0, k);
                }
                reduce_uninit();
                MATH((SFPU_UNARY_INIT(unused)));
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    MATH((pow2_inv_scale_tile(k, clamp_min_bits, inv_448_bits)));
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    pack_tile(k, cb_inv_scale_tiles_id);
                }
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    pack_tile(k, cb_scale_tiles_id);
                }
                tile_regs_release();
                cb_scale_tiles.push_back(tiles_per_block);
                cb_inv_scale_tiles.push_back(tiles_per_block);
                cb_abs.pop_front(tiles_per_block);
            }

            // C: x * (1/scale) per row -> e4m3 rows
            cb_inv_scale_tiles.wait_front(tiles_per_block);
            reconfig_data_format(cb_src_id, cb_inv_scale_tiles_id);
            mul_bcast_cols_init(cb_src_id, cb_inv_scale_tiles_id);
            pack_untilize_dest_init<tiles_per_block, unit_tiles>(cb_output_e4m3_id);
            tile_regs_acquire();
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                mul_tiles_bcast_cols(cb_src_id, cb_inv_scale_tiles_id, k, k, k);
            }
            if constexpr (round_to_nearest_even) {
                MATH((SFPU_UNARY_INIT(unused)));
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    MATH((round_to_e4m3_rne_tile(k)));
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<tiles_per_block, unit_tiles>(cb_output_e4m3_id, 1, chunk);
            tile_regs_release();
            pack_untilize_uninit(cb_output_e4m3_id);
            cb_src.pop_front(tiles_per_block);
            cb_inv_scale_tiles.pop_front(tiles_per_block);
        }
        cb_output_e4m3.push_back(unit_tiles);
    }
#endif  // DIRECT_SRC && INPUT_TILE_LAYOUT
#else
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        {
            // ----- Phase 1: get input tiles into fp32 cb_tile -----
#ifdef INPUT_TILE_LAYOUT
            // TILE input is already tiled, so copy instead of tilize. The fp32 copy is also what makes
            // bf16 correct: reading bf16 cb_in into the reduce would expose only 2 faces (cols 0-15)
            // and corrupt the per-128 amax.
            reconfig_data_format_srca(cb_in_id);
            pack_reconfig_data_format(cb_tile_id);
            copy_init(cb_in_id);
            cb_in.wait_front(tiles_per_block);
            cb_tile.reserve_back(tiles_per_block);
            for (uint32_t k = 0; k < tiles_per_block; ++k) {
                tile_regs_acquire();
                copy_tile(cb_in_id, k, IDST0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_tile_id);
                tile_regs_release();
            }
            cb_tile.push_back(tiles_per_block);
            cb_in.pop_front(tiles_per_block);
#else
            // ROW_MAJOR input: tilize the row-major block into tiles.
            compute_kernel_lib::tilize<tiles_per_block, cb_in_id, cb_tile_id>(block_ht);
#endif

            // ----- Phase 2: block amax -> scale (col 0) and 1/scale (col 0) -----
            cb_tile.wait_front(tiles_per_block);  // read by index; popped after the divide
            for (uint32_t block_h_idx = 0; block_h_idx < block_ht; ++block_h_idx) {
                // Abs the block row's tiles into cb_abs. Use reconfig_full_operand_srca so the SrcA tile/face
                // geometry is reprogrammed for cb_tile too (not just the format): after a bf16 tilize the fp32
                // cb_tile would otherwise be read with a stale geometry and copy_tile would misread it
                // (corrupting the amax for bf16 input).
                reconfig_full_operand_srca(cb_tile_id);
                pack_reconfig_data_format(cb_abs_id);
                copy_init(cb_tile_id);
                cb_abs.reserve_back(block_wt);
                abs_tile_init();
                for (uint32_t k = 0; k < block_wt; ++k) {
                    tile_regs_acquire();
                    copy_tile(cb_tile_id, block_h_idx * block_wt + k, IDST0);
                    abs_tile(IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_abs_id);
                    tile_regs_release();
                }
                cb_abs.push_back(block_wt);

                // Per scale block: reduce -> per-row max (col 0), collapse the block's tiles, clamp,
                // *1/448 -> scale (slot 0); copy to slot 1 and recip -> 1/scale (slot 1). One acquire
                // produces both, so each block row's 1/scale is its own scale (no CB reload, which would
                // always read row 0). A 128 chunk holds chunk_scale_blocks scale blocks (1 for 128-wide).
                cb_abs.wait_front(block_wt);
                for (uint32_t sb = 0; sb < chunk_scale_blocks; ++sb) {
                    cb_scale_tiles.reserve_back(1);
                    cb_inv_scale_tiles.reserve_back(1);
                    tile_regs_acquire();
                    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_scale_tiles_id);
                    for (uint32_t k = 0; k < scale_block_wt; ++k) {
                        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                            cb_abs_id, cb_scaler_id, sb * scale_block_wt + k, 0, k);
                    }
                    reduce_uninit();

                    if constexpr (scale_block_wt > 1) {
                        binary_max_tile_init();
                        for (uint32_t i = 1; i < scale_block_wt; i++) {
                            binary_max_tile(IDST0, i, IDST0);  // slot 0 = amax
                        }
                    }

                    clamp_tile_init();
                    clamp_tile(IDST0, clamp_min_bits, clamp_max_bits);  // slot 0 = clamp(amax)
                    binop_with_scalar_tile_init();
                    mul_unary_tile(IDST0, inv_448_bits);  // slot 0 = scale = clamp(amax)/448
                    if constexpr (round_scale_to_power_of_two) {
                        // UE8M0-style scale: 2^ceil(log2(scale)), formed directly from the
                        // float32 exponent and mantissa so power-of-two boundaries are exact.
                        MATH((ceil_power_of_two_tile(IDST0)));
                    }
                    copy_dest_values_init();
                    copy_dest_values<DataFormat::Float32>(IDST0, IDST1);  // slot 1 = scale
                    recip_tile_init();
                    recip_tile(IDST1);  // slot 1 = 1/scale (col 0 valid; other cols unused by bcast)
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile(IDST0, cb_scale_tiles_id);      // scale output
                    pack_tile(IDST1, cb_inv_scale_tiles_id);  // 1/scale for the divide (same fp32 format)
                    tile_regs_release();

                    cb_scale_tiles.push_back(1);
                    cb_inv_scale_tiles.push_back(1);
                }

                cb_abs.pop_front(block_wt);
            }

            // ----- Phase 3: divide (cb_tile * bcast_col(1/scale)) into DST, then pack_untilize straight
            // to e4m3 -- fused divide+untilize, no cb_out_tile L1 round-trip and no separate untilize pass.
            // In 32-bit DST (fp32_dest_acc) the half-sync pack-untilize cap is 4 tiles = one block. -----
            reconfig_data_format(cb_tile_id, cb_inv_scale_tiles_id);
            mul_bcast_cols_init(cb_tile_id, cb_inv_scale_tiles_id);
            pack_untilize_dest_init<tiles_per_block, tiles_per_block>(cb_output_e4m3_id);
            cb_inv_scale_tiles.wait_front(block_ht * chunk_scale_blocks);
            cb_output_e4m3.reserve_back(tiles_per_block);
            tile_regs_acquire();
            for (uint32_t block_h_idx = 0; block_h_idx < block_ht; ++block_h_idx) {
                for (uint32_t k = 0; k < block_wt; ++k) {
                    mul_tiles_bcast_cols(
                        cb_tile_id,
                        cb_inv_scale_tiles_id,
                        block_h_idx * block_wt + k,
                        block_h_idx * chunk_scale_blocks + k / scale_block_wt,
                        block_h_idx * block_wt + k);
                }
            }
            if constexpr (round_to_nearest_even) {
                MATH((SFPU_UNARY_INIT(unused)));
                for (uint32_t k = 0; k < tiles_per_block; ++k) {
                    MATH((round_to_e4m3_rne_tile(k)));
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<tiles_per_block>(cb_output_e4m3_id);
            tile_regs_release();
            cb_output_e4m3.push_back(tiles_per_block);
            cb_tile.pop_front(tiles_per_block);
            cb_inv_scale_tiles.pop_front(block_ht * chunk_scale_blocks);
            pack_untilize_uninit(cb_output_e4m3_id);
        }
    }
#endif
}
