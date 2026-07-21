// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_to_fp8, step 5: out = cast(input / scale) to e4m3, scale = clamp(amax,1e-4)/448
// per 128-element block.
//
// Per block = tile_h rows x 128 cols = 4 tiles for default 32-wide tiles:
//   Phase 1: tilize cb_in -> cb_tile.
//   Phase 2: compute per-row amax over the 128-element block, clamp(>=1e-4), multiply by 1/448
//            -> scale (col 0) -> cb_scale_tiles. recip(scale) -> 1/scale -> cb_inv_scale_tiles.
//   Phase 3: divide (cb_tile * bcast_col(cb_inv_scale_tiles)) into DST, then pack_untilize_dest straight
//            to cb_output_e4m3 (scaled, cast to e4m3) -- fused divide+untilize, no cb_out_tile round-trip.
// The writer extracts column 0 of cb_scale_tiles into the scale output [..., M, H/128].
//
// fp32_dest_acc_en=True (required for e4m3 on Blackhole; also gives fp32 reduce/divide precision).

#include <cstdint>

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
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

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
    constexpr uint32_t cb_out_tile_id = get_compile_time_arg_val(6);
    CircularBuffer cb_out_tile(cb_out_tile_id);
    constexpr uint32_t cb_output_e4m3_id = get_compile_time_arg_val(7);
    CircularBuffer cb_output_e4m3(cb_output_e4m3_id);
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(8);
    constexpr uint32_t clamp_max_bits = get_compile_time_arg_val(9);
    constexpr uint32_t inv_448_bits = get_compile_time_arg_val(10);

    // Tile width from the tensor's tile spec.
    constexpr uint32_t block_w = 128;  // BlockW
    constexpr uint32_t tile_w = get_compile_time_arg_val(11);
    constexpr bool round_scale_to_power_of_two = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t block_wt = block_w / tile_w;  // BlockWt
    constexpr uint32_t block_ht = 1;                 // BlockHt
    constexpr uint32_t tiles_per_block = block_ht * block_wt;

    constexpr uint32_t IDST0 = 0;
    constexpr uint32_t IDST1 = 1;

    uint32_t num_blocks = get_arg_val<uint32_t>(0);  // tile_h x 128 blocks for this core

    // Configure the unpacker hw on the fp32 reduce/abs operand so num_faces / tile dims are full
    // (4 faces). Configuring on a bf16 cb_in instead leaves the fp32 reduce reading only 2 faces
    // (within-tile column reduce sees cols 0-15) for bf16 input; tilize_init re-inits for cb_in.
    compute_kernel_hw_startup(cb_abs_id, cb_output_e4m3_id);
    cb_scaler.wait_front(1);  // reader-filled 1.0 scaler, reused for every reduce

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        {
            // ----- Phase 1: tilize input row-major -> tile -----
            compute_kernel_lib::tilize<tiles_per_block, cb_in_id, cb_tile_id>(block_ht);

            // ----- Phase 2: block amax -> scale (col 0) and 1/scale (col 0) -----
            cb_tile.wait_front(tiles_per_block);  // read by index; popped after the divide
            for (uint32_t block_h_idx = 0; block_h_idx < block_ht; ++block_h_idx) {
                // Abs the block row's tiles into cb_abs. Force the SrcA tile-dim/stride reconfig
                // (is_tile_dim_reconfig_en=true): the default reconfig keeps the prior element
                // stride, so after a bf16 tilize the fp32 cb_tile would be read with a 2-byte
                // stride and copy_tile would misread it (corrupting the amax for bf16 input).
                reconfig_data_format_srca<false, true>(cb_tile_id);
                pack_reconfig_data_format(cb_abs_id);
                copy_tile_init(cb_tile_id);
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

                // reduce -> per-row max (col 0), accumulate, clamp, *1/448 -> scale (slot 0);
                // copy to slot 1 and recip -> 1/scale (slot 1). One acquire produces both, so each
                // block row's 1/scale is its own scale (no CB reload, which would always read row 0).
                cb_abs.wait_front(block_wt);
                cb_scale_tiles.reserve_back(1);
                cb_inv_scale_tiles.reserve_back(1);
                tile_regs_acquire();
                reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_scale_tiles_id);
                for (uint32_t k = 0; k < block_wt; ++k) {
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, k, 0, k);
                }
                reduce_uninit();

                binary_max_tile_init();
                for (uint32_t i = 1; i < block_wt; i++) {
                    binary_max_tile(IDST0, i, IDST0);  // slot 0 = amax
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
                recip_tile(IDST1);  // slot 1 = 1/scale (col 0 valid; other cols = 1/0 = inf, unused by bcast)
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(IDST0, cb_scale_tiles_id);      // scale output
                pack_tile(IDST1, cb_inv_scale_tiles_id);  // 1/scale for the divide (same fp32 format)
                tile_regs_release();

                cb_scale_tiles.push_back(1);
                cb_inv_scale_tiles.push_back(1);

                cb_abs.pop_front(block_wt);
            }

            // ----- Phase 3: divide (cb_tile * bcast_col(1/scale)) into DST, then pack_untilize straight
            // to e4m3 -- fused divide+untilize, no cb_out_tile L1 round-trip and no separate untilize pass.
            // In 32-bit DST (fp32_dest_acc) the half-sync pack-untilize cap is 4 tiles = one block. -----
            reconfig_data_format(cb_tile_id, cb_inv_scale_tiles_id);
            mul_bcast_cols_init_short(cb_tile_id, cb_inv_scale_tiles_id);
            pack_untilize_dest_init<tiles_per_block, tiles_per_block>(cb_output_e4m3_id);
            cb_inv_scale_tiles.wait_front(block_ht);
            cb_output_e4m3.reserve_back(tiles_per_block);
            tile_regs_acquire();
            for (uint32_t block_h_idx = 0; block_h_idx < block_ht; ++block_h_idx) {
                for (uint32_t k = 0; k < block_wt; ++k) {
                    mul_tiles_bcast_cols(
                        cb_tile_id,
                        cb_inv_scale_tiles_id,
                        block_h_idx * block_wt + k,
                        block_h_idx,
                        block_h_idx * block_wt + k);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<tiles_per_block>(cb_output_e4m3_id);
            tile_regs_release();
            cb_output_e4m3.push_back(tiles_per_block);
            cb_tile.pop_front(tiles_per_block);
            cb_inv_scale_tiles.pop_front(block_ht);
            pack_untilize_uninit(cb_output_e4m3_id);
        }
    }
}
