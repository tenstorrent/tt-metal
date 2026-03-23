// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "experimental/circular_buffer.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    // Runtime args:
    // Total number of outer-loop iterations (N * C * Ht),
    // i.e. how many independent row-reductions this core must perform.
    uint32_t NCHt = get_arg_val<uint32_t>(0);

    // Compile-time args:
    // Number of tiles along the W (reduction) dimension.
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    // The actual number of elements along W (before tiling).
    constexpr uint32_t W = get_compile_time_arg_val(1);
    // Number of elements per tile in the W dimension
    // (typically 32, but can be smaller for narrow tiles).
    constexpr uint32_t tile_width = get_compile_time_arg_val(2);
    // Whether input scaling is required.
    constexpr bool do_scale = get_compile_time_arg_val(3) != 0;
    // Whether to apply Bessel's correction (divide by N-1 instead of N).
    constexpr bool correction = get_compile_time_arg_val(4) != 0;
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_compile_time_arg_val(5) != 0;

    constexpr uint32_t onetile = 1;

    // Circular buffer that the reader kernel fills with input tiles.
    constexpr auto cb_in = tt::CBIndex::c_0;
    // Scalar tile produced by the reader via generate_reduce_scaler.
    // Used to scale every input tile before Welford processing.
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    // Circular buffer where the final variance output tile is written
    // for the writer kernel to consume.
    constexpr auto cb_out = tt::CBIndex::c_16;
    // Scratch circular buffer used to hold the variance tile between
    // the two transpose steps (Welford produces row-oriented results;
    // we transpose back to column orientation via this buffer,
    // and transpose operation can't take data from the DST register).
    constexpr auto cb_var = tt::CBIndex::c_19;
    // 1-tile intermediate: holds the scaled input tile between the
    // mul_tiles_bcast_scalar (scale step) and transpose_wh_tile (Welford step).
    // Only used when do_scale is true.
    // The reason this is needed is because mul_tiles_bcast_scalar writes data
    // to the DST register, but transpose_wh_tile is an unpack operation, so
    // it expects data in a CB. Thus, this CB is used to hold the scaled input tile.
    constexpr auto cb_scaled = tt::CBIndex::c_20;
    // Circular buffer holding a pre-computed 1/n look-up table (one entry
    // per column index 1..W) that Welford's online algorithm uses to avoid
    // runtime division.
    //    constexpr auto cb_reciprocals = tt::CBIndex::c_25;

    // The CB that the transpose step reads from: cb_scaled when scaling,
    // cb_in directly when not.
    constexpr auto cb_transpose_src = do_scale ? cb_scaled : cb_in;

    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_scaler_obj(cb_scaler);
    experimental::CircularBuffer cb_out_obj(cb_out);
    experimental::CircularBuffer cb_var_obj(cb_var);
    experimental::CircularBuffer cb_scaled_obj(cb_scaled);
    experimental::CircularBuffer cb_transpose_src_obj(cb_transpose_src);

    // Destination register indices inside the Tensix DST register file.
    // Welford's LLK uses three adjacent dst registers:
    //   input_dst (0) – scratch for the current transposed input tile,
    //   mean_dst  (1) – running / final mean accumulator,
    //   var_dst   (2) – running / final variance accumulator.
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed.
    constexpr uint32_t last_tile_rows = (W % tile_width) == 0 ? tile_width : W % tile_width;

    compute_kernel_hw_startup(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);

    // Get pointer to the reciprocal LUT
    //    using recip_lut_t = std::array<uint32_t, W>;
    //    auto p_reciprocals = norm::kernel_util::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals,
    //    0);

    if constexpr (do_scale) {
        // Scalar tile stays resident across all rows
        cb_scaler_obj.wait_front(onetile);
    }

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Simultaneous calculation of E[x] and Var[x] using Welford's algorithm.
        // When do_scale is true, each input tile is first multiplied by the scalar,
        // packed to cb_scaled, then transposed and fed to welford_update.
        // When do_scale is false, input tiles are transposed directly from cb_in.
        // The Welford SFPU state (running mean in LREG4, M2 in LREG5) persists
        // across tile_regs_release/acquire cycles because LREGs are SFPU registers,
        // separate from the DST register file controlled by the semaphore.
        uint32_t start_N = 0;

        // Programs SFPU replay buffer + clears LREG4/5
        welford_init();

        for (uint32_t wt = 0; wt < Wt; ++wt) {
            if constexpr (do_scale) {
                // --- Scale step: multiply input tile by scalar ---
                cb_in_obj.wait_front(onetile);
                tile_regs_acquire();
                // Explicit srca reconfig is required because the output packing
                // phase (below) calls reconfig_data_format_srca(cb_var) which
                // sets the unpacker to cb_var's format (Float32 when fp32 dest
                // accumulation is enabled).  mul_tiles_bcast_scalar_init_short
                // does not fully reconfigure the data format, so without this
                // call the unpacker would read cb_in (Float16_b) data using the
                // stale Float32 format, producing garbage on the 2nd+ NCHt
                // iterations.
                reconfig_data_format_srca(cb_in);
                mul_tiles_bcast_scalar_init_short(cb_in, cb_scaler);
                mul_tiles_bcast_scalar(cb_in, cb_scaler, 0, 0, input_dst);
                tile_regs_commit();
                cb_in_obj.pop_front(1);
                cb_scaled_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_reconfig_data_format(cb_scaled);
                pack_tile(input_dst, cb_scaled);
                tile_regs_release();
                cb_scaled_obj.push_back(onetile);
            }

            // --- Transpose + Welford step ---
            cb_transpose_src_obj.wait_front(onetile);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_transpose_src);
            transpose_wh_init_short(cb_transpose_src);
            transpose_wh_tile(cb_transpose_src, 0, input_dst);

            if (wt < (Wt - 1)) {
                //            welford_update<W>(input_dst, start_N, *p_reciprocals);
                welford_update<0>(input_dst, start_N, {});
                tile_regs_commit();
                tile_regs_wait();
                tile_regs_release();
            } else {
                // Last tile: finalize and keep DST acquired for variance packing
                //        welford_update_rows<W>(input_dst, start_N, 0, last_tile_rows, *p_reciprocals);
                welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                // Store the mean and variance to the destination registers.
                // scale_idx controls the divisor for M2 -> variance conversion:
                //   correction=false: scale_idx = W-1, reciprocal = 1/W  (population variance)
                //   correction=true:  scale_idx = W-2, reciprocal = 1/(W-1) (sample variance)
                constexpr uint32_t scale_idx = correction ? (W - 2) : (W - 1);
                welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                tile_regs_commit();
            }
            cb_transpose_src_obj.pop_front(onetile);
            start_N += tile_width;
        }

        // Pack variance and transpose back to column format
        cb_var_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_var);
        pack_tile(var_dst, cb_var);
        tile_regs_release();
        cb_var_obj.push_back(onetile);

        cb_var_obj.wait_front(onetile);
        reconfig_data_format_srca(cb_var);
        transpose_wh_init_short(cb_var);
        tile_regs_acquire();
        transpose_wh_tile(cb_var, 0, var_dst);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(var_dst);
        }
        tile_regs_commit();
        cb_var_obj.pop_front(onetile);

        // Pack transposed variance to output
        cb_out_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(var_dst, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(onetile);

    }  // NCHt loop
}
