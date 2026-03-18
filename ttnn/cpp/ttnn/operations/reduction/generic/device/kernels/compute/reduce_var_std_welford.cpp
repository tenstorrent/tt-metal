// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/welford.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "experimental/circular_buffer.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    // Runtime arg 0: total number of outer-loop iterations (N * C * Ht),
    // i.e. how many independent row-reductions this core must perform.
    uint32_t NCHt = get_arg_val<uint32_t>(0);

    // Number of tiles along the W (reduction) dimension.
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    // The actual number of elements along W (before tiling).
    constexpr uint32_t W = get_compile_time_arg_val(1);
    // Number of elements per tile in the W dimension
    // (typically 32, but can be smaller for narrow tiles).
    constexpr uint32_t tile_width = get_compile_time_arg_val(2);

    // Convenience constant: a single tile count.
    constexpr uint32_t onetile = 1;

    // Circular buffer that the reader kernel fills with input tiles.
    constexpr auto cb_in = tt::CBIndex::c_0;
    // Circular buffer where the final variance output tile is written
    // for the writer kernel to consume.
    constexpr auto cb_out = tt::CBIndex::c_16;
    // Scratch circular buffer used to hold the variance tile between
    // the two transpose steps (Welford produces row-oriented results;
    // we transpose back to column orientation via this buffer).
    constexpr auto cb_var = tt::CBIndex::c_19;
    // Circular buffer holding a pre-computed 1/n look-up table (one entry
    // per column index 1..W) that Welford's online algorithm uses to avoid
    // runtime division.
    constexpr auto cb_reciprocals = tt::CBIndex::c_25;

    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_out_obj(cb_out);
    experimental::CircularBuffer cb_var_obj(cb_var);

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
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = norm::kernel_util::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Simultaneous calculation of E[x] and Var[x] using Welford's algorithm
        uint32_t start_N = 0;
        reconfig_data_format_srca(cb_in);
        transpose_wh_init_short(cb_in);
        tile_regs_acquire();
        welford_init();
        // Process all but the last tile
        for (uint32_t wt = 0; wt < (Wt - 1); ++wt) {
            cb_in_obj.wait_front(onetile);
            // Welford's needs transposed input tile
            transpose_wh_tile(cb_in, 0, input_dst);
            welford_update<W>(input_dst, start_N, *p_reciprocals);
            cb_in_obj.pop_front(1);
            start_N += tile_width;
        }

        // Process the last tile
        // cb_in is synced on full blocks, so we need to wait for the
        // last tile + any remaining in the last block
        cb_in_obj.wait_front(onetile);
        transpose_wh_tile(cb_in, 0, input_dst);
        cb_in_obj.pop_front(1);
        welford_update_rows<W>(input_dst, start_N, 0, last_tile_rows, *p_reciprocals);

        // Store the mean and variance to the destination registers
        welford_finalize_to_row<W>(mean_dst, W - 1, *p_reciprocals);
        tile_regs_commit();

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
