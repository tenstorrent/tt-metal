// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared compute kernel for fused multi-scale deformable attention. Used
// unchanged by both the V1 and the V2 reader — the readers differ only in which
// operands they stage, never in what the sample stream means.
//
// Two phases, interleaved (see below), per output block of up to 32 queries of
// one (batch, head), spanning n_d_tiles tiles side by side for D > 32:
//
//   GEOMETRY   msda_geometry.hpp, on the SFPU. Per sampling point it turns the
//              staged (location | reference + offset) and attn column-0 tiles
//              into floor(px), floor(py) — handed back to the reader, which
//              bounds-tests them and gathers the corners — and the four
//              `attn * bilinear_corner_coefficient` scalar tiles.
//
//   REDUCTION  for each of 4 * L * P (input_tiles, scalar_tile) groups:
//                for each d-tile k:
//                  dest[row, col] = input_tile_k[row, col] * scalar_tile[row, 0]
//                  pack into output_cb slot k, accumulating in L1
//
// The four consecutive corner groups of a point therefore sum to the
// bilinear-interpolated, attention-weighted sample, and the L * P points sum
// into the same accumulator. That accumulator is the op's output: no per-sample
// value is ever written back to DRAM, and no [B, Q, H, L, P, D] tensor exists at
// any point. The scalar tile is shared across a group's d-tiles — the weight is
// per query row and independent of D.
//
// Why the geometry is interleaved with the reduction
// --------------------------------------------------
// The reader cannot gather point j's corners until this kernel has floored
// them, and this kernel cannot reduce point j until the reader has gathered
// them. Solving one point ahead of the reduction is what keeps both busy: while
// the reader gathers point j, the SFPU is already solving point j+1.
//
// Reader contract (see fused_msda_reader_common.hpp):
//   * geom / offset / attn tiles: column 0 holds one query per row, bf16 0 past
//     v_rows; columns 1..31 are zeroed once per CB slot.
//   * input_tile: rows that are in range AND in bounds hold the gathered value
//     stick, every other row is zeroed. This kernel builds the scalar and so
//     cannot mask an out-of-bounds corner itself.
//
// fp32_dest_acc_en is required — see the header note in msda_geometry.hpp.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/compute/msda_geometry.hpp"

constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t n_d_tiles = get_compile_time_arg_val(3);  // = ceil(D / 32)

// Geometry pipes. The reader stages the operands as column-0 tiles here and
// takes back the floored corner per axis.
constexpr uint32_t geom_x_cb_index = get_compile_time_arg_val(4);
constexpr uint32_t geom_y_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t offset_x_cb_index = get_compile_time_arg_val(6);
constexpr uint32_t offset_y_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t attn_tile_cb_index = get_compile_time_arg_val(8);
constexpr uint32_t x0_cb_index = get_compile_time_arg_val(9);
constexpr uint32_t y0_cb_index = get_compile_time_arg_val(10);
constexpr uint32_t frac_x_cb_index = get_compile_time_arg_val(11);
constexpr uint32_t frac_y_cb_index = get_compile_time_arg_val(12);

constexpr uint32_t NUM_LEVELS = get_compile_time_arg_val(13);
constexpr uint32_t NUM_POINTS = get_compile_time_arg_val(14);
constexpr bool FROM_OFFSETS = get_compile_time_arg_val(15) != 0;

constexpr uint32_t POINTS_PER_BLOCK = NUM_LEVELS * NUM_POINTS;

// Runtime args: [0] output tiles, then AXIS_CONSTANTS_PER_LEVEL uint32 per
// level. Must stay in step with the emission loop in
// fused_msda_program_factory.cpp.
constexpr uint32_t AXIS_CONSTANTS_PER_LEVEL = 6;

void kernel_main() {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);

    // Per-level normalized -> pixel constants, folded on the host so that
    // align_corners / locations_in_grid_space / from_offsets never become a
    // branch on device. See fused_msda_program_factory.cpp::axis_constants.
    fused_msda_geometry::AxisConstants kx[NUM_LEVELS];
    fused_msda_geometry::AxisConstants ky[NUM_LEVELS];
    for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
        const uint32_t base = 1 + AXIS_CONSTANTS_PER_LEVEL * l;
        kx[l].primary_scale = get_arg_val<uint32_t>(base + 0);
        kx[l].secondary_scale = get_arg_val<uint32_t>(base + 1);
        kx[l].bias = get_arg_val<uint32_t>(base + 2);
        ky[l].primary_scale = get_arg_val<uint32_t>(base + 3);
        ky[l].secondary_scale = get_arg_val<uint32_t>(base + 4);
        ky[l].bias = get_arg_val<uint32_t>(base + 5);
    }

    constexpr fused_msda_geometry::GeometryPipes geometry_cb{
        .geom_x = geom_x_cb_index,
        .geom_y = geom_y_cb_index,
        .offset_x = offset_x_cb_index,
        .offset_y = offset_y_cb_index,
        .attn = attn_tile_cb_index,
        .x0 = x0_cb_index,
        .y0 = y0_cb_index,
        .frac_x = frac_x_cb_index,
        .frac_y = frac_y_cb_index,
        .scalar = scalar_cb_index,
    };

    CircularBuffer input_cb(input_cb_index);
    CircularBuffer scalar_cb(scalar_cb_index);
    CircularBuffer output_cb(output_cb_index);

    compute_kernel_hw_startup(input_cb_index, scalar_cb_index, output_cb_index);

    // Seeded with what startup left SrcA on (icb0), so the first geometry copy
    // reconfigures rather than comparing a CB against itself. Every CB in this
    // op is bf16, so the reconfigurations are no-ops today; the tracking exists
    // so that can stop being true safely.
    uint32_t srca_cb = input_cb_index;

    for (uint32_t out = 0; out < num_output_tiles; ++out) {
        // Reserve the block's output tiles up front; the reduction accumulates
        // into them in L1, so they stay resident for every point of the block.
        output_cb.reserve_back(n_d_tiles);

        auto solve_geometry = [&](uint32_t j) {
            const uint32_t l = j / NUM_POINTS;
            // The geometry packs plain results. The previous point's reduction
            // left L1 accumulate enabled, so it has to be cleared before these
            // packs or they would add into the accumulator.
            pack_reconfig_l1_acc(0);
            fused_msda_geometry::point<FROM_OFFSETS>(geometry_cb, kx[l], ky[l], srca_cb);
        };

        // One point of lookahead, matching the reader's: point j+1 is solved on
        // the SFPU while the reader gathers point j's corners over the NoC.
        solve_geometry(0);

        for (uint32_t j = 0; j < POINTS_PER_BLOCK; ++j) {
            if (j + 1 < POINTS_PER_BLOCK) {
                solve_geometry(j + 1);
            }

            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(input_cb_index, scalar_cb_index);
            srca_cb = input_cb_index;

            for (uint32_t c = 0; c < 4; ++c) {
                // The block's first corner overwrites the accumulator; every
                // other corner of every other point accumulates into it.
                // solve_geometry left this at 0.
                pack_reconfig_l1_acc((j == 0 && c == 0) ? 0 : 1);

                input_cb.wait_front(n_d_tiles);
                scalar_cb.wait_front(1);

                for (uint32_t k = 0; k < n_d_tiles; ++k) {
                    tile_regs_acquire();
                    mul_tiles_bcast<BroadcastType::COL>(input_cb_index, scalar_cb_index, k, 0, 0);
                    tile_regs_commit();

                    tile_regs_wait();
                    // out_of_order_output=true so each iteration packs to an
                    // explicit slot (= k); the L1-acc mode then decides
                    // overwrite vs accumulate. The default would auto-advance
                    // the write pointer and clobber L1 past the block after the
                    // first group.
                    pack_tile<true>(0, output_cb_index, k);
                    tile_regs_release();
                }

                input_cb.pop_front(n_d_tiles);
                scalar_cb.pop_front(1);
            }
        }

        pack_reconfig_l1_acc(0);
        output_cb.push_back(n_d_tiles);
    }
}
