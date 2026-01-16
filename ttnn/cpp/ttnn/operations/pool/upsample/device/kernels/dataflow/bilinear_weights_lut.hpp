// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fixed_point_arithmetic.hpp"

//
// BilinearWeightsLUT: Compile-time lookup table for bilinear interpolation weights
//
// Purpose:
//   Pre-computes all bilinear interpolation weights needed for upsampling at compile time,
//   storing them as BF16 values for efficient runtime access. Each output pixel position
//   within an upsampling window has a fixed set of 4 weights for combining its 4 nearest
//   input neighbors.
//
// Template Parameters:
//   ScaleH, ScaleW: Upsampling scale factors (e.g., 2x2, 4x4)
//
// Data Layout:
//   For an NxN upsampling (e.g., 2x2 creates 4 output pixels per input pixel):
//   - LUT contains N*N weight sets, one for each unique output pixel position
//   - Each weight set occupies 2 uint32_t entries (8 bytes total):
//     * Entry 0: [w2 | w1] packed as two BF16 values (high 16 bits | low 16 bits)
//     * Entry 1: [w4 | w3] packed as two BF16 values (high 16 bits | low 16 bits)
//
// Weight Interpretation:
//   Given 4 nearest neighbors arranged as:
//     y1x1  y1x2
//     y2x1  y2x2
//
//   Weights represent:
//     w1 = (1-dx) * (1-dy)  →  top-left corner weight     (y1x1)
//     w2 = dx * (1-dy)      →  top-right corner weight    (y1x2)
//     w3 = (1-dx) * dy      →  bottom-left corner weight  (y2x1)
//     w4 = dx * dy          →  bottom-right corner weight (y2x2)
//
//   Where dx and dy are the fractional distances from the top-left neighbor to the
//   interpolated position, in the range [0, 1].
//
// Indexing:
//   For an output pixel at upsampling phase (phase_h, phase_w):
//     lut_index = (phase_h * ScaleW + phase_w) * 2
//     packed_w1_w2 = weights.data[lut_index]
//     packed_w3_w4 = weights.data[lut_index + 1]
//
// Example (2x2 upsampling):
//   Creates 4 weight sets for the 4 possible output positions within each 2x2 block:
//     Phase (0,0): weights for output pixel at relative position (0.25, 0.25) from top left input pixel
//     Phase (0,1): weights for output pixel at relative position (0.25, 0.75) from top left input pixel
//     Phase (1,0): weights for output pixel at relative position (0.75, 0.25) from top left input pixel
//     Phase (1,1): weights for output pixel at relative position (0.75, 0.75) from top left input pixel
//
//   Each output position has 4 pre-computed weights stored as BF16 in the LUT.
//
template <uint32_t ScaleH, uint32_t ScaleW>
struct BilinearWeightsLUT {
    static constexpr uint32_t NumWeightSets = ScaleH * ScaleW;
    static constexpr uint32_t Size = NumWeightSets * 2;

    static constexpr auto weights = []() {
        struct alignas(4) WeightArray {
            uint32_t data[ScaleH * ScaleW * 2];
        };
        WeightArray arr{};

        constexpr int32_t scale_h_inv = fixed_point_arithmetic::FIXED_ONE / ScaleH;
        constexpr int32_t scale_w_inv = fixed_point_arithmetic::FIXED_ONE / ScaleW;
        constexpr int32_t y_start =
            (fixed_point_arithmetic::FIXED_ONE / (2 * ScaleH)) - fixed_point_arithmetic::FIXED_HALF;
        constexpr int32_t x_start =
            (fixed_point_arithmetic::FIXED_ONE / (2 * ScaleW)) - fixed_point_arithmetic::FIXED_HALF;

        uint32_t idx = 0;
        for (uint32_t i = 0; i < ScaleH; ++i) {
            for (uint32_t j = 0; j < ScaleW; ++j) {
                int32_t src_y_fixed = y_start + static_cast<int32_t>(i) * scale_h_inv;
                int32_t src_x_fixed = x_start + static_cast<int32_t>(j) * scale_w_inv;

                int32_t dy_fixed = src_y_fixed - ((src_y_fixed >> fixed_point_arithmetic::FIXED_POINT_SHIFT)
                                                  << fixed_point_arithmetic::FIXED_POINT_SHIFT);
                int32_t dx_fixed = src_x_fixed - ((src_x_fixed >> fixed_point_arithmetic::FIXED_POINT_SHIFT)
                                                  << fixed_point_arithmetic::FIXED_POINT_SHIFT);

                if (dy_fixed < 0) {
                    dy_fixed += fixed_point_arithmetic::FIXED_ONE;
                }
                if (dx_fixed < 0) {
                    dx_fixed += fixed_point_arithmetic::FIXED_ONE;
                }

                int32_t one_minus_dy = fixed_point_arithmetic::FIXED_ONE - dy_fixed;
                int32_t one_minus_dx = fixed_point_arithmetic::FIXED_ONE - dx_fixed;

                int32_t w1_fixed = fixed_point_arithmetic::fixed_mul(one_minus_dx, one_minus_dy);
                int32_t w2_fixed = fixed_point_arithmetic::fixed_mul(dx_fixed, one_minus_dy);
                int32_t w3_fixed = fixed_point_arithmetic::fixed_mul(one_minus_dx, dy_fixed);
                int32_t w4_fixed = fixed_point_arithmetic::fixed_mul(dx_fixed, dy_fixed);

                uint16_t bf16_w1 = fixed_point_arithmetic::fixed_to_bf16(w1_fixed);
                uint16_t bf16_w2 = fixed_point_arithmetic::fixed_to_bf16(w2_fixed);
                uint16_t bf16_w3 = fixed_point_arithmetic::fixed_to_bf16(w3_fixed);
                uint16_t bf16_w4 = fixed_point_arithmetic::fixed_to_bf16(w4_fixed);

                arr.data[idx++] = (static_cast<uint32_t>(bf16_w2) << 16) | bf16_w1;
                arr.data[idx++] = (static_cast<uint32_t>(bf16_w4) << 16) | bf16_w3;
            }
        }
        return arr;
    }();
};
