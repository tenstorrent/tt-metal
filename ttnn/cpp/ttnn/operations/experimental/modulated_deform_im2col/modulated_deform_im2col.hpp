// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

// The column matrix of a modulated deformable convolution (DCNv2): bilinear
// sampling at the offset kernel positions, scaled by the mask, laid out so that
// one matmul with the (K_h * K_w * C, C_out) weight finishes the convolution.
//
// For output pixel m = (b, h_o, w_o), kernel position kk = k_h * K_w + k_w and
// channel c:
//   y = h_o * stride_h - pad_h + k_h * dilation_h + offset[b, h_o, w_o, 2 * kk]
//   x = w_o * stride_w - pad_w + k_w * dilation_w + offset[b, h_o, w_o, 2 * kk + 1]
//   out[m, kk * C + c] = mask[b, h_o, w_o, kk] * bilinear(input[b, :, :, c], y, x)
// Corners outside the input contribute zero, as in mmcv and torchvision.
//
// Shapes (inputs ROW_MAJOR bfloat16, INTERLEAVED):
//   input   (B, H, W, C)               C a multiple of 32
//   offset  (B, H_out, W_out, 2*K*K)   pixels, (y, x) per kernel position
//   mask    (B, H_out, W_out, K*K)     the modulation, after its sigmoid
//   output  (1, 1, B*H_out*W_out, K*K*C) TILE bfloat16
ttnn::Tensor modulated_deform_im2col(
    const ttnn::Tensor& input,
    const ttnn::Tensor& offset,
    const ttnn::Tensor& mask,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride = {1, 1},
    std::array<uint32_t, 2> padding = {0, 0},
    std::array<uint32_t, 2> dilation = {1, 1},
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
