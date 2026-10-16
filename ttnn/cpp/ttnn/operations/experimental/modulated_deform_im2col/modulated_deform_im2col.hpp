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
// The offsets and the mask logits come in one tensor, as mmcv's
// ModulatedDeformConv2dPack offset conv emits them, and the mask's sigmoid runs in
// the op; a separate offset and post-sigmoid mask, torchvision's form, are not taken.
//
// For output pixel m = (b, h_o, w_o), kernel position kk = k_h * K_w + k_w and
// channel c:
//   y = h_o * stride_h - pad_h + k_h * dilation_h + offset_mask[b, h_o, w_o, 2 * kk]
//   x = w_o * stride_w - pad_w + k_w * dilation_w + offset_mask[b, h_o, w_o, 2 * kk + 1]
//   mask = sigmoid(offset_mask[b, h_o, w_o, 2 * K * K + kk])
//   out[m, kk * C + c] = mask * bilinear(input[b, :, :, c], y, x)
// Corners outside the input contribute zero, as in mmcv and torchvision.
//
// Shapes (inputs ROW_MAJOR bfloat16, INTERLEAVED):
//   input        (B, H, W, C)                  C a multiple of 32
//   offset_mask  (B, H_out, W_out, 3*K*K)      the offsets in pixels, (y, x) per kernel
//                                              position, then the mask logits
//   output       (1, 1, B*H_out*W_out, K*K*C)  TILE bfloat16
ttnn::Tensor modulated_deform_im2col(
    const ttnn::Tensor& input,
    const ttnn::Tensor& offset_mask,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride = {1, 1},
    std::array<uint32_t, 2> padding = {0, 0},
    std::array<uint32_t, 2> dilation = {1, 1},
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

// The whole modulated deformable convolution: the columns above multiplied by the
// (K_h * K_w * C, C_out) weight inside the op, so they never reach DRAM, plus the bias
// and an optional ReLU. The columns are bfloat16, the K*K*C reduction accumulates in fp32, and
// the output is rounded to bfloat16 after the bias and the ReLU.
//
// Each core keeps the fp32 outputs of all its 32-row blocks and two (C, C_out) weight slices in
// L1, so the op fits only while rows per core, C and C_out stay small; it fails with the L1 it
// needs otherwise. BEVFormer's layer4 (C = C_out = 512) does not fit on Blackhole.
//
// Shapes (as above, and):
//   weight  (K*K*C, C_out)              TILE bfloat16, INTERLEAVED, rows ordered (kk, c);
//                                       C_out a multiple of 32
//   bias    (1, C_out)                  TILE bfloat16, INTERLEAVED, optional
//   output  (1, 1, B*H_out*W_out, C_out) TILE bfloat16
ttnn::Tensor modulated_deform_conv2d(
    const ttnn::Tensor& input,
    const ttnn::Tensor& offset_mask,
    const ttnn::Tensor& weight,
    const std::optional<const ttnn::Tensor>& bias,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride = {1, 1},
    std::array<uint32_t, 2> padding = {0, 0},
    std::array<uint32_t, 2> dilation = {1, 1},
    bool relu = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
