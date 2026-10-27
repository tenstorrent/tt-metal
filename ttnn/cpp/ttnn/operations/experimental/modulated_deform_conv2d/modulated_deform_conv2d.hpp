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

// A modulated deformable convolution (DCNv2), groups and deform_groups 1: bilinear sampling at
// the offset kernel positions, scaled by the mask, times the (K_h * K_w * C, C_out) weight, plus
// the bias and an optional ReLU. The offsets and the mask logits come in one tensor, as mmcv's
// ModulatedDeformConv2dPack offset conv emits them, and the mask's sigmoid runs in the op; a
// separate offset and post-sigmoid mask, torchvision's form, are not taken.
//
// For output pixel m = (b, h_o, w_o), kernel position kk = k_h * K_w + k_w and channel c:
//   y = h_o * stride_h - pad_h + k_h * dilation_h + offset_mask[b, h_o, w_o, 2 * kk]
//   x = w_o * stride_w - pad_w + k_w * dilation_w + offset_mask[b, h_o, w_o, 2 * kk + 1]
//   mask = sigmoid(offset_mask[b, h_o, w_o, 2 * K * K + kk])
//   column[m, kk * C + c] = mask * bilinear(input[b, :, :, c], y, x)
//   out[m, :] = column[m, :] @ weight + bias
// Corners outside the input contribute zero, as in mmcv and torchvision. The columns never
// leave L1: each is multiplied by the weight in the core that samples it.
//
// The columns are bfloat16 and the products run at HiFi2 fidelity; each kernel position's
// reduction accumulates in fp32, the sum over positions in bfloat16. Each core keeps the outputs
// of all its 32-row blocks and two weight slices in L1, and fails with the L1 it needs when
// they do not fit; C over 256 takes several passes per kernel position to keep the slices small.
//
// Shapes:
//   input        (B, H, W, C)                    ROW_MAJOR bfloat16, INTERLEAVED; C a multiple of 32
//   offset_mask  (B, H_out, W_out, 3*K*K)        ROW_MAJOR bfloat16, INTERLEAVED: the offsets in
//                                                pixels, (y, x) per kernel position, then the mask
//                                                logits
//   weight       (K*K*C, C_out)                  TILE bfloat16, INTERLEAVED, rows ordered (kk, c);
//                                                C_out a multiple of 32
//   bias         (1, C_out)                      TILE bfloat16, INTERLEAVED, optional
//   output       (1, 1, B*H_out*W_out, C_out)    TILE bfloat16, conv2d's layout
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
