// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample {

/**
 * @brief Creates a multi-core program for 3D upsampling with DRAM interleaved memory layout
 *
 * @param input Input 5D tensor with shape (N, D, H, W, C) in row-major layout
 * @param output Output 5D tensor with shape (N, D*scale_d, H*scale_h, W*scale_w, C)
 * @param scale_factor_d Upsampling scale factor for depth dimension
 * @param scale_factor_h Upsampling scale factor for height dimension
 * @param scale_factor_w Upsampling scale factor for width dimension
 * @return ProgramWithCallbacks The program and its runtime argument update callback
 *
 * @note Currently only supports row-major layout with DRAM interleaved memory
 * @note Uses nearest-neighbor interpolation
 */
tt::tt_metal::operation::ProgramWithCallbacks upsample3d_multi_core_interleaved(
    const Tensor& input,
    Tensor& output,
    const uint32_t scale_factor_d,
    const uint32_t scale_factor_h,
    const uint32_t scale_factor_w);

}  // namespace ttnn::operations::upsample
