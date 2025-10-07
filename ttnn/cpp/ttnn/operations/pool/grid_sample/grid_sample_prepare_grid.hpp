// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <vector>
#include <optional>

namespace ttnn {
namespace operations {
namespace grid_sample {

/**
 * Precomputes grid sample data for optimized kernel execution.
 *
 * This function takes a normalized grid tensor and precomputes the pixel coordinates
 * and bilinear interpolation weights needed for grid sampling. This preprocessing
 * allows for more efficient kernel execution by avoiding repeated coordinate
 * transformations during the actual sampling operation.
 *
 * @param grid Grid tensor of shape (N, H_out, W_out, 2) with normalized coordinates in [-1, 1]
 * @param input_shape Array containing input tensor dimensions [N, H_in, W_in, C] in NHWC format
 * @param padding_mode How to handle out-of-bounds coordinates (currently only "zeros" supported)
 * @param output_dtype Data type for the output tensor (default: bfloat16)
 *
 * @return Precomputed grid tensor of shape (N, H_out, W_out, 6) where:
 *         - [:, :, :, 0]: North-west height coordinate (as integer stored in bfloat16)
 *         - [:, :, :, 1]: North-west width coordinate (as integer stored in bfloat16)
 *         - [:, :, :, 2]: Weight for north-west pixel
 *         - [:, :, :, 3]: Weight for north-east pixel
 *         - [:, :, :, 4]: Weight for south-west pixel
 *         - [:, :, :, 5]: Weight for south-east pixel
 *
 * The function expects the input grid to be on host with float32 data type and
 * returns a tensor on host with the specified output data type.
 */
ttnn::Tensor prepare_grid_sample_grid(
    const ttnn::Tensor& grid,
    const std::vector<uint32_t>& input_shape,
    const std::string& padding_mode = "zeros",
    const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace grid_sample
}  // namespace operations
}  // namespace ttnn
