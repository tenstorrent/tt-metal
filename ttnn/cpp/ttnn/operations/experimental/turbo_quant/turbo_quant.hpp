// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/turbo_quant_device_operation.hpp"

namespace ttnn {

// Fused bucketize: normalised rotated values → bucket indices (BF16).
// boundaries: inner boundary values (size = 2^bits − 1).
Tensor turbo_quant_bucketize(const Tensor& input_tensor, const std::vector<float>& boundaries);

// Fused gather centroids: integer indices (BF16) → centroid float values.
// centroids: centroid values (size = 2^bits).
Tensor turbo_quant_gather_centroids(const Tensor& input_tensor, const std::vector<float>& centroids);

}  // namespace ttnn
