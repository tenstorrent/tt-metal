// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/turbo_quant_device_operation.hpp"
#include "sdpa/device/sdpa_tq_device_operation.hpp"

namespace ttnn {

// Fused bucketize: normalised rotated values → bucket indices (BF16).
// boundaries: inner boundary values (size = 2^bits − 1).
Tensor turbo_quant_bucketize(const Tensor& input_tensor, const std::vector<float>& boundaries);

// Fused gather centroids: integer indices (BF16) → centroid float values.
// centroids: centroid values (size = 2^bits).
Tensor turbo_quant_gather_centroids(const Tensor& input_tensor, const std::vector<float>& centroids);

// Fused TurboQuant SDPA decode: reads BFP4 indices + BF16 norms from paged
// cache, dequantizes on-the-fly, runs SDPA. No full-cache BF16 temporary.
Tensor turbo_quant_sdpa_decode(
    const Tensor& q,
    const Tensor& k_indices,
    const Tensor& k_norms,
    const Tensor& v_indices,
    const Tensor& v_norms,
    const Tensor& page_table,
    const Tensor& cur_pos,
    const std::vector<float>& centroids,
    float scale,
    bool pre_rescaled = false);

}  // namespace ttnn
