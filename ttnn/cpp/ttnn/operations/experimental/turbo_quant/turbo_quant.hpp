// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

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
//
// When return_lse is true, the result is a 2-element vector [out, lse] where
// lse holds LSE = max + log(sum) per (B, NQH) for the host-level sliding-
// window hybrid combine. Otherwise the result is a 1-element vector [out].
std::vector<Tensor> turbo_quant_sdpa_decode(
    const Tensor& q,
    const Tensor& k_indices,
    const Tensor& k_norms,
    const Tensor& v_indices,
    const Tensor& v_norms,
    const Tensor& page_table,
    const Tensor& cur_pos,
    const std::vector<float>& centroids,
    float scale,
    bool pre_rescaled = false,
    uint32_t num_cores_per_head = 1,
    bool return_lse = false,
    uint32_t recent_window = 0,
    const std::optional<Tensor>& k_ring = std::nullopt,
    const std::optional<Tensor>& v_ring = std::nullopt,
    const std::optional<Tensor>& ring_page_table = std::nullopt);

}  // namespace ttnn
