// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "turbo_quant.hpp"

namespace ttnn {

Tensor turbo_quant_bucketize(const Tensor& input_tensor, const std::vector<float>& boundaries) {
    return prim::turbo_quant_bucketize(input_tensor, boundaries);
}

Tensor turbo_quant_gather_centroids(const Tensor& input_tensor, const std::vector<float>& centroids) {
    return prim::turbo_quant_gather_centroids(input_tensor, centroids);
}

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
    bool pre_rescaled) {
    return ::ttnn::prim::turbo_quant_sdpa_decode(
        q, k_indices, k_norms, v_indices, v_norms, page_table, cur_pos, centroids, scale, pre_rescaled);
}

}  // namespace ttnn
