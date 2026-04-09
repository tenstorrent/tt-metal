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

}  // namespace ttnn
